"""
run_ingest.py

Orchestrates hybrid data ingestion:
1. Historical Tweets (Static) + New Tweets (API)
2. SpaceX Launches (LL2)
3. GDELT News (BigQuery)
4. Polymarket: Historical Structure + Incremental Price History
"""

import os
import sys
import json
import re
import requests
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

# --- Path Configuration ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ingestion.api_ingestor import ApiIngestor
    from src.ingestion.xtracker_feed import snowflake_to_dt_utc
    from src.ingestion.poly_feed import PolymarketFeed
    from config.bins_definition import BINS_ORDER
    from tools.spacex_ll2 import fetch_spacex_launches_from_ll2
    
    # Importamos la lógica de generación diaria que creamos antes
    # (Asegúrate de que tools/generate_daily_poly.py exista o integra su lógica aquí)
    from tools.generate_daily_poly import generate_daily_features as generate_daily_poly_csv
except ImportError as e:
    logger.error(f"Could not import modules. Error: {e}")
    # Si falla la importación de generate_daily_poly, definimos una dummy para no romper todo
    generate_daily_poly_csv = None

# --- Path Definitions ---
raw_dir = os.path.join(project_root, "data", "raw")
processed_dir = os.path.join(project_root, "data", "processed")
markets_dir = os.path.join(project_root, "data", "markets")
historical_base_path = os.path.join(raw_dir, "elonmusk.csv")
merged_output_path = os.path.join(processed_dir, "merged_elon_tweets.csv")
spacex_launches_output_path = os.path.join(processed_dir, "spacex_launches.csv")
elon_news_output_path = os.path.join(processed_dir, "elon_news_cleaned.csv")

# Polymarket Paths
HISTORICAL_MAP_OUTPUT_PATH = os.path.join(markets_dir, "historical_market_map.json")
PRICE_HISTORIES_OUTPUT_PATH = os.path.join(markets_dir, "all_price_histories.json")

POLYMARKET_BASE_URL = "https://polymarket.com"
DEFAULT_SPACEX_START = "2025-01-01"
DEFAULT_SPACEX_END = "2025-12-31"

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(markets_dir, exist_ok=True)


# --- Tweet Ingestion Logic ---

def load_historical_base():
    """Loads and cleans the static, long-term historical CSV."""
    if not os.path.exists(historical_base_path):
        logger.error(f"Historical base file not found: {historical_base_path}")
        return pd.DataFrame()
    logger.info(f"Loading historical base from '{os.path.basename(historical_base_path)}'...")
    try:
        df_hist = pd.read_csv(historical_base_path, sep=",", on_bad_lines="skip")
        df_hist.columns = [c.strip().lower() for c in df_hist.columns]
        if "id" not in df_hist.columns:
            logger.error("Historical base file is missing the 'id' column.")
            return pd.DataFrame()
        df_hist["created_at"] = df_hist["id"].apply(snowflake_to_dt_utc)
        df_hist = df_hist[["id", "text", "created_at"]]
        df_hist.dropna(subset=["id", "created_at"], inplace=True)
        return df_hist
    except Exception as e:
        logger.error(f"Failed to load historical base file: {e}")
        return pd.DataFrame()


# --- External Data Ingestion ---

def get_real_spacex_launches(start: str = DEFAULT_SPACEX_START, end: str = DEFAULT_SPACEX_END) -> pd.DatetimeIndex:
    try:
        launch_dates = fetch_spacex_launches_from_ll2(start=start, end=end)
        if launch_dates.empty:
            logger.warning(f"⚠️ LL2 returned 0 SpaceX launches for range {start}–{end}.")
        else:
            logger.info(f"✅ Retrieved {len(launch_dates)} SpaceX launches from LL2.")
        return launch_dates
    except Exception as e:
        logger.error(f"❌ SpaceX LL2 API Error: {e}. Returning empty DatetimeIndex.")
        return pd.DatetimeIndex([])


def fetch_gdelt_data():
    logger.info("📡 Executing BigQuery query for GDELT news data...")
    try:
        client = bigquery.Client()
        query = """
        SELECT
          PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
          COUNT(*) AS news_volume,
          AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_sentiment
        FROM
          `gdelt-bq.gdeltv2.gkg`
        WHERE
          DATE >= CAST(FORMAT_DATE('%Y%m%d000000', DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)) AS INT64)
          AND V2Persons LIKE '%Elon Musk%'
        GROUP BY
          date
        ORDER BY
          date ASC
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date", "news_volume", "avg_sentiment"])

        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(elon_news_output_path, index=False)
        logger.success(f"✅ GDELT news data saved to '{os.path.basename(elon_news_output_path)}'.")
        return df
    except Exception as e:
        logger.error(f"❌ BigQuery/GDELT API Error: {e}")
        return pd.DataFrame(columns=["date", "news_volume", "avg_sentiment"])


# --- Polymarket Logic (Optimized) ---

WEEK_TO_SLUG_MAP = {
    "2025-09-12": "elon-musk-of-tweets-september-12-september-19",
    "2025-09-19": "elon-musk-of-tweets-september-19-september-26",
    "2025-09-26": "elon-musk-of-tweets-september-26-october-3",
    "2025-10-03": "elon-musk-of-tweets-october-3-october-10",
    "2025-10-10": "elon-musk-of-tweets-october-10-october-17",
    "2025-10-17": "elon-musk-of-tweets-october-17-october-24",
    "2025-10-24": "elon-musk-of-tweets-october-24-october-31",
    "2025-10-31": "elon-musk-of-tweets-october-31-november-7",
    "2025-11-07": "elon-musk-of-tweets-november-7-november-14",
    "2025-11-14": "elon-musk-of-tweets-november-14-november-21",
    "2025-11-21": "elon-musk-of-tweets-november-21-november-28",
    "2025-11-28": "elon-musk-of-tweets-november-28-december-5",
    "2025-12-05": "elon-musk-of-tweets-december-5-december-12",
}

def get_current_build_id() -> str:
    try:
        response = requests.get(POLYMARKET_BASE_URL, timeout=10)
        response.raise_for_status()
        match = re.search(r'"buildId":"([^"]+)"', response.text)
        if match:
            return match.group(1)
    except Exception:
        pass
    logger.warning("Could not fetch BUILD_ID. Scraper might fail.")
    return ""

def fetch_market_data_from_nextjs(event_slug: str, build_id: str) -> dict | None:
    if not build_id: return None
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{event_slug}.json"
    try:
        response = requests.get(url, params={"slug": event_slug}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            queries = data.get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
            for query in queries:
                if "markets" in query.get("state", {}).get("data", {}):
                    return query["state"]["data"]
    except Exception:
        pass
    return None

def run_market_history_ingestion():
    logger.info("--- Starting Optimized Polymarket Ingestion ---")

    # 1. Load Existing Data (Cache)
    existing_map = {}
    if os.path.exists(HISTORICAL_MAP_OUTPUT_PATH):
        with open(HISTORICAL_MAP_OUTPUT_PATH, "r") as f:
            existing_map = json.load(f)
            
    existing_prices = {}
    if os.path.exists(PRICE_HISTORIES_OUTPUT_PATH):
        with open(PRICE_HISTORIES_OUTPUT_PATH, "r") as f:
            existing_prices = json.load(f)

    # 2. Identify Tasks
    build_id = get_current_build_id()
    poly_feed = PolymarketFeed()
    
    current_utc = datetime.now(timezone.utc)
    tokens_to_fetch = set()
    
    updated_map = existing_map.copy()

    for week_start_str, slug in WEEK_TO_SLUG_MAP.items():
        week_start_dt = pd.to_datetime(week_start_str).replace(tzinfo=timezone.utc)
        week_end_dt = week_start_dt + timedelta(days=7)
        
        # Determine Status
        is_past = week_end_dt < current_utc
        is_active = not is_past

        # If it's fully in the past AND we already have map AND we have prices for its bins -> SKIP
        has_metadata = week_start_str in existing_map
        has_prices = False
        if has_metadata:
             bins = existing_map[week_start_str].get("bins", {})
             # Check if we have price history for at least one bin (proxy for having data)
             if bins and any(tid in existing_prices for tid in bins.values()):
                 has_prices = True
        
        if is_past and has_metadata and has_prices:
            logger.info(f"⏭️  Skipping closed week {week_start_str} (Data exists).")
            continue
            
        logger.info(f"🔄 Processing week {week_start_str} (Active/Missing)...")
        
        # Fetch Metadata
        event_data = fetch_market_data_from_nextjs(slug, build_id)
        if not event_data or "markets" not in event_data:
            logger.warning(f"Could not fetch metadata for {slug}")
            continue

        market_bins = {}
        for market in event_data["markets"]:
            question = market.get("question", "")
            match = re.search(r"(\d+-\d+\+?|\d+\+)", question)
            if match:
                bin_label = match.group(1).replace(",", "")
                clob_ids = market.get("clobTokenIds", [])
                if isinstance(clob_ids, str): clob_ids = json.loads(clob_ids)
                if clob_ids:
                    tid = clob_ids[0]
                    market_bins[bin_label] = tid
                    tokens_to_fetch.add(tid) # Mark for fetching

        if market_bins:
             updated_map[week_start_str] = {"slug": slug, "bins": market_bins}

    # Save Map Updates
    with open(HISTORICAL_MAP_OUTPUT_PATH, "w") as f:
        json.dump(updated_map, f, indent=2)

    # 3. Fetch Prices (Only for identified tokens)
    if not tokens_to_fetch:
        logger.info("✅ No new prices needed.")
    else:
        logger.info(f"🚀 Fetching history for {len(tokens_to_fetch)} tokens...")
        
        if not poly_feed.valid:
            logger.error("Polymarket Client invalid.")
            return

        count = 0
        for tid in tokens_to_fetch:
            history = poly_feed.get_price_history(tid)
            if history:
                existing_prices[tid] = history
                count += 1
            # Simple rate limit prevention
            import time; time.sleep(0.1)

        # Save Price Updates
        with open(PRICE_HISTORIES_OUTPUT_PATH, "w") as f:
            json.dump(existing_prices, f)
        logger.success(f"✅ Updated price history for {count} tokens.")

    # 4. AUTO-GENERATE DAILY FEATURES
    if generate_daily_poly_csv:
        logger.info("⚙️ Auto-generating Daily Polymarket Features CSV...")
        try:
            generate_daily_poly_csv()
        except Exception as e:
            logger.error(f"Failed to generate daily features: {e}")
    else:
        logger.warning("⚠️ generate_daily_poly module not found. Skipping CSV generation.")


# --- Main Orchestrator ---

def run_data_ingestion():
    logger.info("--- Starting Smart Data Ingestion ---")

    # 1. Tweets
    df_hist = load_historical_base()
    if not df_hist.empty:
        last_hist_date = df_hist["created_at"].max()
        logger.info(f"Last historical tweet: {last_hist_date}")
        
        ingestor = ApiIngestor()
        df_new = ingestor.fetch(start_date=last_hist_date, end_date=datetime.now(timezone.utc))
        
        df_unified = pd.concat([df_hist, df_new], ignore_index=True)
        df_unified["created_at"] = pd.to_datetime(df_unified["created_at"], utc=True)
        df_unified = df_unified.drop_duplicates(subset=["id"], keep="last").sort_values("created_at")
        
        df_unified.to_csv(merged_output_path, index=False)
        logger.success(f"✅ Tweets saved: {len(df_unified)} records.")

    # 2. External Features
    get_real_spacex_launches()
    fetch_gdelt_data()

    # 3. Polymarket (Incremental)
    run_market_history_ingestion()

    logger.info("\n🎉 Ingestion Pipeline Complete 🎉")

if __name__ == "__main__":
    run_data_ingestion()