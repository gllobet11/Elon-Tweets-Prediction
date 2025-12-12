import json
import os
import sys

from loguru import logger

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.poly_feed import PolymarketFeed
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)


import json
import os
import sys
import re
import requests
from loguru import logger
from datetime import datetime, timedelta

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.poly_feed import PolymarketFeed
    from config.bins_definition import BINS_ORDER
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
POLYMARKET_BASE_URL = "https://polymarket.com"
WEEK_TO_SLUG_MAP_PATH = os.path.join(
    project_root, "data", "markets", "week_to_market_id_map.json"
)
HISTORICAL_MAP_OUTPUT_PATH = os.path.join(
    project_root, "data", "markets", "historical_market_map.json"
)
PRICE_HISTORIES_OUTPUT_PATH = os.path.join(
    project_root, "data", "markets", "all_price_histories.json"
)

# --- Logic from map_weeks_to_markets.py ---


def get_current_build_id() -> str:
    """Fetches the current Next.js BUILD_ID from the Polymarket homepage."""
    logger.info("Attempting to fetch current Next.js BUILD_ID...")
    try:
        response = requests.get(POLYMARKET_BASE_URL, timeout=10)
        response.raise_for_status()
        match = re.search(r'"buildId":"([^"]+)"', response.text)
        if match:
            build_id = match.group(1)
            logger.success(f"Successfully fetched BUILD_ID: {build_id}")
            return build_id
    except requests.RequestException as e:
        logger.error(f"Error fetching BUILD_ID: {e}. Exiting.")
        sys.exit(1)
    logger.error("BUILD_ID pattern not found on Polymarket homepage. Exiting.")
    sys.exit(1)


def fetch_market_data_from_nextjs(event_slug: str, build_id: str) -> dict | None:
    """Fetches market data from Polymarket's NextJS data endpoint."""
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{event_slug}.json"
    try:
        logger.debug(f"Fetching data for slug: {event_slug}")
        response = requests.get(url, params={"slug": event_slug}, timeout=10)
        if response.status_code == 404:
            logger.warning(
                f"  -> 404 Not Found for slug: {event_slug}. This week's market may have been resolved or slug is incorrect."
            )
            return None
        response.raise_for_status()
        data = response.json()
        event_data = (
            data.get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        )
        for query in event_data:
            if "data" in query.get("state", {}) and isinstance(
                query["state"]["data"], dict
            ):
                if "markets" in query["state"]["data"]:
                    return query["state"]["data"]
        logger.warning(
            f"Could not find 'markets' data in JSON response for {event_slug}"
        )
        return None
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Error fetching or parsing data for {event_slug}: {e}")
        return None


def build_and_fetch_histories():
    """
    Step 1: Builds a complete map of historical markets and their bins.
    Step 2: Fetches the price history for every single bin found.
    """
    logger.info("🚀 Starting full historical data ingestion process...")

    # --- Part 1: Build the Historical Market Map ---

    if not os.path.exists(WEEK_TO_SLUG_MAP_PATH):
        logger.error(f"Required file not found: {WEEK_TO_SLUG_MAP_PATH}")
        sys.exit(1)

    with open(WEEK_TO_SLUG_MAP_PATH, "r") as f:
        week_to_slug_map = json.load(f)

    current_build_id = get_current_build_id()
    full_historical_map = {}
    all_token_ids_to_fetch = set()

    logger.info(f"Building map for {len(week_to_slug_map)} historical weeks...")
    for week_start_str, slug in week_to_slug_map.items():
        event_data = fetch_market_data_from_nextjs(slug, current_build_id)
        if not event_data or "markets" not in event_data:
            logger.warning(
                f"Could not get data for slug {slug}, skipping week {week_start_str}."
            )
            continue

        market_bins = {}
        for market in event_data["markets"]:
            question = market.get("question", "")
            match = re.search(r"(\d+-\d+\+?|\d+\+)", question)
            if match:
                bin_label = match.group(1).replace(",", "")
                clob_token_ids_str = market.get("clobTokenIds", "[]")
                try:
                    clob_token_ids = json.loads(clob_token_ids_str)
                    if clob_token_ids:
                        yes_token_id = clob_token_ids[0]
                        market_bins[bin_label] = yes_token_id
                        all_token_ids_to_fetch.add(yes_token_id)
                except (json.JSONDecodeError, IndexError):
                    continue

        if market_bins:
            sorted_market_bins = {
                bin_label: market_bins[bin_label]
                for bin_label in BINS_ORDER
                if bin_label in market_bins
            }
            full_historical_map[week_start_str] = {
                "slug": slug,
                "bins": sorted_market_bins,
            }
            logger.success(
                f"  -> Mapped {len(sorted_market_bins)} bins for week {week_start_str}."
            )

    if not full_historical_map:
        logger.error("Failed to build historical map. No markets processed. Exiting.")
        sys.exit(1)

    with open(HISTORICAL_MAP_OUTPUT_PATH, "w") as f:
        json.dump(full_historical_map, f, indent=2)
    logger.success(
        f"✅ Step 1 complete: Full historical map saved to '{HISTORICAL_MAP_OUTPUT_PATH}'"
    )

    # --- Part 2: Fetch Price History for All Found Tokens ---

    logger.info(
        f"\n🚀 Fetching price history for {len(all_token_ids_to_fetch)} unique tokens..."
    )
    poly_feed = PolymarketFeed()
    if not poly_feed.valid:
        logger.error("Failed to initialize PolymarketFeed. Exiting.")
        sys.exit(1)

    all_price_histories = {}
    for token_id in list(all_token_ids_to_fetch):
        logger.info(f"Fetching history for Token ID: {token_id[:20]}...")
        history = poly_feed.get_price_history(token_id)
        if history:
            all_price_histories[token_id] = history
            logger.success(f"  -> Fetched {len(history)} data points.")
        else:
            logger.warning(f"  -> No history returned for token {token_id}.")
            all_price_histories[token_id] = []

    with open(PRICE_HISTORIES_OUTPUT_PATH, "w") as f:
        json.dump(all_price_histories, f)  # No indent for smaller file size
    logger.success(
        f"✅ Step 2 complete: All price histories saved to '{PRICE_HISTORIES_OUTPUT_PATH}'"
    )
    logger.info("--- Data ingestion process finished successfully! ---")


if __name__ == "__main__":
    build_and_fetch_histories()
