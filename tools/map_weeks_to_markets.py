import json
import requests
import os
import sys
import re
from datetime import datetime, timedelta
from loguru import logger

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Import the feed to fetch prices later
    from src.ingestion.poly_feed import PolymarketFeed
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
POLYMARKET_BASE_URL = "https://polymarket.com"

# Final output files
HISTORICAL_MAP_OUTPUT_PATH = os.path.join(
    project_root, "data", "markets", "historical_market_map.json"
)
PRICE_HISTORIES_OUTPUT_PATH = os.path.join(
    project_root, "data", "markets", "all_price_histories.json"
)

# --- 1. SLUG MAPPING LOGIC (Preserved from your script) ---

# Initial known map (can be partial)
WEEK_TO_SLUG_MAP = {
    "2025-09-12": "elon-musk-of-tweets-september-12-september-19",
    "2025-09-19": "elon-musk-of-tweets-september-19-september-26",
    "2025-09-26": "elon-musk-of-tweets-september-23-september-30",
    "2025-10-03": "elon-musk-of-tweets-october-3-october-10",
    "2025-10-10": "elon-musk-of-tweets-october-10-october-17",
    "2025-10-24": "elon-musk-of-tweets-october-24-october-31",
    "2025-10-31": "elon-musk-of-tweets-october-31-november-7",
    "2025-11-07": "elon-musk-of-tweets-november-7-november-14",
    "2025-11-14": "elon-musk-of-tweets-november-14-november-21",
    "2025-11-21": "elon-musk-of-tweets-november-21-november-28",
    "2025-11-28": "elon-musk-of-tweets-november-28-december-5",
}

# Weeks we suspect might be wrong or missing, to trigger the search logic
MISSING_WEEKS = {
    "2025-09-12": None,
    "2025-10-03": None,
    "2025-10-10": None,
    "2025-10-17": None,
    "2025-10-24": None,
    "2025-11-07": None,
    "2025-11-14": None,
    "2025-11-21": None,
}


def get_current_build_id() -> str:
    """Fetches the current Next.js BUILD_ID."""
    logger.info("Attempting to fetch current Next.js BUILD_ID...")
    try:
        response = requests.get(POLYMARKET_BASE_URL, timeout=10)
        response.raise_for_status()
        match = re.search(r'"buildId":"([^"]+)"', response.text)
        if match:
            build_id = match.group(1)
            logger.success(f"Successfully fetched BUILD_ID: {build_id}")
            return build_id
    except Exception as e:
        logger.error(f"Error fetching BUILD_ID: {e}")
    # Fallback (from your script)
    return "UdU0de4nipnMnOz1tnFxg"


def generate_slug_variations(week_start_str: str) -> list[str]:
    """Generates possible URL slugs for a given date."""
    week_start = datetime.strptime(week_start_str, "%Y-%m-%d")
    week_end = week_start + timedelta(days=7)
    ms, me = week_start.strftime("%B").lower(), week_end.strftime("%B").lower()
    ds, de = week_start.day, week_end.day

    variations = []
    if ms == me:
        variations.append(f"elon-musk-of-tweets-{ms}-{ds}-{de}")
    variations.append(f"elon-musk-of-tweets-{ms}-{ds}-{me}-{de}")
    if ms == me:
        variations.append(f"elon-musk-number-of-tweets-{ms}-{ds}-{de}")
    variations.append(f"elon-musk-number-of-tweets-{ms}-{ds}-{me}-{de}")

    ms_short, me_short = (
        week_start.strftime("%b").lower(),
        week_end.strftime("%b").lower(),
    )
    variations.append(f"elon-musk-of-tweets-{ms_short}-{ds}-{de}")

    return variations


def test_slug(slug: str, build_id: str) -> bool:
    """Checks if a slug yields a valid 200 OK with market data."""
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{slug}.json"
    try:
        r = requests.get(url, params={"slug": slug}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # Simple check for 'markets' key deeper in the structure
            txt = json.dumps(data)
            return '"markets":' in txt
        return False
    except:
        return False


def find_missing_slugs(current_build_id: str):
    """Updates WEEK_TO_SLUG_MAP with any found missing slugs."""
    logger.info("🔍 Checking for better slugs for marked weeks...")
    for week in MISSING_WEEKS:
        variations = generate_slug_variations(week)
        for slug in variations:
            if test_slug(slug, current_build_id):
                logger.success(f" ✅ Found slug for {week}: {slug}")
                WEEK_TO_SLUG_MAP[week] = slug
                break


# --- 2. DATA EXTRACTION LOGIC (The New Part) ---


def fetch_market_data(event_slug: str, build_id: str) -> dict | None:
    """Fetches the raw JSON for the event."""
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{event_slug}.json"
    try:
        r = requests.get(url, params={"slug": event_slug}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # Deep traverse to find the queries array
        queries = (
            data.get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        )
        for q in queries:
            if "markets" in q.get("state", {}).get("data", {}):
                return q["state"]["data"]
        return None
    except Exception as e:
        logger.error(f"Error fetching {event_slug}: {e}")
        return None


def extract_all_bins(event_data: dict) -> dict:
    """
    Parses the event data and extracts EVERY bin range and its YES token ID.
    Returns: {"100-119": "0x123...", "120-139": "0x456..."}
    """
    bins_map = {}
    markets = event_data.get("markets", [])

    for market in markets:
        question = market.get("question", "") or market.get("groupItemTitle", "")

        # Regex to find patterns like "200-219" or "240+"
        # We strip commas to handle "1,000+" cases if they exist
        match = re.search(r"(\d+-\d+\+?|\d+\+)", question.replace(",", ""))

        if match:
            bin_label = match.group(1)

            # Polymarket stores IDs in 'clobTokenIds' as a JSON string or list
            raw_ids = market.get("clobTokenIds", "[]")
            if isinstance(raw_ids, str):
                try:
                    ids = json.loads(raw_ids)
                except:
                    ids = []
            else:
                ids = raw_ids

            # The first ID is typically the "YES" token
            if ids and len(ids) > 0:
                bins_map[bin_label] = ids[0]

    return bins_map


# --- 3. MAIN EXECUTION ---


def main():
    logger.info("🚀 Starting Full Market History Build...")

    # A. Setup
    build_id = get_current_build_id()
    find_missing_slugs(build_id)  # Ensure our map is as complete as possible

    full_historical_map = {}
    all_token_ids = set()

    # B. Build the Map
    logger.info(f"\nProcessing {len(WEEK_TO_SLUG_MAP)} weeks...")

    for week_date, slug in WEEK_TO_SLUG_MAP.items():
        logger.info(f"Processing {week_date}...")
        event_data = fetch_market_data(slug, build_id)

        if not event_data:
            logger.warning(f" ⚠️ No data found for {week_date} (slug: {slug})")
            continue

        bins = extract_all_bins(event_data)

        if bins:
            full_historical_map[week_date] = {"slug": slug, "bins": bins}
            # Collect IDs for the next step
            for token_id in bins.values():
                all_token_ids.add(token_id)
            logger.success(f"  -> Mapped {len(bins)} bins.")
        else:
            logger.warning(f"  -> Data found, but no bins extracted for {week_date}.")

    # Save the map
    with open(HISTORICAL_MAP_OUTPUT_PATH, "w") as f:
        json.dump(full_historical_map, f, indent=2)
    logger.success(f"\n✅ Market Map saved to: {HISTORICAL_MAP_OUTPUT_PATH}")

    # C. Fetch Prices
    logger.info(
        f"\n🚀 Fetching Price History for {len(all_token_ids)} unique tokens..."
    )
    logger.info("This may take a minute...")

    poly_feed = PolymarketFeed()
    price_db = {}

    count = 0
    for token_id in list(all_token_ids):
        count += 1
        # Simple progress logger
        if count % 10 == 0:
            logger.info(f"Fetching {count}/{len(all_token_ids)}...")

        history = poly_feed.get_price_history(token_id)
        if history:
            price_db[token_id] = history
        else:
            price_db[token_id] = []  # Save empty list to prevent re-fetching errors

    # Save the prices
    with open(PRICE_HISTORIES_OUTPUT_PATH, "w") as f:
        json.dump(price_db, f)  # No indent to save space

    logger.success(f"✅ All Price Histories saved to: {PRICE_HISTORIES_OUTPUT_PATH}")
    logger.info("Process Complete. You can now run the backtester.")


if __name__ == "__main__":
    main()
