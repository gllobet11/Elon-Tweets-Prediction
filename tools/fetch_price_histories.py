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


def fetch_and_save_price_histories():
    """
    Loads the week-to-market-ID map, fetches the price history for each
    market ID, and saves all histories to a single JSON file.
    """
    logger.info("🚀 Starting the process to fetch and save historical market prices...")

    # 1. Load the market ID map
    map_path = os.path.join(
        project_root, "data", "markets", "week_to_market_id_map.json"
    )
    if not os.path.exists(map_path):
        logger.error(f"Market ID map not found at '{map_path}'.")
        logger.error("Please run `tools/map_weeks_to_markets.py` first.")
        return

    try:
        with open(map_path, "r") as f:
            week_to_market_map = json.load(f)
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from '{map_path}'. The file might be empty or corrupt."
        )
        return

    if not week_to_market_map:
        logger.warning("The week-to-market-ID map is empty. Nothing to fetch.")
        return

    logger.info(f"Found {len(week_to_market_map)} markets to fetch price history for.")

    # 2. Instantiate PolymarketFeed
    poly_feed = PolymarketFeed()
    if not poly_feed.valid:
        logger.error("Failed to initialize PolymarketFeed. Exiting.")
        return

    # 3. Fetch price history for each market ID
    all_price_histories = {}
    for week, market_id in week_to_market_map.items():
        if not market_id:
            logger.warning(f"Skipping week {week} due to a null market ID.")
            continue

        logger.info(
            f"Fetching price history for week {week}, Market ID: {market_id}..."
        )
        history = poly_feed.get_price_history(market_id)
        if history:
            all_price_histories[market_id] = history
            logger.success(f"  [✅] Successfully fetched {len(history)} data points.")
        else:
            logger.warning(
                f"  [⚠️] No price history returned for market ID {market_id}."
            )
            all_price_histories[
                market_id
            ] = []  # Save empty list to indicate an attempt was made

    # 4. Save all histories to a single file
    output_path = os.path.join(
        project_root, "data", "markets", "all_price_histories.json"
    )
    try:
        with open(output_path, "w") as f:
            json.dump(all_price_histories, f, indent=4)
        logger.success(f"💾 All price histories saved successfully to '{output_path}'")
    except IOError as e:
        logger.error(f"Failed to save price histories file: {e}")


if __name__ == "__main__":
    fetch_and_save_price_histories()
