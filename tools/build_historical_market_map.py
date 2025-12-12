# tools/build_historical_market_map.py

import os
import sys
import json
import re
import requests  # Added requests
from loguru import logger
from datetime import datetime, timedelta  # Added timedelta for parsing slugs

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.bins_definition import BINS_ORDER
except ImportError as e:
    logger.error(f"Could not import modules. Error: {e}")
    sys.exit(1)

# --- Configuration ---
POLYMARKET_BASE_URL = "https://polymarket.com"  # Copied from map_weeks_to_markets.py
WEEK_TO_SLUG_MAP_PATH = os.path.join(
    project_root, "data", "markets", "week_to_market_id_map.json"
)
OUTPUT_MAP_PATH = os.path.join(
    project_root, "data", "markets", "historical_market_map.json"
)


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
        logger.info(f"Fetching data for slug: {event_slug}")
        response = requests.get(url, params={"slug": event_slug}, timeout=10)
        if response.status_code == 404:
            logger.warning(
                f"  -> 404 Not Found for slug: {event_slug}. This week's market may not have existed or been resolved."
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


def build_map():
    """
    Builds a comprehensive historical map of market structures using Polymarket's Next.js data endpoint.
    """
    logger.info("--- Starting Historical Market Map Builder from Next.js Endpoint ---")

    if not os.path.exists(WEEK_TO_SLUG_MAP_PATH):
        logger.error(f"Required file not found: {WEEK_TO_SLUG_MAP_PATH}")
        sys.exit(1)

    with open(WEEK_TO_SLUG_MAP_PATH, "r") as f:
        # This file actually contains week_start_date -> slug, not ID
        week_to_parent_slug = json.load(f)

    current_build_id = get_current_build_id()

    historical_market_map = {}

    logger.info(f"Processing {len(week_to_parent_slug)} historical markets...")

    for week_start_str, slug in week_to_parent_slug.items():
        logger.info(
            f"Fetching details for market slug: {slug} (Week: {week_start_str})"
        )

        event_data = fetch_market_data_from_nextjs(slug, current_build_id)

        if not event_data or "markets" not in event_data:
            logger.warning(
                f"  -> Could not retrieve full market details or tokens for slug {slug}."
            )
            continue

        market_bins = {}
        for market in event_data["markets"]:
            question = market.get("question", "")
            # Extract bin label from the question, e.g., "Elon Musk: # tweets December 5 - December 12? 140-159"
            match = re.search(r"(\d+-\d+\+?|\d+\+)", question)  # Handles "X-Y" or "Z+"
            if match:
                bin_label = match.group(1).replace(
                    ",", ""
                )  # Remove commas for consistency
                clob_token_ids_str = market.get("clobTokenIds", "[]")
                try:
                    clob_token_ids = json.loads(clob_token_ids_str)
                    if clob_token_ids and len(clob_token_ids) > 0:
                        # Assume the first token ID is the "Yes" outcome for the bin
                        market_bins[bin_label] = clob_token_ids[0]
                    else:
                        logger.warning(
                            f"  -> No CLOB token IDs found for bin '{bin_label}' in market '{question}'"
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"  -> Could not parse clobTokenIds for market '{question}'"
                    )

        if market_bins:
            # Sort the bins according to the project's canonical order (BINS_ORDER)
            sorted_market_bins = {
                bin_label: market_bins[bin_label]
                for bin_label in BINS_ORDER
                if bin_label in market_bins
            }
            if len(sorted_market_bins) == len(
                BINS_ORDER
            ):  # Check if all expected bins are present
                historical_market_map[week_start_str] = {
                    "slug": slug,
                    "bins": sorted_market_bins,
                }
                logger.success(
                    f"  -> Successfully mapped {len(sorted_market_bins)} bins for week {week_start_str}."
                )
            else:
                logger.warning(
                    f"  -> Incomplete bins mapped for week {week_start_str}. Expected {len(BINS_ORDER)}, found {len(sorted_market_bins)}. Skipping this week."
                )
        else:
            logger.warning(
                f"  -> No valid bins extracted for slug {slug} for week {week_start_str}."
            )

    if historical_market_map:
        with open(OUTPUT_MAP_PATH, "w") as f:
            json.dump(historical_market_map, f, indent=2)
        logger.success(
            f"--- Full Historical Market Map created successfully at: {OUTPUT_MAP_PATH} ---"
        )
    else:
        logger.error(
            "--- Failed to create full historical market map. No markets were processed. ---"
        )


if __name__ == "__main__":
    build_map()
