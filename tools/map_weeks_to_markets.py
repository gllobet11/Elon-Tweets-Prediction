import json
import requests
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
import re

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.poly_feed import PolymarketFeed
    from config.settings import MARKET_KEYWORDS
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
POLYMARKET_BASE_URL = "https://polymarket.com"
BUILD_ID = "UdU0de4nipnMnOz1tnFxg"  # Will be updated dynamically

# Mapping of your backtest weeks to Polymarket event slugs
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

# Semanas que necesitamos encontrar los slugs (inicialmente vacías, se llenarán dinámicamente)
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


def generate_slug_variations(week_start_str: str) -> list[str]:
    """
    Genera posibles variaciones de slugs para una semana dada

    Patrones observados:
    - "elon-musk-of-tweets-september-5-12"
    - "elon-musk-of-tweets-september-19-september-26"
    - "elon-musk-of-tweets-september-23-september-30"
    """
    week_start = datetime.strptime(week_start_str, "%Y-%m-%d")
    week_end = week_start + timedelta(days=7)

    month_start = week_start.strftime("%B").lower()  # "september"
    month_end = week_end.strftime("%B").lower()
    day_start = week_start.day
    day_end = week_end.day

    variations = []

    # Patrón 1: "month-day-day" (mismo mes)
    if month_start == month_end:
        variations.append(f"elon-musk-of-tweets-{month_start}-{day_start}-{day_end}")

    # Patrón 2: "month-day-month-day" (puede cruzar meses)
    variations.append(
        f"elon-musk-of-tweets-{month_start}-{day_start}-{month_end}-{day_end}"
    )

    # Patrón 3: Con "number-of" en lugar de "of"
    if month_start == month_end:
        variations.append(
            f"elon-musk-number-of-tweets-{month_start}-{day_start}-{day_end}"
        )
    variations.append(
        f"elon-musk-number-of-tweets-{month_start}-{day_start}-{month_end}-{day_end}"
    )

    # Patrón 4: Versión corta del mes (sep, oct, nov)
    month_start_short = week_start.strftime("%b").lower()
    month_end_short = week_end.strftime("%b").lower()
    variations.append(f"elon-musk-of-tweets-{month_start_short}-{day_start}-{day_end}")

    return variations


def test_slug(slug: str, build_id: str) -> bool:
    """
    Prueba si un slug existe en Polymarket
    """
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{slug}.json"
    params = {"slug": slug}

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Verificar que tiene datos de mercado
            event_data = (
                data.get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
            )
            for query in event_data:
                if "data" in query.get("state", {}) and isinstance(
                    query["state"]["data"], dict
                ):
                    if "markets" in query["state"]["data"]:
                        return True
        return False
    except:
        return False


def find_missing_slugs(current_build_id: str) -> dict:
    """
    Encuentra los slugs correctos para las semanas faltantes
    """
    logger.info("🔍 Buscando slugs para las semanas faltantes...\n")

    found_slugs = {}

    # Hacemos una copia para no modificar el MISSING_WEEKS original durante la iteración
    weeks_to_search = list(MISSING_WEEKS.keys())

    for week_date in sorted(weeks_to_search):
        logger.info(f"--- Buscando slug para: {week_date} ---")

        variations = generate_slug_variations(week_date)
        logger.info(f"Probando {len(variations)} variaciones...")

        for i, slug in enumerate(variations, 1):
            logger.info(f"  {i}. Probando: {slug}")
            if test_slug(slug, current_build_id):
                logger.success(f"     ✅ ENCONTRADO!")
                found_slugs[week_date] = slug
                break
        else:
            logger.warning(f"  ❌ No se encontró slug válido para {week_date}")

        logger.info("")

    # Mostrar resultados
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS DE BÚSQUEDA DE SLUGS")
    logger.info("=" * 70)

    if found_slugs:
        logger.success(f"\n✅ Se encontraron {len(found_slugs)} slugs:")
        for week, slug in sorted(found_slugs.items()):
            logger.info(f'    "{week}": "{slug}",')

    not_found = set(weeks_to_search) - set(found_slugs.keys())
    if not_found:
        logger.warning(
            f"\n⚠️ No se encontraron {len(not_found)} slugs para las semanas:"
        )
        for week in sorted(not_found):
            logger.warning(f"    {week}")

    return found_slugs


def get_current_build_id() -> str:
    """
    Fetches the current Next.js BUILD_ID from the Polymarket homepage.
    """
    logger.info("Attempting to fetch current Next.js BUILD_ID...")
    try:
        response = requests.get(POLYMARKET_BASE_URL, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors

        match = re.search(r'"buildId":"([^"]+)"', response.text)
        if match:
            build_id = match.group(1)
            logger.success(f"Successfully fetched BUILD_ID: {build_id}")
            return build_id
        else:
            logger.error(
                "BUILD_ID pattern not found on Polymarket homepage. Using fallback."
            )
            return BUILD_ID  # Fallback to hardcoded ID
    except requests.RequestException as e:
        logger.error(f"Error fetching BUILD_ID: {e}. Using fallback.")
        return BUILD_ID  # Fallback to hardcoded ID


def fetch_market_data_from_nextjs(event_slug: str, build_id: str) -> dict | None:
    """
    Fetches market data from Polymarket's NextJS data endpoint

    Args:
        event_slug: The event slug (e.g., "elon-musk-of-tweets-september-5-12")
        build_id: The NextJS build ID (can be found in page source)

    Returns:
        dict: Market data including all clobTokenIds
    """
    url = f"{POLYMARKET_BASE_URL}/_next/data/{build_id}/event/{event_slug}.json"
    params = {"slug": event_slug}

    try:
        logger.info(f"Fetching data for: {event_slug}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Navigate to the market data
        event_data = (
            data.get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        )

        # Find the event query (usually the 3rd one)
        for query in event_data:
            if "data" in query.get("state", {}) and isinstance(
                query["state"]["data"], dict
            ):
                if "markets" in query["state"]["data"]:
                    return query["state"]["data"]

        logger.warning(f"Could not find market data in response for {event_slug}")
        return None

    except requests.RequestException as e:
        logger.error(f"Error fetching {event_slug}: {e}")
        return None


def extract_market_ids(event_data: dict, target_range: str = "220-239") -> str | None:
    """
    Extracts the clobTokenId for a specific outcome range

    Args:
        event_data: The event data from NextJS
        target_range: The outcome range to find (e.g., "220-239")

    Returns:
        str: The clobTokenId (first element = YES token)
    """
    markets = event_data.get("markets", [])

    for market in markets:
        title = market.get("groupItemTitle", "")
        if target_range in title:
            clob_token_ids = market.get("clobTokenIds", [])
            if clob_token_ids and len(clob_token_ids) > 0:
                # Parse the JSON string if needed
                if isinstance(clob_token_ids, str):
                    clob_token_ids = json.loads(clob_token_ids)

                # Return the first token ID (YES outcome)
                asset_id = clob_token_ids[0]
                logger.success(f"  Found asset_id for {target_range}: {asset_id}")
                return asset_id

    logger.warning(f"  Could not find market for range: {target_range}")
    return None


def map_all_weeks_to_market_ids():
    """
    Main function to map all backtest weeks to their market IDs
    """
    logger.info("🚀 Starting NextJS-based market ID extraction...")

    # Dynamically get BUILD_ID
    current_build_id = get_current_build_id()

    # Find missing slugs and update the main map
    found_missing_slugs = find_missing_slugs(current_build_id)
    WEEK_TO_SLUG_MAP.update(found_missing_slugs)

    week_to_market_map = {}

    for week_date_str, slug in WEEK_TO_SLUG_MAP.items():
        logger.info(f"\n--- Processing week: {week_date_str} ---")

        # Fetch the market data
        event_data = fetch_market_data_from_nextjs(slug, current_build_id)

        if not event_data:
            logger.warning(f"❌ Failed to fetch data for {week_date_str}")
            continue

        # Extract the asset_id for the most likely winning range
        # You can customize this based on your strategy
        # For now, let's get the "220-239" range as an example
        # TODO: Dynamically determine the target range based on y_pred for that week
        asset_id = extract_market_ids(event_data, target_range="220-239")

        if asset_id:
            week_to_market_map[week_date_str] = asset_id
            logger.success(f"✅ Mapped {week_date_str} -> {asset_id}")
        else:
            logger.warning(f"⚠️ No asset_id found for {week_date_str}")

    # Save the mapping
    project_root_dynamic = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    output_path = os.path.join(
        project_root_dynamic, "data", "markets", "week_to_market_id_map.json"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w") as f:
            json.dump(week_to_market_map, f, indent=4)
        logger.success(f"\n💾 Saved mapping to: {output_path}")
        logger.info(f"📊 Successfully mapped {len(week_to_market_map)} weeks")
    except IOError as e:
        logger.error(f"Failed to save mapping: {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MAPPING SUMMARY")
    logger.info("=" * 60)
    for week, market_id in sorted(week_to_market_map.items()):
        logger.info(f"{week} -> {market_id[:20]}...")

    unmapped_slugs = set(WEEK_TO_SLUG_MAP.keys()) - set(week_to_market_map.keys())
    if unmapped_slugs:
        logger.warning(
            f"\n⚠️ Unmapped weeks (no asset_id found after slug search): {sorted(unmapped_slugs)}"
        )

    # Generate copy-paste ready code for WEEK_TO_SLUG_MAP
    logger.info("\n" + "=" * 70)
    logger.info("CÓDIGO PARA ACTUALIZAR WEEK_TO_SLUG_MAP:")
    logger.info("=" * 70)
    logger.info("WEEK_TO_SLUG_MAP = {")
    for week in sorted(WEEK_TO_SLUG_MAP.keys()):
        logger.info(f'    "{week}": "{WEEK_TO_SLUG_MAP[week]}",')
    logger.info("}")


if __name__ == "__main__":
    map_all_weeks_to_market_ids()


def find_all_outcome_ranges(event_slug: str):
    """
    Helper function to explore all available outcome ranges for a given event
    Useful for determining which range to use in your strategy
    """
    logger.info(f"\n🔍 Exploring all outcomes for: {event_slug}")

    # Dynamically get BUILD_ID for this helper function
    current_build_id_helper = get_current_build_id()

    event_data = fetch_market_data_from_nextjs(event_slug, current_build_id_helper)

    if not event_data:
        logger.error("Failed to fetch event data")
        return

    markets = event_data.get("markets", [])
    logger.info(f"Found {len(markets)} outcome markets:\n")

    for i, market in enumerate(markets, 1):
        title = market.get("groupItemTitle", "Unknown")
        question = market.get("question", "")
        volume = market.get("volume", "0")
        clob_token_ids = market.get("clobTokenIds", [])

        if isinstance(clob_token_ids, str):
            clob_token_ids = json.loads(clob_token_ids)

        asset_id = clob_token_ids[0] if clob_token_ids else "N/A"

        logger.info(f"{i}. Range: {title}")
        logger.info(f"   Question: {question[:60]}...")
        logger.info(f"   Volume: ${float(volume):,.2f}")
        logger.info(f"   Asset ID: {asset_id}")
        logger.info("")
