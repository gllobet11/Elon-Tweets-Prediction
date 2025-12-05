import httpx
from loguru import logger

# --- Configuration ---
CLOB_API_URL = "https://clob.polymarket.com"
KEYWORDS = ["elon musk", "tweet"]


def find_active_market():
    """
    Scans the Polymarket API for an active market matching the keywords.
    """
    logger.info("Attempting to find an active market matching keywords...")
    markets_url = f"{CLOB_API_URL}/markets"

    try:
        response = httpx.get(markets_url, params={"next_cursor": ""})
        response.raise_for_status()
        data = response.json()

        for market in data.get("data", []):
            question = market.get("question", "").lower()
            is_active = market.get("active", False)

            if all(keyword in question for keyword in KEYWORDS) and is_active:
                logger.success(f"Found an active market: '{market.get('question')}'")
                logger.info(f" -> Product ID: {market.get('product_id')}")
                return market.get("product_id")

        logger.warning("No active markets found in the first page of API results.")
        return None

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error while fetching markets: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching markets: {e}")
        return None


def fetch_price_history(product_id: int):
    """
    Fetches the historical prices for a given product_id.
    """
    if not product_id:
        logger.error("Cannot fetch history without a valid product_id.")
        return

    logger.info(f"Fetching price history for Product ID: {product_id}...")
    history_url = f"{CLOB_API_URL}/prices-history"

    # Parameters for the request, including the product_id and a timeframe
    # The API seems to require a timeframe, let's ask for the last 30 days.
    params = {
        "product_id": product_id,
        "timeframe": "30d",  # Options might be 1d, 7d, 30d, all
    }

    try:
        response = httpx.get(history_url, params=params)
        response.raise_for_status()
        history_data = response.json()

        if history_data:
            logger.success("Successfully fetched price history!")
            # Print a sample of the data
            sample_size = min(5, len(history_data))
            logger.info(f"Sample of the first {sample_size} data points:")
            for record in history_data[:sample_size]:
                logger.info(
                    f"  - Timestamp: {record.get('timestamp')}, Price: {record.get('price')}"
                )

            # Save the full response to a file for inspection
            with open("historical_prices_sample.json", "w") as f:
                import json

                json.dump(history_data, f, indent=2)
            logger.info("Full historical data saved to 'historical_prices_sample.json'")

        else:
            logger.warning("API returned an empty list for historical prices.")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error while fetching price history: {e}")
        logger.error(f"Response body: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching price history: {e}")


if __name__ == "__main__":
    active_product_id = find_active_market()
    if active_product_id:
        fetch_price_history(active_product_id)
    else:
        logger.error(
            "Could not find an active market to test the price history endpoint."
        )
