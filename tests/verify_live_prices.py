"""
verify_live_prices.py

This script provides a ground truth for the live order book prices from the CLOB API.
It fetches all markets for a given slug, filters for those with an enabled order book,
and prints the bids and asks to definitively determine the sorting order.
"""

import httpx
from loguru import logger

# --- Configuration ---
# Ensure this matches the market you are debugging in the dashboard
MARKET_SLUG = "elon-musk-of-tweets-december-5-december-12"
CLOB_BASE = "https://clob.polymarket.com"

def get_all_market_definitions(slug):
    """Fetches all market objects from the CLOB API matching the base slug."""
    logger.info(f"Searching for all markets with slug containing: '{slug}'...")
    url = f"{CLOB_BASE}/markets"
    next_cursor = ""
    all_markets = []
    page = 0

    with httpx.Client(timeout=30.0) as client:
        while page < 500 and (not next_cursor or next_cursor != "LTE="):
            params = {"next_cursor": next_cursor} if next_cursor else {}
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                markets = data.get("data", [])
                if not markets:
                    break
                
                for market in markets:
                    if slug in market.get("market_slug", ""):
                        all_markets.append(market)
                
                next_cursor = data.get("next_cursor")
                page += 1

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP Error fetching markets: {e}")
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break
                
    logger.success(f"Found {len(all_markets)} total markets matching the slug.")
    return all_markets

def analyze_order_books(markets):
    """
    For a list of market definitions, finds the active ones and prints their
    order book sorting structure.
    """
    logger.info("Analyzing order books for active markets...")
    
    try:
        from py_clob_client.client import ClobClient
        client = ClobClient(CLOB_BASE)
    except ImportError:
        logger.error("Please install py_clob_client: pip install py_clob_client")
        return
    except Exception as e:
        logger.error(f"Failed to initialize ClobClient: {e}")
        return

    active_markets = [m for m in markets if m.get("enable_order_book")]
    
    if not active_markets:
        logger.warning("No active markets with enabled order books found. Cannot verify live prices.")
        return

    logger.success(f"Found {len(active_markets)} active market(s) with enabled order books. Checking each one.")
    print("=" * 80)

    for market in active_markets:
        question = market.get("question", "N/A")
        yes_token = next((t for t in market.get("tokens", []) if t.get("outcome") == "Yes"), None)

        if not yes_token:
            logger.warning(f"No 'Yes' token found for market: {question}")
            continue

        token_id = yes_token.get("token_id")
        print(f"\nMarket: {question}")
        logger.info(f"Fetching order book for token_id: {token_id}")

        try:
            ob = client.get_order_book(token_id)
            if not ob or (not ob.bids and not ob.asks):
                logger.warning("Order book is empty. No liquidity to analyze.")
                print("-" * 80)
                continue

            # --- Analyze Bids ---
            if ob.bids:
                print(f"\n--- BIDS ({len(ob.bids)} found) ---")
                print(f"  Bids[0] (First Element): {ob.bids[0].price}")
                print(f"  Bids[-1] (Last Element):  {ob.bids[-1].price}")
                if len(ob.bids) > 1:
                     is_ascending = float(ob.bids[-1].price) > float(ob.bids[0].price)
                     print(f"  Sorting Order Appears To Be: {'ASCENDING' if is_ascending else 'DESCENDING'}")
                     print(f"  Therefore, the BEST BID (highest price) is at index: {'[-1]' if is_ascending else '[0]'}")
                else:
                    print("  Only one bid, cannot determine sorting.")

            # --- Analyze Asks ---
            if ob.asks:
                print(f"\n--- ASKS ({len(ob.asks)} found) ---")
                print(f"  Asks[0] (First Element): {ob.asks[0].price}")
                print(f"  Asks[-1] (Last Element):  {ob.asks[-1].price}")
                if len(ob.asks) > 1:
                    is_ascending = float(ob.asks[-1].price) > float(ob.asks[0].price)
                    print(f"  Sorting Order Appears To Be: {'ASCENDING' if is_ascending else 'DESCENDING'}")
                    print(f"  Therefore, the BEST ASK (lowest price) is at index: {'[0]' if is_ascending else '[-1]'}")
                else:
                    print("  Only one ask, cannot determine sorting.")
            
            # --- Alternative Method ---
            try:
                print("\n--- ALTERNATIVE: get_price() ---")
                buy_price = client.get_price(token_id, side="BUY")
                sell_price = client.get_price(token_id, side="SELL")
                print(f"  get_price(side='BUY'):  {buy_price}")
                print(f"  get_price(side='SELL'): {sell_price}")
            except Exception as e:
                logger.error(f"  get_price() failed: {e}")

            print("-" * 80)

        except Exception as e:
            logger.error(f"Could not fetch or analyze order book for {question}. Error: {e}")
            print("-" * 80)


def main():
    """Main execution function."""
    all_markets = get_all_market_definitions(MARKET_SLUG)
    if all_markets:
        analyze_order_books(all_markets)

if __name__ == "__main__":
    main()
