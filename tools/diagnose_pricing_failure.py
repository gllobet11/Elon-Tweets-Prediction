"""
diagnose_pricing_failure.py

Script para diagnosticar EXACTAMENTE dónde falla get_market_valuation()
Usa la CLOB API directamente y prueba token_id DECIMAL (sin hex).
"""

from py_clob_client.client import ClobClient
import httpx

CLOB_BASE = "https://clob.polymarket.com"
MARKET_SLUG = "elon-musk-of-tweets-december-9-december-16"


def test_full_pipeline():
    """Simular el flujo completo del dashboard."""
    print("=" * 80)
    print("FULL PIPELINE DIAGNOSTIC")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # 1. Fetch markets from CLOB
    # ------------------------------------------------------------------ #
    print("\n📊 STEP 1: Fetching markets from CLOB...")
    url = f"{CLOB_BASE}/markets"
    markets_found: list[dict] = []

    with httpx.Client(timeout=30.0) as c:
        next_cursor = ""
        for _ in range(300):
            params = {"next_cursor": next_cursor} if next_cursor else {}
            r = c.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            for m in data.get("data", []):
                if MARKET_SLUG in (m.get("market_slug") or ""):
                    markets_found.append(m)
                    if len(markets_found) >= 3:  # Solo primeros 3 para test
                        break

            if len(markets_found) >= 3:
                break

            next_cursor = data.get("next_cursor")
            if not next_cursor or next_cursor == "LTE=":
                break

    print(f"  ✅ Found {len(markets_found)} markets")

    if not markets_found:
        print("  ❌ No markets found! Check MARKET_SLUG")
        return

    # ------------------------------------------------------------------ #
    # 2. Extract token IDs from first market
    # ------------------------------------------------------------------ #
    print("\n📊 STEP 2: Extracting token IDs...")
    first_market = markets_found[0]

    print(f"  Market: {first_market.get('question', '')[:60]}...")
    print(f"  Active: {first_market.get('active')}")
    print(f"  Closed: {first_market.get('closed')}")
    print(f"  Enable Order Book: {first_market.get('enable_order_book')}")

    tokens = first_market.get("tokens") or []
    print(f"  Tokens count: {len(tokens)}")

    yes_token_id = None
    no_token_id = None

    for token in tokens:
        print("\n  Token:")
        print(f"    Outcome: {token.get('outcome')}")
        print(f"    Token ID: {token.get('token_id')}")
        print(f"    Token ID type: {type(token.get('token_id'))}")

        if token.get("outcome") == "Yes":
            yes_token_id = token.get("token_id")
        elif token.get("outcome") == "No":
            no_token_id = token.get("token_id")

    if not yes_token_id or not no_token_id:
        print("  ❌ Could not extract token IDs!")
        return

    print(f"\n  ✅ YES token ID: {yes_token_id}")
    print(f"  ✅ NO token ID: {no_token_id}")

    # ------------------------------------------------------------------ #
    # 3. Test get_price directly with DECIMAL token_id
    # ------------------------------------------------------------------ #
    print("\n📊 STEP 3: Testing get_price() directly (decimal token_id)...")

    client = ClobClient(CLOB_BASE)

    price_token_id = str(yes_token_id)
    print(f"\n  Testing YES token (decimal ID): {price_token_id}")
    try:
        buy_res = client.get_price(price_token_id, side="BUY")
        sell_res = client.get_price(price_token_id, side="SELL")

        print(f"    BUY result: {buy_res}")
        print(f"    SELL result: {sell_res}")

        if buy_res and sell_res:
            bid = float(buy_res.get("price", "0"))
            ask = float(sell_res.get("price", "1"))

            print(f"    ✅ Bid: {bid:.4f}")
            print(f"    ✅ Ask: {ask:.4f}")
            print(f"    ✅ Mid: {(bid + ask) / 2:.4f}")

            # Verificación de condiciones de liquidez “sana”
            print("\n    Condition checks:")
            print(f"      bid > 0: {bid > 0}")
            print(f"      ask < 1.0: {ask < 1.0}")
            print(f"      ask > bid: {ask > bid}")

            if bid > 0 and ask < 1.0 and ask > bid:
                print("    ✅ All conditions pass - price should work!")
            else:
                print("    ❌ Condition failed - your logic podría devolver NO_LIQUIDITY")
        else:
            print("    ❌ get_price returned None!")

    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

    # ------------------------------------------------------------------ #
    # 4. Redundant check: same token_id again (confirma que no hace falta hex)
    # ------------------------------------------------------------------ #
    print(f"\n  Testing YES token AGAIN (decimal ID, no hex): {yes_token_id}")
    try:
        buy_res_raw = client.get_price(yes_token_id, side="BUY")
        sell_res_raw = client.get_price(yes_token_id, side="SELL")

        print(f"    BUY result: {buy_res_raw}")
        print(f"    SELL result: {sell_res_raw}")

        if buy_res_raw and sell_res_raw:
            print("    ✅ Works WITHOUT any hex conversion!")
        else:
            print("    ❌ Also fails without hex (no orderbook / no liquidity)")

    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ------------------------------------------------------------------ #
    # 5. Test multiple markets del mismo slug
    # ------------------------------------------------------------------ #
    print("\n📊 STEP 4: Testing multiple markets...")

    for i, market in enumerate(markets_found[:3], 1):
        print(f"\n  Market {i}: {market.get('question', '')[:50]}...")
        print(f"    Active: {market.get('active')}, Closed: {market.get('closed')}, "
              f"Enable Order Book: {market.get('enable_order_book')}")

        tokens = market.get("tokens") or []
        yes_t = next((t for t in tokens if t.get("outcome") == "Yes"), None)

        if yes_t:
            token_id = yes_t.get("token_id")
            print(f"    YES token_id: {token_id}")
            try:
                result = client.get_price(token_id, side="BUY")
                if result:
                    print(f"    ✅ get_price works! Price: {result.get('price')}")
                else:
                    print("    ❌ get_price returned None (posible sin liquidez)")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            print("    ❌ No YES token found in this market")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PRICING FAILURE DIAGNOSTIC" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")

    test_full_pipeline()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(
        """
    Si STEP 3 muestra precios correctos con el token_id decimal:
      → El problema está en cómo tu código llama a get_market_valuation()
      → Verifica que los token IDs se están pasando como strings decimales, sin hex.

    Si STEP 3 falla:
      → El problema está en get_price() o en el propio mercado (sin orderbook).
      → Asegúrate de NO convertir a hex y de que el market tenga enable_order_book=True.

    Si algunos markets funcionan y otros no:
      → Es normal: algunos bins no tienen orderbook / liquidez.
      → Filtra por active=True y enable_order_book=True en tu lógica de discovery.
    """
    )


if __name__ == "__main__":
    main()
