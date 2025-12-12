"""
test_get_price_method.py

Verificar que el método get_price() funciona correctamente
y devuelve la estructura esperada.
"""

from py_clob_client.client import ClobClient

# Token del bin 500+ que sabemos que funciona
TOKEN_500_PLUS = "82290494205126596221932116595597359864571478478999982656316176190479156221288"


def test_get_price():
    """Test directo del método get_price()"""
    print("=" * 80)
    print("TESTING get_price() METHOD")
    print("=" * 80)
    
    client = ClobClient("https://clob.polymarket.com")
    
    print(f"\n🔍 Testing token: {TOKEN_500_PLUS}")
    
    # Test BUY side
    print(f"\n📊 Testing BUY side:")
    try:
        buy_result = client.get_price(TOKEN_500_PLUS, side="BUY")
        print(f"  Type: {type(buy_result)}")
        print(f"  Result: {buy_result}")
        
        if isinstance(buy_result, dict):
            print(f"  Keys: {list(buy_result.keys())}")
            if "price" in buy_result:
                print(f"  ✅ Price found: {buy_result['price']}")
            else:
                print(f"  ❌ No 'price' key in result!")
        else:
            print(f"  ⚠️  Result is not a dict!")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test SELL side
    print(f"\n📊 Testing SELL side:")
    try:
        sell_result = client.get_price(TOKEN_500_PLUS, side="SELL")
        print(f"  Type: {type(sell_result)}")
        print(f"  Result: {sell_result}")
        
        if isinstance(sell_result, dict):
            print(f"  Keys: {list(sell_result.keys())}")
            if "price" in sell_result:
                print(f"  ✅ Price found: {sell_result['price']}")
            else:
                print(f"  ❌ No 'price' key in result!")
        else:
            print(f"  ⚠️  Result is not a dict!")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Compare with order book
    print(f"\n📊 Comparing with get_order_book():")
    try:
        ob = client.get_order_book(TOKEN_500_PLUS)
        
        if ob and ob.bids and ob.asks:
            best_bid_ob = float(ob.bids[-1].price)
            best_ask_ob = float(ob.asks[-1].price)
            
            print(f"  Order Book:")
            print(f"    Best Bid: {best_bid_ob:.4f}")
            print(f"    Best Ask: {best_ask_ob:.4f}")
            print(f"    Mid: {(best_bid_ob + best_ask_ob) / 2:.4f}")
            
            # Compare
            if buy_result and isinstance(buy_result, dict) and "price" in buy_result:
                buy_price = float(buy_result["price"])
                print(f"\n  get_price(BUY): {buy_price:.4f}")
                print(f"  Matches best_ask? {abs(buy_price - best_ask_ob) < 0.01}")
            
            if sell_result and isinstance(sell_result, dict) and "price" in sell_result:
                sell_price = float(sell_result["price"])
                print(f"  get_price(SELL): {sell_price:.4f}")
                print(f"  Matches best_bid? {abs(sell_price - best_bid_ob) < 0.01}")
                
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "GET_PRICE TEST" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    
    test_get_price()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    Si get_price() funciona correctamente:
      ✅ Devuelve un dict con key "price"
      ✅ BUY price ≈ best_ask (precio para comprar YES)
      ✅ SELL price ≈ best_bid (precio para vender YES)
    
    Si falla:
      ❌ Devuelve None o estructura diferente
      ❌ Necesitamos volver a usar get_order_book()
    """)


if __name__ == "__main__":
    main()