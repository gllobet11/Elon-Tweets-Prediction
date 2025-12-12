"""
test_gamma_comprehensive.py

Probar TODAS las formas posibles de buscar en Gamma API
para ver si podemos encontrar los mercados de Elon.
"""

import httpx
import json

GAMMA_BASE = "https://gamma-api.polymarket.com"


def test_gamma_markets_endpoint():
    """Test 1: Endpoint /markets con diferentes filtros"""
    print("=" * 80)
    print("TEST 1: GAMMA /markets ENDPOINT")
    print("=" * 80)
    
    url = f"{GAMMA_BASE}/markets"
    
    # Diferentes estrategias de búsqueda
    strategies = [
        {"limit": 100},
        {"limit": 100, "closed": "false"},
        {"limit": 100, "active": "true"},
        {"limit": 100, "closed": "false", "active": "true"},
        # Intentar con offset/paginación
        {"limit": 100, "offset": 0},
        {"limit": 100, "offset": 1000},
        {"limit": 100, "offset": 5000},
    ]
    
    with httpx.Client(timeout=30.0) as c:
        for i, params in enumerate(strategies, 1):
            print(f"\n🔍 Strategy {i}: {params}")
            
            try:
                r = c.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                
                if isinstance(data, list):
                    print(f"  ✅ Got {len(data)} markets")
                    
                    # Buscar Elon
                    elon_markets = [m for m in data if 'elon' in str(m.get('question', '')).lower()]
                    print(f"  🎯 Elon markets: {len(elon_markets)}")
                    
                    if elon_markets:
                        for m in elon_markets[:2]:
                            print(f"    - {m.get('question')[:80]}...")
                            print(f"      Slug: {m.get('slug')}")
                else:
                    print(f"  ⚠️ Unexpected response type: {type(data)}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")


def test_gamma_events_endpoint():
    """Test 2: Endpoint /events con diferentes filtros"""
    print("\n" + "=" * 80)
    print("TEST 2: GAMMA /events ENDPOINT")
    print("=" * 80)
    
    url = f"{GAMMA_BASE}/events"
    
    strategies = [
        {"limit": 100},
        {"limit": 100, "closed": "false"},
        {"limit": 100, "active": "true"},
        {"limit": 100, "offset": 0},
        {"limit": 100, "offset": 1000},
    ]
    
    with httpx.Client(timeout=30.0) as c:
        for i, params in enumerate(strategies, 1):
            print(f"\n🔍 Strategy {i}: {params}")
            
            try:
                r = c.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                
                if isinstance(data, list):
                    print(f"  ✅ Got {len(data)} events")
                    
                    # Buscar Elon
                    elon_events = [e for e in data if 'elon' in str(e.get('title', '')).lower()]
                    print(f"  🎯 Elon events: {len(elon_events)}")
                    
                    if elon_events:
                        for e in elon_events[:2]:
                            print(f"    - {e.get('title')}")
                            if 'markets' in e:
                                print(f"      Markets: {len(e.get('markets', []))}")
                else:
                    print(f"  ⚠️ Unexpected response type: {type(data)}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")


def test_gamma_search_by_slug():
    """Test 3: Buscar por slug específico"""
    print("\n" + "=" * 80)
    print("TEST 3: SEARCH BY SPECIFIC SLUG")
    print("=" * 80)
    
    slugs_to_try = [
        "elon-musk-of-tweets-december-9-december-16",
        "elon-musk-tweets",
        "elon-musk",
    ]
    
    with httpx.Client(timeout=30.0) as c:
        for slug in slugs_to_try:
            print(f"\n🔍 Trying slug: '{slug}'")
            
            # Intentar en /markets
            try:
                url = f"{GAMMA_BASE}/markets/{slug}"
                r = c.get(url)
                if r.status_code == 200:
                    data = r.json()
                    print(f"  ✅ Found in /markets/{slug}")
                    print(f"     Question: {data.get('question', 'N/A')[:80]}")
                else:
                    print(f"  ❌ Not found in /markets/{slug} (status {r.status_code})")
            except Exception as e:
                print(f"  ❌ /markets/{slug} error: {e}")
            
            # Intentar en /events
            try:
                url = f"{GAMMA_BASE}/events/{slug}"
                r = c.get(url)
                if r.status_code == 200:
                    data = r.json()
                    print(f"  ✅ Found in /events/{slug}")
                    print(f"     Title: {data.get('title', 'N/A')[:80]}")
                else:
                    print(f"  ❌ Not found in /events/{slug} (status {r.status_code})")
            except Exception as e:
                print(f"  ❌ /events/{slug} error: {e}")


def test_gamma_with_condition_id():
    """Test 4: Buscar por condition_id si lo tenemos de CLOB"""
    print("\n" + "=" * 80)
    print("TEST 4: SEARCH BY CONDITION_ID")
    print("=" * 80)
    
    # Este condition_id lo obtuvimos de CLOB en tests anteriores
    condition_id = "0x800d8919a5ea55e88699c50f612d25a5524b4bdbd9c1ca3458da489ef8602851"
    
    print(f"🔍 Using condition_id from CLOB: {condition_id}")
    
    with httpx.Client(timeout=30.0) as c:
        # Intentar diferentes endpoints
        endpoints = [
            f"/markets?condition_id={condition_id}",
            f"/events?condition_id={condition_id}",
        ]
        
        for endpoint in endpoints:
            print(f"\n  Testing: {endpoint}")
            try:
                url = f"{GAMMA_BASE}{endpoint}"
                r = c.get(url)
                r.raise_for_status()
                data = r.json()
                
                if data:
                    print(f"    ✅ Found data!")
                    print(f"    Type: {type(data)}")
                    if isinstance(data, list):
                        print(f"    Items: {len(data)}")
                    elif isinstance(data, dict):
                        print(f"    Keys: {list(data.keys())[:5]}")
                else:
                    print(f"    ⚠️ Empty response")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")


def compare_clob_vs_gamma():
    """Test 5: Comparar qué devuelve cada API"""
    print("\n" + "=" * 80)
    print("TEST 5: CLOB vs GAMMA COMPARISON")
    print("=" * 80)
    
    slug = "elon-musk-of-tweets-december-9-december-16-500plus"
    
    print(f"\n🔍 Searching for: {slug}")
    
    with httpx.Client(timeout=30.0) as c:
        # 1. CLOB API
        print(f"\n📊 CLOB API:")
        try:
            clob_url = f"https://clob.polymarket.com/markets"
            
            # Buscar el mercado
            next_cursor = ""
            for page in range(300):
                params = {"next_cursor": next_cursor} if next_cursor else {}
                r = c.get(clob_url, params=params)
                data = r.json()
                
                for m in data.get("data", []):
                    if slug in m.get("market_slug", ""):
                        print(f"  ✅ Found in CLOB at page {page}")
                        print(f"     Question: {m.get('question')}")
                        print(f"     Token IDs available: {len(m.get('tokens', []))}")
                        
                        # Guardar para comparar
                        clob_market = m
                        break
                else:
                    next_cursor = data.get("next_cursor")
                    if not next_cursor or next_cursor == "LTE=":
                        break
                    continue
                break
            else:
                print(f"  ❌ Not found in CLOB after 300 pages")
                clob_market = None
                
        except Exception as e:
            print(f"  ❌ CLOB error: {e}")
            clob_market = None
        
        # 2. Gamma API
        print(f"\n📊 Gamma API:")
        try:
            # Intentar /markets con filtros
            gamma_url = f"{GAMMA_BASE}/markets"
            params = {"limit": 1000, "closed": "false"}
            
            r = c.get(gamma_url, params=params)
            data = r.json()
            
            gamma_market = None
            if isinstance(data, list):
                for m in data:
                    if slug in str(m.get('slug', '')):
                        print(f"  ✅ Found in Gamma!")
                        print(f"     Question: {m.get('question')}")
                        gamma_market = m
                        break
                
                if not gamma_market:
                    print(f"  ❌ Not found in Gamma (checked {len(data)} markets)")
            else:
                print(f"  ⚠️ Unexpected response type")
                
        except Exception as e:
            print(f"  ❌ Gamma error: {e}")
            gamma_market = None
        
        # Comparación
        print(f"\n📋 COMPARISON:")
        print(f"  CLOB has market: {'✅' if clob_market else '❌'}")
        print(f"  Gamma has market: {'✅' if gamma_market else '❌'}")
        
        if clob_market and not gamma_market:
            print(f"\n  💡 CONCLUSION: Gamma API doesn't index these markets yet")
            print(f"     CLOB is the only source available")
            print(f"     Must use CLOB pagination (slow but necessary)")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "GAMMA API COMPREHENSIVE TEST" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run all tests
    test_gamma_markets_endpoint()
    test_gamma_events_endpoint()
    test_gamma_search_by_slug()
    test_gamma_with_condition_id()
    compare_clob_vs_gamma()
    
    print("\n" + "=" * 80)
    print("FINAL CONCLUSION")
    print("=" * 80)
    print("""
    Si Gamma API tiene los mercados:
      ✅ Usar Gamma para buscar (1 segundo)
      ✅ Obtener token_ids de Gamma
      ✅ Usar CLOB solo para precios (get_order_book)
      ⚡ Total: ~2-3 segundos
    
    Si Gamma NO tiene los mercados:
      ❌ No podemos usar Gamma
      ✅ Debemos usar CLOB completo (paginar todo)
      🐌 Total: 40-50 segundos
      
    En el caso de Elon Musk tweets Diciembre 2024:
      Gamma parece NO tener estos mercados indexados aún.
      Por eso necesitamos escanear CLOB (no hay alternativa).
    """)


if __name__ == "__main__":
    main()