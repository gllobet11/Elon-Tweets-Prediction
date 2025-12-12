import time
import re
import httpx
from loguru import logger
from py_clob_client.client import ClobClient
from py_clob_client.exceptions import PolyApiException

class PolymarketFeed:
    """
    Versión FINAL CORREGIDA:
    - Usa get_price() correctamente (convierte string a float)
    - Cache de mercados
    - Early exit optimization
    """
    def __init__(self, host="https://clob.polymarket.com", chain_id=137):
        self.valid = False
        self.clob_base = host
        try:
            self.client = ClobClient(host, chain_id=chain_id)
            self.valid = True
            logger.success("✅ ClobClient initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing ClobClient: {e}")
            self.client = None
        
        self._markets_cache = {}

    def _robust_api_call(self, api_func, *args, retries=3, delay=2, **kwargs):
        for attempt in range(retries):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, PolyApiException) and e.status_code == 404:
                    return None
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logger.warning(f"API call failed after {retries} retries: {e}")
                    return None

    def _to_hex_token_id(self, token_id_str: str) -> str:
        """Convierte Asset ID numérico a Hex para la API del CLOB."""
        if not token_id_str: return None
        if str(token_id_str).startswith("0x"): return token_id_str
        if str(token_id_str).isdigit():
            try:
                return hex(int(token_id_str))
            except:
                return token_id_str
        return token_id_str

    def get_market_valuation(self, yes_token_id: str, no_token_id: str, entry_price_fallback=0.0) -> dict:
        """
        CORREGIDO: Usa get_price() SIN conversión hex.
        
        get_price() acepta token_id como STRING DECIMAL directamente.
        NO necesita conversión a hexadecimal.
        
        Returns:
        - BUY: best_bid (precio para comprar)
        - SELL: best_ask (precio para vender)
        """
        if not self.valid:
            return {"mid_price": entry_price_fallback, "status": "CLIENT_ERR"}

        # IMPORTANTE: NO convertir a hex, usar token_id decimal directo
        yes_token_str = str(yes_token_id)
        no_token_str = str(no_token_id)
        
        # 1. Intentar YES token con get_price()
        try:
            buy_price_res = self._robust_api_call(self.client.get_price, yes_token_str, side="BUY")
            sell_price_res = self._robust_api_call(self.client.get_price, yes_token_str, side="SELL")

            if buy_price_res and sell_price_res:
                # get_price() devuelve {"price": "0.007"} - convertir string a float
                best_bid = float(buy_price_res.get("price", "0"))
                best_ask = float(sell_price_res.get("price", "1"))

                # Verificar que los precios son válidos
                if best_bid > 0 and best_ask < 1.0 and best_ask > best_bid:
                    mid_price = (best_bid + best_ask) / 2
                    return {
                        "mid_price": mid_price,
                        "bid": best_bid,
                        "ask": best_ask,
                        "status": "ACTIVE_YES"
                    }
        except Exception as e:
            logger.debug(f"get_price on YES token failed: {e}")

        # 2. Fallback: Intentar NO token
        try:
            buy_price_no_res = self._robust_api_call(self.client.get_price, no_token_str, side="BUY")
            sell_price_no_res = self._robust_api_call(self.client.get_price, no_token_str, side="SELL")

            if buy_price_no_res and sell_price_no_res:
                best_bid_no = float(buy_price_no_res.get("price", "0"))
                best_ask_no = float(sell_price_no_res.get("price", "1"))
                
                # Derivar precio YES desde NO: YES = 1 - NO
                yes_bid = 1.0 - best_ask_no
                yes_ask = 1.0 - best_bid_no

                if yes_bid > 0 and yes_ask < 1.0 and yes_ask > yes_bid:
                    mid_price = (yes_bid + yes_ask) / 2
                    return {
                        "mid_price": mid_price,
                        "bid": yes_bid,
                        "ask": yes_ask,
                        "status": "ACTIVE_NO"
                    }
        except Exception as e:
            logger.debug(f"get_price on NO token failed: {e}")

        return {"mid_price": entry_price_fallback, "status": "NO_LIQUIDITY"}
    
    def get_all_bins_prices(self, bins_dict: dict):
        """Obtiene precios para todos los bins mapeados"""
        snapshot = {}
        if not bins_dict: return snapshot
        
        valid_bins = {k: v for k, v in bins_dict.items() if v.get("id_yes")}
        logger.info(f"📊 Getting live prices for {len(valid_bins)} mapped bins...")
        
        for bin_label, bin_data in valid_bins.items():
            snapshot[bin_label] = self.get_market_valuation(
                bin_data["id_yes"], 
                bin_data["id_no"],
                entry_price_fallback=0.5
            )
        
        for k in bins_dict:
            if k not in snapshot: 
                snapshot[k] = {"mid_price": 0.0, "status": "MISSING_ID"}
        
        return snapshot

    def find_markets_by_slug(self, base_slug: str, expected_count: int = 26) -> list:
        """Busca mercados en CLOB con early exit"""
        cache_key = f"slug_{base_slug}"
        if cache_key in self._markets_cache:
            logger.info(f"📦 Using cached markets for '{base_slug}'")
            return self._markets_cache[cache_key]
        
        url = f"{self.clob_base}/markets"
        next_cursor = ""
        found = []

        logger.info(f"🔎 CLOB Search: Looking for '{base_slug}'")
        logger.warning(f"⏱️  This may take 30-90 seconds (scanning until {expected_count} markets found)...")

        try:
            with httpx.Client(timeout=45.0) as c:
                page_count = 0
                start_time = time.time()
                
                while True:
                    params = {"next_cursor": next_cursor} if next_cursor else {}
                    r = c.get(url, params=params)
                    r.raise_for_status()
                    payload = r.json()

                    markets = payload.get("data", [])
                    page_count += 1
                    
                    for m in markets:
                        slug = m.get("market_slug", "")
                        if base_slug in slug:
                            found.append(m)
                    
                    # EARLY EXIT
                    if len(found) >= expected_count:
                        elapsed = time.time() - start_time
                        logger.success(
                            f"✅ Found all {len(found)} markets at page {page_count} "
                            f"({elapsed:.1f}s) - stopping early"
                        )
                        self._markets_cache[cache_key] = found
                        return found

                    if page_count % 20 == 0:
                        elapsed = time.time() - start_time
                        logger.debug(
                            f"   Page {page_count}: {len(found)}/{expected_count} found "
                            f"({elapsed:.1f}s elapsed)"
                        )
                    
                    next_cursor = payload.get("next_cursor")
                    if not next_cursor or next_cursor == "LTE=":
                        break

            elapsed = time.time() - start_time
            logger.success(
                f"✅ Found {len(found)} markets (scanned {page_count} pages in {elapsed:.1f}s)"
            )
            
            if found:
                self._markets_cache[cache_key] = found
            
            return found

        except Exception as e:
            logger.error(f"❌ CLOB search failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def fetch_market_ids_automatically(self, keywords: list, bins_dict: dict, market_slug: str = None):
        """Busca mercados y mapea bins"""
        working_bins = {k: {**v, "id_yes": None, "id_no": None} for k, v in bins_dict.items()}
        
        if not market_slug:
            logger.error("❌ market_slug is required")
            return working_bins
        
        logger.info(f"🚀 Strategy: CLOB with get_price() for pricing")
        markets = self.find_markets_by_slug(market_slug, expected_count=len(working_bins))
        
        if not markets:
            logger.error("❌ No markets found in CLOB")
            return working_bins
        
        bins_found = self._map_bins_from_markets(markets, working_bins)
        logger.success(f"🏁 Mapped {bins_found}/{len(working_bins)} bins")
        
        return working_bins

    def _map_bins_from_markets(self, markets: list, working_bins: dict) -> int:
        """Mapea bins desde una lista de mercados"""
        bins_found = 0
        
        for market in markets:
            question = market.get("question", "")
            
            match = re.search(r"(\d[\d,]*)\s*[–-]\s*(\d[\d,]*)|(\d[\d,]*)\+", question)
            
            if match:
                normalized_bin = ""
                if match.group(1):
                    s = match.group(1).replace(",", "")
                    e = match.group(2).replace(",", "")
                    normalized_bin = f"{s}-{e}"
                elif match.group(3):
                    n = match.group(3).replace(",", "")
                    normalized_bin = f"{n}+"
                
                if normalized_bin in working_bins:
                    tokens = market.get("tokens", [])
                    
                    for token in tokens:
                        token_id = token.get("token_id")
                        outcome = token.get("outcome", "")
                        
                        if outcome == "Yes":
                            working_bins[normalized_bin]["id_yes"] = token_id
                        elif outcome == "No":
                            working_bins[normalized_bin]["id_no"] = token_id
                    
                    if working_bins[normalized_bin]["id_yes"] and working_bins[normalized_bin]["id_no"]:
                        bins_found += 1
                        logger.debug(f"  ✓ Mapped {normalized_bin}")
        
        return bins_found

    def get_price_history(self, market_id: str, fidelity: int = 15, start_timestamp: int = 1577836800) -> list | None:
        """Required for run_ingest.py"""
        if not self.valid: return None
        try:
            url = "https://clob.polymarket.com/prices-history"
            params = {"market": market_id, "fidelity": fidelity, "startTs": start_timestamp}
            with httpx.Client(timeout=15.0) as c:
                r = c.get(url, params=params)
                r.raise_for_status()
                return r.json().get("history", [])
        except Exception as e:
            logger.warning(f"History fetch error: {e}")
            return None
    
    def get_market_details(self, keywords: list) -> dict | None:
        """Legacy helper"""
        return None