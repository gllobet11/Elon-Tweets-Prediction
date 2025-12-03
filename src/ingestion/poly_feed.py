import sys
import time
from functools import lru_cache
from py_clob_client.client import ClobClient
from py_clob_client.exceptions import PolyApiException
import re
import os
import json
from loguru import logger
import pandas as pd # Import pandas here to resolve NameError

class PolymarketFeed:
    def __init__(self, host="https://clob.polymarket.com", chain_id=137):
        self.valid = False
        try:
            self.client = ClobClient(host, chain_id=chain_id)
            self.valid = True
        except Exception as e:
            print(f"❌ Error inicializando ClobClient: {e}", file=sys.stderr)
            self.client = None

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
                    logger.error(f"API call failed after {retries} attempts: {e}")
                    return None

    def fetch_market_ids_automatically(self, keywords: list, bins_dict: dict):
        if not self.valid: return None
        
        cache_path = "data/markets/cached_ids.json"
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                if cached_data.get("keywords") == keywords:
                    logger.info(f"✅ IDs de mercado cargados desde la caché: {cache_path}")
                    return cached_data["bins_dict"]
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ No se pudo leer la caché de IDs ({e}).")

        logger.info(f"🔎 Buscando mercados en la API (Keywords: {keywords})...")
        
        for bin_info in bins_dict.values():
            bin_info['id_yes'] = None
            bin_info['id_no'] = None
            
        mapped_count = 0
        next_cursor = ""
        while True:
            markets_resp = self._robust_api_call(self.client.get_markets, next_cursor=next_cursor)
            if not markets_resp or not markets_resp.get('data'): break
            
            for market in markets_resp.get('data', []):
                question = market.get('question', '').lower()
                if all(k.lower() in question for k in keywords) and market.get('active'):
                    match = re.search(r'([\d,]+-[\d,]+|[\d,]+\+)', market.get('question', ''))
                    if match:
                        bin_label = match.group(1)
                        tokens = market.get('tokens', [])
                        yes_token = next((t for t in tokens if t.get('outcome') == 'Yes'), None)
                        no_token = next((t for t in tokens if t.get('outcome') == 'No'), None)
                        
                        if yes_token and no_token and bin_label in bins_dict:
                            if bins_dict[bin_label].get('id_yes') is None:
                                bins_dict[bin_label]['id_yes'] = yes_token['token_id']
                                bins_dict[bin_label]['id_no'] = no_token['token_id']
                                logger.info(f"   [✅] Bin {bin_label} -> Mapeado")
                                mapped_count += 1
            
            next_cursor = markets_resp.get('next_cursor')
            if not next_cursor or next_cursor == "LTE=": break
        
        logger.info(f"✅ Mapeo completado: {mapped_count}/{len(bins_dict)} bins listos.")
        
        if mapped_count > 0:
            try:
                with open(cache_path, 'w') as f:
                    cache_content = {"keywords": keywords, "bins_dict": bins_dict}
                    json.dump(cache_content, f, indent=2)
                logger.info(f"✅ IDs de mercado guardados en caché: {cache_path}")
            except IOError as e:
                logger.warning(f"⚠️ No se pudo escribir en la caché de IDs: {e}")
        
        return bins_dict

    def get_market_valuation(self, yes_token_id: str, no_token_id: str, entry_price_fallback=0.0) -> dict:
        if not self.valid: return {'mid_price': entry_price_fallback, 'status': 'CLIENT_ERR'}

        ob_yes = self._robust_api_call(self.client.get_order_book, yes_token_id)
        if ob_yes and (ob_yes.bids or ob_yes.asks):
            highest_bid = max([float(b.price) for b in ob_yes.bids]) if ob_yes.bids else 0.0
            lowest_ask = min([float(a.price) for a in ob_yes.asks]) if ob_yes.asks else 1.0
            
            if highest_bid > 0 or lowest_ask < 1:
                return {'mid_price': (highest_bid + lowest_ask) / 2, 'bid': highest_bid, 'ask': lowest_ask, 'status': 'ACTIVE_YES'}

        ob_no = self._robust_api_call(self.client.get_order_book, no_token_id)
        if ob_no and (ob_no.bids or ob_no.asks):
            highest_bid_no = max([float(b.price) for b in ob_no.bids]) if ob_no.bids else 0.0
            lowest_ask_no = min([float(a.price) for a in ob_no.asks]) if ob_no.asks else 1.0
            
            if highest_bid_no > 0 or lowest_ask_no < 1:
                ask = 1.0 - highest_bid_no
                bid = 1.0 - lowest_ask_no
                return {'mid_price': (bid + ask) / 2, 'bid': bid, 'ask': ask, 'status': 'ACTIVE_NO'}
            
        return {'mid_price': entry_price_fallback, 'status': 'NO_LIQUIDITY'}

    def get_all_bins_prices(self, bins_dict: dict):
        snapshot = {}
        logger.info(f"📊 Obteniendo precios para {len(bins_dict)} bins...")
        for bin_label, bin_data in bins_dict.items():
            if bin_data.get('id_yes') and bin_data.get('id_no'):
                valuation = self.get_market_valuation(bin_data['id_yes'], bin_data['id_no'])
                snapshot[bin_label] = valuation
        return snapshot
        
    def get_market_details(self, keywords: list) -> dict | None:
        if not self.valid: return None
        next_cursor = ""
        while True:
            markets_resp = self._robust_api_call(self.client.get_markets, next_cursor=next_cursor)
            if not markets_resp or not markets_resp.get('data'): break
            for market in markets_resp.get('data', []):
                question = market.get('question', '').lower()
                if all(keyword.lower() in question for keyword in keywords) and market.get('active') is True:
                    return market
            next_cursor = markets_resp.get('next_cursor')
            if not next_cursor or next_cursor == "LTE=": break
        return None

    def get_market_dates(self, market_description: str) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
        """
        Parses the market start and end dates from the description text robustly.
        Handles cases where the market spans across the end of a year.
        """
        logger.debug(f"Attempting to parse dates from description: '{market_description}'")
        if not market_description:
            return None, None

        match = re.search(r'from (.*? ET) to (.*? ET)', market_description)
        if not match:
            logger.error("Could not extract date strings using regex from description.")
            return None, None
            
        start_date_str, end_date_str = match.groups()
        
        # Clean the strings
        start_clean = start_date_str.replace(" ET", "").strip()
        end_clean = end_date_str.replace(" ET", "").strip()
        
        try:
            # Check for year in end date string to use as a reference
            end_year_match = re.search(r'(\d{4})', end_clean)
            ref_year = int(end_year_match.group(1)) if end_year_match else datetime.now().year

            # If start date is missing a year, add the reference year
            if not re.search(r'(\d{4})', start_clean):
                start_clean = f"{start_clean}, {ref_year}"

            # Parse dates now that they should both have years
            end_dt = pd.to_datetime(end_clean)
            start_dt = pd.to_datetime(start_clean)

            # If start date is after end date, it must be from the previous year
            if start_dt > end_dt:
                start_dt = start_dt.replace(year=start_dt.year - 1)

            # Localize to ET and then convert to UTC
            tz = 'America/New_York'
            market_start_date = start_dt.tz_localize(tz).tz_convert('UTC')
            market_end_date = end_dt.tz_localize(tz).tz_convert('UTC')
            
            logger.info(f"Parsed market dates: START={market_start_date}, END={market_end_date}")
            return market_start_date, market_end_date

        except Exception as e:
            logger.error(f"Error parsing market dates: {e}")
            return None, None