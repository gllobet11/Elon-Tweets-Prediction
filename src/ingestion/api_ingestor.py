"""
api_ingestor.py

Handles fetching tweets and official market metadata from xTracker.
Logic: Uses settings.py keywords to pinpoint the EXACT official market and gets its UTC timestamps.
"""

from datetime import datetime, timezone
import httpx
import pandas as pd
from loguru import logger
import time

class ApiIngestor:
    POSTS_ENDPOINT = "https://xtracker.polymarket.com/api/users/elonmusk/posts"
    USER_ENDPOINT = "https://xtracker.polymarket.com/api/users/elonmusk"
    
    HEADERS = {
        "accept": "*/*",
        "referer": "https://xtracker.polymarket.com/user/elonmusk",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get_official_market_dates(self, target_keywords: list = None):
        """
        Fetches metadata and finds the market that matches YOUR keywords.
        
        Args:
            target_keywords: List of strings from settings.py (e.g. ["December 5", "December 12"])
        
        Returns:
            (start_date_utc, end_date_utc, title)
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # logger.info(f"📡 Scanning xTracker for market matching: {target_keywords}...")
                resp = client.get(self.USER_ENDPOINT, headers=self.HEADERS)
                resp.raise_for_status()
                data = resp.json().get("data", {})
                trackings = data.get("trackings", [])
                
                matched_market = None

                # 1. SEARCH BY KEYWORDS (Priority)
                if target_keywords:
                    # Normalize keywords to lowercase for comparison
                    kws_lower = [k.lower() for k in target_keywords]
                    
                    for t in trackings:
                        title = t.get("title", "").lower()
                        # Check if ALL keywords are present in the title
                        if all(kw in title for kw in kws_lower):
                            matched_market = t
                            logger.success(f"✅ Found Market by Keywords: '{t.get('title')}'")
                            break
                
                # 2. FALLBACK (If keywords fail or are empty, take the newest active one)
                if not matched_market:
                    if target_keywords:
                        logger.warning(f"⚠️ No market matched keywords {target_keywords}. Trying auto-detection...")
                    
                    # Auto-detect: Pick the most recently created ACTIVE market
                    active_candidates = [t for t in trackings if t.get("isActive")]
                    if active_candidates:
                        active_candidates.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
                        matched_market = active_candidates[0]
                        logger.info(f"⚠️ Auto-selected most recent active market: '{matched_market.get('title')}'")

                if matched_market:
                    # Use utc=True to robustly parse the date string and get a UTC-aware timestamp.
                    s_date_raw = pd.to_datetime(matched_market["startDate"], utc=True)
                    e_date_raw = pd.to_datetime(matched_market["endDate"], utc=True)
                    title = matched_market.get("title")

                    # --- Timezone Standardization (Centralized Logic) ---
                    # The market start/end times are 12:00 PM ET. The API provides them in UTC.
                    # We need to ensure they are correctly set to 12 PM ET regardless of DST.
                    try:
                        from zoneinfo import ZoneInfo
                        ET_TZ = ZoneInfo("America/New_York")
                    except ImportError:
                        from dateutil import tz
                        ET_TZ = tz.gettz("America/New_York")

                    # 1. Convert the UTC dates to ET to see what time they represent there.
                    s_date_et = s_date_raw.tz_convert(ET_TZ)
                    e_date_et = e_date_raw.tz_convert(ET_TZ)

                    # 2. Force the time to be exactly 12:00 PM (noon) in ET.
                    s_date_noon_et = s_date_et.replace(hour=12, minute=0, second=0, microsecond=0)
                    e_date_noon_et = e_date_et.replace(hour=12, minute=0, second=0, microsecond=0)

                    # 3. Convert the correct ET noon time back to UTC for internal use.
                    s_date_utc = s_date_noon_et.astimezone(timezone.utc)
                    e_date_utc = e_date_noon_et.astimezone(timezone.utc)
                    
                    logger.info(f"Standardized market time: {s_date_utc} to {e_date_utc}")

                    return s_date_utc.to_pydatetime(), e_date_utc.to_pydatetime(), title
                
                logger.error("❌ No matching or active market found in API.")
                return None, None, None

        except Exception as e:
            logger.error(f"Error fetching official dates: {e}")
            return None, None, None

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches tweets and applies STRICT filtering based on the passed dates.
        """
        if start_date.tzinfo is None: start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None: end_date = end_date.replace(tzinfo=timezone.utc)

        # Buffer: Request slightly more data
        req_start = start_date - pd.Timedelta(hours=4)
        req_end = end_date + pd.Timedelta(hours=4)

        start_str = req_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = req_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        all_tweets = []
        next_cursor = None
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                while True:
                    params = {"startDate": start_str, "endDate": end_str, "limit": 100}
                    if next_cursor: params["cursor"] = next_cursor

                    response = client.get(self.POSTS_ENDPOINT, params=params, headers=self.HEADERS)
                    
                    if response.status_code == 429:
                        time.sleep(2)
                        continue
                        
                    response.raise_for_status()
                    json_response = response.json()
                    
                    data_chunk = json_response.get("data", [])
                    if isinstance(data_chunk, dict): data_chunk = [data_chunk]
                    
                    if not data_chunk: break
                        
                    all_tweets.extend(data_chunk)
                    
                    next_cursor = json_response.get("next_cursor")
                    if not next_cursor or next_cursor == "LTE=": break
                    time.sleep(0.1)

            if not all_tweets:
                return pd.DataFrame()

            df = pd.DataFrame(all_tweets)
            
            if "id" in df.columns and "platformId" in df.columns:
                df = df.drop(columns=["id"])
            
            rename_map = {"platformId": "id", "content": "text", "createdAt": "created_at"}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype("int64")
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
            
            # STRICT FILTERING
            mask = (df["created_at"] >= start_date) & (df["created_at"] <= end_date)
            df_filtered = df.loc[mask].copy()
            
            df_filtered = df_filtered.sort_values("created_at")
            df_filtered = df_filtered.drop_duplicates(subset=["id"])
            
            return df_filtered

        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return pd.DataFrame()