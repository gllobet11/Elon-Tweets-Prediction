"""
dashboard_data_loader.py
Hybrid Loader: Combines Historical CSV with LIVE API Data for strict accuracy.
IMPROVED: Now uses CLOB /markets API for more reliable market discovery
"""
import os
import pickle
import glob
import pandas as pd
from loguru import logger
from datetime import datetime, timezone, timedelta

try:
    from src.ingestion.unified_feed import load_unified_data
    from src.ingestion.poly_feed import PolymarketFeed
    from src.ingestion.api_ingestor import ApiIngestor
    from config.bins_definition import MARKET_BINS
    from config.settings import MARKET_KEYWORDS, MARKET_SLUG
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    MARKET_KEYWORDS = ["Elon", "Musk", "tweets"]
    MARKET_SLUG = "elon-musk-of-tweets-december-5-december-12"

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
    logger.info("Using 'zoneinfo' for America/New_York TZ.")
except ImportError:
    try:
        from dateutil import tz # Necesitarás dateutil en los imports
        ET_TZ = tz.gettz("America/New_York")
        if ET_TZ is None:
            raise ImportError("dateutil could not find America/New_York.")
        logger.info("Using 'dateutil.tz' for America/New_York TZ.")
    except ImportError:
        # Fallback de emergencia, menos robusto por DST
        logger.warning("Could not load full TZ library. Falling back to fixed UTC-5.")
        ET_TZ = timezone(timedelta(hours=-5))


class DashboardDataLoader:
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = os.getcwd()
        
        self.project_root = project_root
        self.poly_feed = PolymarketFeed()
        self.api_ingestor = ApiIngestor()

    def load_and_prepare_tweets_data(self):
        """
        Loads historical data and patches it with LIVE data for the current active market.
        This ensures the counter is pixel-perfect with Polymarket.
        """
        # 1. Load Historical Base (CSV/Parquet)
        granular_data = load_unified_data()
        if granular_data.empty:
            granular_data = pd.DataFrame(columns=["created_at", "text", "id"])

        # Ensure datetime index is UTC-aware
        if "created_at" in granular_data.columns:
            # Esta línea hace que la data sea UTC-aware (tz-aware)
            granular_data["created_at"] = pd.to_datetime(granular_data["created_at"], utc=True)
            granular_data = granular_data.set_index("created_at")
        
        if not isinstance(granular_data.index, pd.DatetimeIndex):
             # Fallback if index isn't datetime
             return pd.DataFrame(), pd.DataFrame()

        # 2. LIVE PATCH: Fetch official market tweets right now
        try:
            # Get current official window
            off_start, off_end, _ = self.api_ingestor.get_official_market_dates(MARKET_KEYWORDS)
            
            if off_start and off_end:
                logger.info(f"⚡ LIVE PATCH: Fetching real-time tweets from {off_start} to {off_end}...")
                live_df = self.api_ingestor.fetch(off_start, off_end)
                
                if not live_df.empty:
                    if "created_at" in live_df.columns:
                        live_df["created_at"] = pd.to_datetime(live_df["created_at"], utc=True)
                        live_df = live_df.set_index("created_at")
                    
                    # Merge logic:
                    # We remove any data in granular_data that overlaps with the live window
                    # to avoid duplicates, then append the fresh live data.
                    mask_historical = granular_data.index < off_start
                    granular_history = granular_data.loc[mask_historical]
                    
                    # Concatenate History + Live
                    granular_data = pd.concat([granular_history, live_df])
                    
                    # Deduplicate just in case
                    if "id" in granular_data.columns:
                        granular_data = granular_data.sort_index().drop_duplicates(subset=["id"], keep="last")
                    else:
                        granular_data = granular_data.sort_index().drop_duplicates()
                        
                    logger.success(f"✅ Live Patch Applied. Total Tweets: {len(granular_data)}")
        
        except Exception as e:
            logger.error(f"⚠️ Live patch failed (using cached data only): {e}")

        # ----------------------------------------------------------------------
        # 3. RESAMPLEO DIARIO EN UTC Y LUEGO CONVERSIÓN DE ZONA HORARIA
        # ----------------------------------------------------------------------

        # A. Asegurar que el índice esté en UTC
        if granular_data.index.tz is None:
            granular_data.index = granular_data.index.tz_localize("UTC")
        else:
            # Forzar conversión a UTC si es otra zona horaria
            granular_data.index = granular_data.index.tz_convert("UTC")

        # B. Resample to Daily en UTC
        # El resampleo se hace sobre la medianoche de UTC.
        daily_data = granular_data.resample("D").size().to_frame(name="n_tweets")
        
        # C. Convertir el índice diario a la zona horaria del mercado (ET_TZ)
        # Esto alinea el DataFrame diario con la perspectiva de "medianoche de Nueva York",
        # que es la referencia visual del dashboard.
        try:
            daily_data.index = daily_data.index.tz_convert(ET_TZ)
            logger.info(f"🕒 Daily data resampled in UTC and index converted to {ET_TZ}.")
        except Exception as e:
            logger.error(f"⚠️ Failed to convert daily index to ET: {e}. Keeping UTC.")
        
        # Opcional: Deslocalizar la data diaria a Naive para Prophet
        # Prophet requiere fechas Naive. Si tu modelo lo maneja, esta línea es opcional.
        # daily_data.index = daily_data.index.tz_localize(None) 
        
        # Devolver datos granulares en UTC y los datos diarios en la zona horaria de ET
        return granular_data.tz_convert(ET_TZ), daily_data

    def load_prophet_model(self):
        pattern = os.path.join(self.project_root, "best_prophet_model_*.pkl")
        files = glob.glob(pattern)
        if not files: files = glob.glob("best_prophet_model_*.pkl")
        if not files: return {"model": None, "model_name": "Not Found", "mae": 0.0}
        try:
            latest_model = max(files, key=os.path.getctime)
            with open(latest_model, "rb") as f: data = pickle.load(f)
            return {"model": data.get("model"), "model_name": os.path.basename(latest_model), "mae": data.get("metrics", {}).get("mae", 0.0)}
        except: return {"model": None, "model_name": "Error", "mae": 0.0}

    def load_risk_parameters(self):
        path = os.path.join(self.project_root, "risk_params.pkl")
        defaults = {"alpha": 0.3, "kelly": 0.2, "min_edge": 0.05}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f: loaded = pickle.load(f)
                defaults["alpha"] = loaded.get("alpha_nbinom", loaded.get("alpha", 0.3))
                defaults["kelly"] = loaded.get("kelly_fraction", loaded.get("kelly", 0.2))
            except: pass
        return defaults

    def load_historical_performance(self):
        path = os.path.join(os.path.join(self.project_root, "data", "processed"), "historical_performance.csv")
        if os.path.exists(path): return pd.read_csv(path, index_col="week_start_date", parse_dates=True)
        return pd.DataFrame()

    def fetch_market_data(self):
        """
        IMPROVED LOGIC:
        1. Look for market in xTracker using config.settings.MARKET_KEYWORDS for dates
        2. Use MARKET_SLUG for more reliable market discovery in Polymarket
        3. Get EXACT dates and Title from xTracker
        4. Fetch Prices from Polymarket using the slug-based discovery
        """
        logger.info(f"🔎 Looking for market matching config keywords: {MARKET_KEYWORDS}")
        logger.info(f"🎯 Using market slug for Polymarket search: {MARKET_SLUG}")
        
        # 1. Get Official Dates from xTracker (which are now correctly adjusted to 12 PM ET, in UTC)
        official_start, official_end, official_title = self.api_ingestor.get_official_market_dates(MARKET_KEYWORDS)

        # Título base (independiente de que haya fechas o no)
        final_question = official_title if official_title else "Unknown Market"

        # 2. Fetch Bins using IMPROVED SLUG-BASED METHOD
        logger.info("🔄 Fetching market IDs using slug-based discovery...")
        
        current_ids = self.poly_feed.fetch_market_ids_automatically(
            keywords=MARKET_KEYWORDS,  # Fallback keywords
            bins_dict=MARKET_BINS.copy(),
            market_slug=MARKET_SLUG   # PRIMARY: Use slug for discovery
        )
        
        # 3. Get Current Prices for all bins
        snapshot = self.poly_feed.get_all_bins_prices(current_ids)

        # 4. Count successfully mapped bins
        mapped_bins = sum(1 for v in current_ids.values() if v.get("id_yes") and v.get("id_no"))
        total_bins = len(MARKET_BINS)
        
        if mapped_bins < total_bins:
            logger.warning(f"⚠️ Only {mapped_bins}/{total_bins} bins were mapped. Some markets might be missing.")
        else:
            logger.success(f"✅ All {mapped_bins} bins successfully mapped!")

        # 5. Try to get market details (optional, for better question text)
        market = None
        if self.poly_feed.valid:
            markets_found = self.poly_feed.find_markets_by_slug(MARKET_SLUG)
            if markets_found:
                first_market = markets_found[0]
                market_question = first_market.get("question", "")
                if market_question and "How many" in market_question:
                    final_question = market_question.split("?")[0] + "?"

        return {
            "market_details": market,
            "market_question": final_question,
            "bins_config": [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()],
            "market_snapshot": snapshot,
            "market_start_date": official_start,
            "market_end_date": official_end,
            "updated_bins": current_ids,
            "bins_mapped": f"{mapped_bins}/{total_bins}",
        }
