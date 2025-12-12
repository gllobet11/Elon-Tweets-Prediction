import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from loguru import logger
from datetime import datetime, timedelta

from google.cloud import bigquery

client = bigquery.Client()

# --- PATH CONFIGURATION ---
# Usamos os.getcwd() para asegurar rutas absolutas desde la raíz del proyecto
PROJECT_ROOT = os.getcwd()

SPACEX_LAUNCHES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "spacex_launches.csv")
ELON_NEWS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "elon_news_cleaned.csv")
# Apuntamos al nuevo archivo diario generado
POLY_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "eda", "poly_daily_features.csv")

# --- DATA LOADING FUNCTIONS ---

def load_spacex_launches_from_csv():
    if not os.path.exists(SPACEX_LAUNCHES_PATH):
        return pd.DatetimeIndex([])
    try:
        df_launches = pd.read_csv(SPACEX_LAUNCHES_PATH)
        launch_dates = pd.to_datetime(df_launches["launch_date"], utc=True)
        return pd.DatetimeIndex(launch_dates.unique()).sort_values()
    except Exception as e:
        logger.error(f"Error loading SpaceX data: {e}")
        return pd.DatetimeIndex([])


def load_elon_news_data():
    if not os.path.exists(ELON_NEWS_PATH):
        return pd.DataFrame()
    try:
        df_news = pd.read_csv(ELON_NEWS_PATH)
        df_news["date"] = pd.to_datetime(df_news["date"], utc=True)
        return df_news.set_index("date")
    except Exception as e:
        logger.error(f"Error loading News data: {e}")
        return pd.DataFrame()


def load_polymarket_features():
    """
    Carga los features DIARIOS de Polymarket generados por tools/generate_daily_poly.py
    """
    if not os.path.exists(POLY_FEATURES_PATH):
        logger.warning(f"⚠️ Polymarket file not found at {POLY_FEATURES_PATH}. Features will be 0.")
        return pd.DataFrame()
    try:
        df_poly = pd.read_csv(POLY_FEATURES_PATH, index_col=0)
        df_poly.index = pd.to_datetime(df_poly.index, utc=True)
        return df_poly
    except Exception as e:
        logger.error(f"Error loading Polymarket data: {e}")
        return pd.DataFrame()


def calculate_spacex_features_vectorized(target_dates, launch_dates, future_days=7):
    """
    Calcula proximidad usando merge_asof (Vectorizado y rápido).
    """
    if launch_dates.empty:
        return pd.Series(14, index=target_dates), pd.Series(0, index=target_dates)

    # Preparar Targets
    targets = pd.DataFrame({"date": target_dates}).sort_values("date")
    if targets["date"].dt.tz is None:
        targets["date"] = targets["date"].dt.tz_localize("UTC")

    # Preparar Launches
    launches = pd.DataFrame({"launch_date": launch_dates}).sort_values("launch_date")
    if launches["launch_date"].dt.tz is None:
        launches["launch_date"] = launches["launch_date"].dt.tz_localize("UTC")

    # Proximidad (Días hasta el SIGUIENTE lanzamiento)
    merged = pd.merge_asof(
        targets, launches, left_on="date", right_on="launch_date", direction="forward"
    )

    days_to_launch = (merged["launch_date"] - merged["date"]).dt.days
    proximity_series = days_to_launch.fillna(14).clip(upper=14).astype(int)
    proximity_series.index = targets["date"]

    # Conteo Futuro
    target_vals = targets["date"].values
    launch_vals = launches["launch_date"].values
    idx_start = np.searchsorted(launch_vals, target_vals, side="right")
    target_plus_window = target_vals + np.timedelta64(future_days, "D")
    idx_end = np.searchsorted(launch_vals, target_plus_window, side="right")

    count_series = pd.Series(idx_end - idx_start, index=targets["date"])

    return proximity_series.reindex(target_dates), count_series.reindex(target_dates)


def fetch_gdelt_data():
    """
    Fetches daily Elon Musk related news data (Volume and Sentiment) from GDELT.
    """
    logger.info("📡 Executing BigQuery query for GDELT news data...")

    try:
        from google.auth import default as get_default_credentials
        try:
            credentials, project = get_default_credentials()
        except Exception as e:
            pass

        query = """
        SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
        COUNT(*) AS news_volume,
        AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_sentiment
        FROM `gdelt-bq.gdeltv2.gkg`
        WHERE
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8))
            >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
        AND V2Persons LIKE '%Elon Musk%'
        GROUP BY date
        ORDER BY date ASC;
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()

        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(ELON_NEWS_PATH, index=False)
        logger.success(
            f"✅ GDELT news data saved to '{os.path.basename(ELON_NEWS_PATH)}' ({len(df)} days)."
        )
        return df

    except Exception as e:
        logger.error(f"❌ BigQuery/GDELT API Error: {e}. News features will be zero.")
        return pd.DataFrame(columns=["date", "news_volume", "avg_sentiment"])


def fetch_tesla_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        tsla_data = yf.download(
            "TSLA", start=start_date, end=end_date, progress=False, auto_adjust=False
        )

        if isinstance(tsla_data.columns, pd.MultiIndex):
            try:
                close_data = tsla_data.xs("Close", level=0, axis=1)
            except:
                close_data = tsla_data.iloc[:, 0]
        else:
            close_data = (
                tsla_data["Close"]
                if "Close" in tsla_data.columns
                else tsla_data.iloc[:, 0]
            )

        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]

        returns = close_data.pct_change().dropna()
        if returns.empty:
            return pd.DataFrame()

        garch_model = arch_model(
            100 * returns, vol="Garch", p=1, q=1, mean="Constant", dist="t"
        )
        res = garch_model.fit(update_freq=5, disp="off")

        df_result = pd.DataFrame(index=returns.index)
        df_result["tsla_returns"] = returns
        df_result["tsla_volatility"] = res.conditional_volatility

        if df_result.index.tz is None:
            df_result.index = df_result.index.tz_localize("UTC")
        else:
            df_result.index = df_result.index.tz_convert("UTC")

        full_idx = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
        df_result = df_result.reindex(full_idx)
        df_result["tsla_volatility"] = df_result["tsla_volatility"].ffill()
        df_result["tsla_returns"] = df_result["tsla_returns"].fillna(0)

        return df_result

    except Exception as e:
        logger.error(f"Error fetching Tesla data: {e}")
        return pd.DataFrame()


class FeatureEngineer:
    def __init__(self):
        self.rolling_window_size = 28
        self.regime_threshold_z_score = 1.0

    def _resample_to_daily(self, df_granular: pd.DataFrame) -> pd.DataFrame:
        df_copy = df_granular.copy()
        if "created_at" in df_copy.columns:
            df_copy["created_at"] = pd.to_datetime(df_copy["created_at"], utc=True)
            df_copy = df_copy.set_index("created_at")

        df_daily = df_copy.resample("D").size().to_frame("n_tweets")
        if not df_daily.empty:
            # Reindex to ensure all days are present, using the timezone from the resampled data
            all_days = pd.date_range(
                start=df_daily.index.min(), end=df_daily.index.max(), freq="D", tz=df_daily.index.tz
            )
            df_daily = df_daily.reindex(all_days, fill_value=0)
        return df_daily

    def _calculate_regime_change(self, daily_series):
        if daily_series.empty:
            return pd.Series(0), pd.Series(0), pd.Series(0), pd.Series(0)

        rolling_mean = daily_series.rolling(
            window=self.rolling_window_size, min_periods=1
        ).mean()
        rolling_std = daily_series.rolling(
            window=self.rolling_window_size, min_periods=1
        ).std()

        z_score = (
            (daily_series - rolling_mean) / rolling_std.replace(0, pd.NA)
        ).fillna(0)

        high = (z_score > self.regime_threshold_z_score).astype(int)
        low = (z_score < -self.regime_threshold_z_score).astype(int)
        change = ((high.diff().abs() > 0) | (low.diff().abs() > 0)).astype(int)
        change.iloc[0] = 0
        return z_score, high, low, change

    def _add_event_features(self, df):
        # 1. BASE: Lagged Tweets
        lag_1 = df["n_tweets"].shift(1).fillna(0)
        df["lag_1"] = lag_1
        df["roll_sum_7"] = lag_1.rolling(7).sum().fillna(0)
        df["momentum"] = lag_1.diff(7).fillna(0)

        roll_mean_7 = lag_1.rolling(7).mean()
        roll_mean_30 = lag_1.rolling(30).mean()

        df["last_burst"] = (lag_1 > (roll_mean_7 * 1.5)).astype(int)
        df["overheat_ratio"] = (roll_mean_7 / (roll_mean_30 + 1)).fillna(1.0)
        df["is_overheated"] = (df["overheat_ratio"] > 1.5).astype(int)

        # 3. SPACEX
        if "spacex_launch_proximity" in df.columns:
            df["event_spacex_hype"] = (df["spacex_launch_proximity"] <= 1).astype(int)
            df["event_spacex_t_minus_1"] = (df["spacex_launch_proximity"] == 1).astype(int)
        else:
            df["event_spacex_hype"] = 0
            df["event_spacex_t_minus_1"] = 0

        # 4. TESLA
        if "tsla_returns" in df.columns:
            tsla_lag = df["tsla_returns"].shift(1).fillna(0)
            df["event_tsla_pump"] = (tsla_lag > 0.04).astype(int)
            df["event_tsla_crash"] = (tsla_lag < -0.04).astype(int)
            
            if "tsla_volatility" in df.columns:
                 # Si la volatilidad es 20% mayor a la media del mes
                 df["event_high_volatility"] = (df["tsla_volatility"] > df["tsla_volatility"].rolling(30).mean() * 1.2).astype(int)
            else:
                 df["event_high_volatility"] = 0
        else:
            df["event_tsla_pump"] = 0
            df["event_tsla_crash"] = 0
            df["event_high_volatility"] = 0

        # 5. NEWS
        df["event_viral_news"] = 0
        df["news_momentum"] = 0
        df["event_negative_news_spike"] = 0

        if "news_vol_log" in df.columns:
            news_vol_lag = df["news_vol_log"].shift(1).fillna(0)
            df["news_momentum"] = news_vol_lag
            df["event_viral_news"] = (news_vol_lag > 7.5).astype(int)

            if "avg_sentiment" in df.columns:
                sentiment_lag = df["avg_sentiment"].shift(1).fillna(0)
                is_loud = news_vol_lag > 7.0
                is_bad = sentiment_lag < -0.2
                df["event_negative_news_spike"] = (is_loud & is_bad).astype(int)

        # 6. CALENDAR
        df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        df["is_sunday"] = (df.index.dayofweek == 6).astype(int)

        return df.fillna(0)

    def process_data(self, df_raw):
        # 1. CARGA INICIAL Y LIMPIEZA
        if df_raw.empty:
            return pd.DataFrame()

        if isinstance(df_raw.index, pd.DatetimeIndex) and "n_tweets" in df_raw.columns:
            df = df_raw.copy()
        else:
            df = self._resample_to_daily(df_raw)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()

        # 2. INTEGRACIÓN DE DATOS EXTERNOS

        # A) SpaceX
        try:
            launches = load_spacex_launches_from_csv()
            prox, counts = calculate_spacex_features_vectorized(df.index, launches)
            df["spacex_launch_proximity"] = prox
            df["spacex_future_launch_count"] = counts
        except Exception:
            df["spacex_launch_proximity"] = 14
            df["spacex_future_launch_count"] = 0

        # B) Tesla
        try:
            start_str = df.index.min().strftime("%Y-%m-%d")
            end_str = df.index.max().strftime("%Y-%m-%d")
            df_tsla = fetch_tesla_market_data(start_str, end_str)

            if not df_tsla.empty:
                cols_to_drop = [c for c in df_tsla.columns if c in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                df = df.join(df_tsla, how="left")
        except Exception as e:
            logger.error(f"Tesla data join failed: {e}")

        # C) News
        try:
            df_news = load_elon_news_data()
            if not df_news.empty:
                cols_to_drop = [c for c in ["news_volume", "avg_sentiment", "news_vol_log"] if c in df.columns]
                df = df.drop(columns=cols_to_drop)
                df = df.join(df_news[["news_volume", "avg_sentiment"]], how="left")
                df["news_vol_log"] = np.log1p(df["news_volume"].fillna(0))
                df["avg_sentiment"] = df["avg_sentiment"].fillna(0)
                df = df.drop(columns=["news_volume"])
        except Exception:
            pass
        
        # D) POLYMARKET (NUEVO BLOQUE: DAILY FEATURES)
        # -------------------------------------------------------------------
        try:
            df_poly = load_polymarket_features()
            if not df_poly.empty:
                # 1. Eliminar colisiones
                cols_to_drop = [c for c in df_poly.columns if c in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                
                # 2. Unir (Join Left) usando el índice fecha
                df = df.join(df_poly, how="left")

                # 3. Relleno ligero (solo para fines de semana o huecos mínimos)
                poly_cols = df_poly.columns.tolist()
                df[poly_cols] = df[poly_cols].ffill(limit=3).fillna(0)
                
                logger.info(f"✅ Integrated {len(poly_cols)} Polymarket features.")
        except Exception as e:
            logger.error(f"Error integrating Polymarket features: {e}")
        # -------------------------------------------------------------------

        # 3. GENERACIÓN DE FEATURES TÉCNICAS
        df = self._add_event_features(df)

        # 4. REGIMEN CHANGE
        z, high, low, change = self._calculate_regime_change(df["lag_1"])
        df["is_high_regime"] = high
        df["is_regime_change"] = change
        df["regime_intensity"] = z

        return df.fillna(0)

    def generate_live_features(self, df_raw):
        full_features = self.process_data(df_raw)
        
        # Lista FINAL de features requeridas por el modelo
        required = [
            "lag_1",
            "roll_sum_7",
            "momentum",
            "last_burst",
            "is_high_regime",
            "is_regime_change",
            "event_spacex_hype",
            "event_spacex_t_minus_1",
            "event_tsla_crash",
            "event_tsla_pump",
            "event_high_volatility",
            "event_negative_news_spike",
            "is_weekend",
            "is_sunday",
            # Nuevas features de Polymarket
            "poly_implied_mean_tweets",
            "poly_entropy",
            "poly_daily_vol",
            "poly_max_prob"
        ]

        # Asegurar que todas existan (rellenar con 0 si faltan)
        for col in required:
            if col not in full_features.columns:
                full_features[col] = 0

        if full_features.empty:
            return pd.DataFrame(columns=required)
        
        return full_features[required].iloc[-1].to_frame().T