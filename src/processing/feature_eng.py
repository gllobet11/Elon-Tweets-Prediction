import os
import json
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone
import yfinance as yf
from arch import arch_model
from loguru import logger
from transformers import pipeline
import streamlit as st # Necesario para st.cache_resource

def get_spacex_launches_2025():
    """
    Generates a synthetic, realistic SpaceX launch schedule for 2025.
    """
    try:
        url = "https://api.spacexdata.com/v4/launches/past"
        
        # --- Robust Request with Retries and Timeout ---
        session = requests.Session()
        retry_strategy = Retry(
            total=5, # Number of retries
            backoff_factor=1, # Wait 1, 2, 4, 8, 16 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these status codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        response = session.get(url, timeout=10) # Set a timeout of 10 seconds
        response.raise_for_status()
        launches = response.json()

        # Corrección: datetime.utcfromtimestamp está deprecado. Usamos timezone.utc
        launch_dates_hist = sorted(
            [
                datetime.fromtimestamp(l["date_unix"], tz=timezone.utc)
                for l in launches
                if l.get("date_unix")
                and datetime.fromtimestamp(l["date_unix"], tz=timezone.utc).year >= 2022
            ]
        )

        if len(launch_dates_hist) < 2:
            raise ValueError("Not enough historical launches to determine frequency.")

        time_deltas = pd.Series(launch_dates_hist).diff().dt.total_seconds() / (24 * 3600)
        avg_days_between_launches = time_deltas.mean()

        if pd.isna(avg_days_between_launches) or avg_days_between_launches <= 0:
            avg_days_between_launches = 4

        logger.info(f"Historical analysis: Average days between launches is {avg_days_between_launches:.2f} days.")

        synthetic_dates = []
        current_date = datetime(2025, 1, 1)
        while current_date.year == 2025:
            synthetic_dates.append(current_date)
            current_date += timedelta(days=round(avg_days_between_launches))

        return pd.to_datetime(synthetic_dates, utc=True)

    except Exception as e:
        logger.error(f"Error generating synthetic launch dates: {e}. Using a fixed fallback.")
        fallback_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="4D", tz="UTC")
        return pd.to_datetime(fallback_dates, utc=True)


def calculate_launch_proximity(dates_series, launch_dates):
    """
    Calculates the minimum number of days to the nearest SpaceX launch.
    """
    if launch_dates.empty:
        return pd.Series(14, index=dates_series, dtype=int)

    if not isinstance(dates_series, pd.DatetimeIndex):
        dates_series = pd.to_datetime(dates_series)

    if dates_series.tz is None:
        dates_series = dates_series.tz_localize("UTC")
    if launch_dates.tz is None:
        launch_dates = launch_dates.tz_localize("UTC")

    min_days_to_launch = dates_series.to_series().apply(
        lambda x: min(abs((x - ld).days) for ld in launch_dates)
        if not launch_dates.empty
        else 14
    )

    return min_days_to_launch.clip(upper=14)


def calculate_tesla_garch_volatility(start_date: str, end_date: str) -> pd.Series:
    """
    Calculates the conditional volatility of TSLA stock price using a GARCH(1,1) model.
    """
    try:
        logger.info(f"Fetching TSLA historical data from {start_date} to {end_date}...")
        tsla_data = yf.download("TSLA", start=start_date, end=end_date, progress=False)

        # Manejo seguro si yfinance devuelve MultiIndex
        if isinstance(tsla_data.columns, pd.MultiIndex):
            try:
                close_data = tsla_data.xs('Close', level=0, axis=1) 
                if close_data.empty: 
                     close_data = tsla_data["Close"]
            except:
                close_data = tsla_data.iloc[:, 0]
        else:
             close_data = tsla_data["Close"] if "Close" in tsla_data.columns else tsla_data.iloc[:, 0]

        returns = close_data.pct_change().dropna()

        if returns.empty:
            logger.warning("No returns to calculate. Returning empty series.")
            return pd.Series(dtype=float)

        garch_model = arch_model(100 * returns, vol="Garch", p=1, q=1, mean="Constant", dist="t")
        res = garch_model.fit(update_freq=5, disp="off")
        
        conditional_volatility = res.conditional_volatility

        # --- CORRECCIÓN DE ZONA HORARIA ---
        # Convertimos todo a UTC explícito para que coincida con los tweets
        if conditional_volatility.index.tz is None:
            conditional_volatility.index = conditional_volatility.index.tz_localize('UTC')
        else:
            conditional_volatility.index = conditional_volatility.index.tz_convert('UTC')

        # El índice completo también debe ser UTC
        full_idx = pd.date_range(start=returns.index.min(), end=returns.index.max(), freq="D", tz='UTC')
        daily_volatility = conditional_volatility.reindex(full_idx, method="ffill")

        return daily_volatility

    except Exception as e:
        logger.error(f"Error calculating TSLA GARCH volatility: {e}")
        return pd.Series(dtype=float)


class FeatureEngineer:
    def __init__(self):
        self.rolling_window_size = 28
        self.regime_threshold_z_score = 1.0

    def _resample_to_daily(self, df_granular: pd.DataFrame) -> pd.DataFrame:
        """Resamples granular tweet data to a daily DataFrame with a 'n_tweets' column."""
        df_copy = df_granular.copy()

        # Handle case where 'created_at' is already the index, or is a column
        if not isinstance(df_copy.index, pd.DatetimeIndex) and 'created_at' not in df_copy.columns:
             raise ValueError("Input for resampling must have 'created_at' column or a DatetimeIndex.")
        
        if 'created_at' in df_copy.columns:
            df_copy["created_at"] = pd.to_datetime(df_copy["created_at"], utc=True)
            df_copy = df_copy.set_index("created_at")
        
        # By now, we can assume the index is a DatetimeIndex
        df_daily = df_copy.resample("D").size().to_frame("n_tweets")
        
        if not df_daily.empty:
            all_days = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
            df_daily = df_daily.reindex(all_days, fill_value=0)
        
        return df_daily

    def _calculate_regime_change(self, daily_series):
        if daily_series.empty:
            return pd.Series(0, index=daily_series.index), pd.Series(0, index=daily_series.index), pd.Series(0, index=daily_series.index), pd.Series(0, index=daily_series.index)

        rolling_mean = daily_series.rolling(window=self.rolling_window_size, min_periods=1).mean()
        rolling_std = daily_series.rolling(window=self.rolling_window_size, min_periods=1).std()

        safe_rolling_std = rolling_std.replace(0, pd.NA)
        z_score = (daily_series - rolling_mean) / safe_rolling_std
        z_score = z_score.fillna(0)

        is_high_regime = (z_score > self.regime_threshold_z_score).astype(int)
        is_low_regime = (z_score < -self.regime_threshold_z_score).astype(int)

        is_regime_change = ((is_high_regime.diff().abs() > 0) | (is_low_regime.diff().abs() > 0)).astype(int)
        is_regime_change.iloc[0] = 0

        return z_score, is_high_regime, is_low_regime, is_regime_change

    def _add_technical_features(self, df):
        df["lag_1"] = df["n_tweets"].shift(1)
        df["roll_sum_3"] = df["n_tweets"].rolling(window=3).sum()
        df["roll_sum_7"] = df["n_tweets"].rolling(window=7).sum()
        df["momentum"] = df["n_tweets"].diff(periods=7)

        roll_mean_7 = df["n_tweets"].rolling(window=7).mean().shift(1)
        df["last_burst"] = (df["n_tweets"] > (roll_mean_7 * 1.5)).astype(int)

        df["dow"] = df.index.dayofweek
        df["is_weekend"] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

        roll_std_7 = df["n_tweets"].rolling(window=7).std().shift(1)
        df["cv_7"] = roll_std_7 / roll_mean_7
        df["cv_7"] = df["cv_7"].fillna(0)
        
        return df

    def process_data(self, df_raw):
        if df_raw.empty:
            return pd.DataFrame()
        
        # Check if df_raw is already a daily, processed-like DataFrame
        # by checking if it has 'n_tweets' and a DatetimeIndex
        if isinstance(df_raw.index, pd.DatetimeIndex) and 'n_tweets' in df_raw.columns:
            df = df_raw.copy()
        else:
            # Assume it's raw granular data and resample
            df = self._resample_to_daily(df_raw)
        
        df = df.copy().sort_index()

        df = self._add_technical_features(df)
        
        z, high, low, change = self._calculate_regime_change(df['n_tweets'])
        df['is_high_regime'] = high
        df['is_regime_change'] = change
        df['regime_intensity'] = z
        
        try:
            launches = get_spacex_launches_2025()
            proximity = calculate_launch_proximity(df.index, launches)
            df['spacex_launch_proximity'] = proximity
        except Exception as e:
            logger.error(f"Fallo al integrar SpaceX data: {e}")
            df['spacex_launch_proximity'] = 14
            
        try:
            start_str = df.index.min().strftime('%Y-%m-%d')
            end_str = df.index.max().strftime('%Y-%m-%d')
            volatility_series = calculate_tesla_garch_volatility(start_str, end_str)
            df['tesla_volatility_garch'] = volatility_series.reindex(df.index, method='ffill').fillna(0.02)
        except Exception as e:
            logger.error(f"Fallo al integrar Tesla data: {e}")
            df['tesla_volatility_garch'] = 0.02

        if 'event_magnitude' not in df.columns: df['event_magnitude'] = 0
        if 'reply_momentum' not in df.columns: df['reply_momentum'] = 0
        if 'news_sentiment_spike' not in df.columns: df['news_sentiment_spike'] = 0

        df = df.fillna(0)
        return df

    def generate_live_features(self, df_raw):
        """
        Retorna la ÚLTIMA fila de features procesadas.
        """
        full_features = self.process_data(df_raw)

        model_required_features = [
            "lag_1", "roll_sum_7", "momentum", "last_burst",
            "is_high_regime", "is_regime_change", "event_magnitude",
            "reply_momentum", "spacex_launch_proximity",
            "tesla_volatility_garch", "news_sentiment_spike",
        ]

        missing_features = [f for f in model_required_features if f not in full_features.columns]
        for f in missing_features:
            full_features[f] = 0

        if full_features.empty:
            return pd.DataFrame(columns=model_required_features)

        live_row = full_features[model_required_features].iloc[-1].to_frame().T
        return live_row