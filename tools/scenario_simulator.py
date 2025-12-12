import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.models.prophet_inference import (
        ProphetInferenceModel,
    )  # For consistent model handling
    from config.settings import WEEKS_TO_VALIDATE  # Might be used for a historical view
except ImportError as e:
    logger.error(f"A critical import failed: {e}")
    sys.exit(1)

# Define features to use for the model for scenario simulation
# These are the ones where we will manipulate values for different scenarios
NEWS_REGRESSORS = ["news_vol_log", "avg_sentiment"]
# Consider adding other relevant features that the final model might use
BASE_REGRESSORS = [
    "lag_1",
    "roll_sum_7",
    "momentum",
    "last_burst",
    "is_high_regime",
    "is_regime_change",
]
ALL_SCENARIO_REGRESSORS = (
    BASE_REGRESSORS + NEWS_REGRESSORS
)  # Add other regressors as needed


def run_scenarios(
    model_instance: ProphetInferenceModel, df_history: pd.DataFrame, periods: int = 1
):
    """
    Generates 3 predictions based on the expected news intensity using a pre-trained Prophet model.

    Args:
        model_instance: An instance of ProphetInferenceModel, already trained.
        df_history: DataFrame with historical data, including 'ds', 'y', and all regressors.
        periods: Number of future periods (weeks) to predict.

    Returns:
        A tuple of (fcst_quiet, fcst_normal, fcst_chaos) DataFrames with predictions.
    """
    if model_instance.model is None:
        raise ValueError("Model must be trained before running scenarios.")

    # Prepare df_history for Prophet's future dataframe creation
    prophet_df_history = df_history.copy()
    prophet_df_history["ds"] = prophet_df_history.index
    prophet_df_history["y"] = prophet_df_history["n_tweets"]
    prophet_df_history = prophet_df_history.reset_index(drop=True)

    # Ensure 'ds' is timezone-naive for Prophet
    if prophet_df_history["ds"].dt.tz is not None:
        prophet_df_history["ds"] = prophet_df_history["ds"].dt.tz_localize(None)

    # 1. Create the skeleton for future predictions
    future_base = model_instance.model.make_future_dataframe(
        periods=periods, freq="D"
    )  # Predict daily

    # 2. Merge historical regressors to future_base for known past dates
    # And to provide context for future feature calculation (e.g. lags)
    future_base = pd.merge(
        future_base,
        prophet_df_history[["ds"] + model_instance.regressors],
        on="ds",
        how="left",
    )

    # Calculate statistics for defining what is "Low", "Normal", and "High" for news features
    # Use only the news regressors
    stats = df_history[NEWS_REGRESSORS].describe()

    # --- Define values for the NEXT prediction periods (the last 'periods' rows) ---

    # --- SCENARIO 1: SILENCIO (Quiet) ---
    # Fewest news (25th percentile of news_vol_log) and neutral sentiment (0)
    future_quiet = future_base.copy()
    future_quiet["news_vol_log"] = future_quiet["news_vol_log"].fillna(
        stats.loc["25%", "news_vol_log"]
    )
    future_quiet["avg_sentiment"] = future_quiet["avg_sentiment"].fillna(0)  # Neutral

    # --- SCENARIO 2: BASELINE (Status Quo) ---
    # Average news (mean) and average sentiment (mean)
    future_normal = future_base.copy()
    future_normal["news_vol_log"] = future_normal["news_vol_log"].fillna(
        stats.loc["mean", "news_vol_log"]
    )
    future_normal["avg_sentiment"] = future_normal["avg_sentiment"].fillna(
        stats.loc["mean", "avg_sentiment"]
    )

    # --- SCENARIO 3: CAOS / EXPLOSIVO (High Stress - Forced Extreme) ---
    # Maximum historical news * 1.5 and extremely negative sentiment
    future_chaos = future_base.copy()
    # Force 50% more news than historical max (log scale)
    future_chaos["news_vol_log"] = future_chaos["news_vol_log"].fillna(
        stats.loc["max", "news_vol_log"] * 1.5
    )
    # Propagamos el sentimiento más negativo, multiplicado para hacerlo aún más extremo.
    future_chaos["avg_sentiment"] = future_chaos["avg_sentiment"].fillna(
        stats.loc["min", "avg_sentiment"] * 1.5
    )

    # Fill other regressors (non-news) using last known values (Prophet handles this in its prediction if not explicitly filled)
    # However, for regressors we're NOT manipulating, we should ffill their historical values into the future if they are known.
    for regressor in model_instance.regressors:
        if regressor not in NEWS_REGRESSORS:
            future_quiet[regressor] = future_quiet[regressor].ffill()
            future_normal[regressor] = future_normal[regressor].ffill()
            future_chaos[regressor] = future_chaos[regressor].ffill()

            # For features like spacex_future_launch_count, we would need external knowledge for future.
            # For now, ffill is a reasonable assumption for non-news features not explicitly handled.
            future_quiet[regressor] = future_quiet[regressor].fillna(0)
            future_normal[regressor] = future_normal[regressor].fillna(0)
            future_chaos[regressor] = future_chaos[regressor].fillna(0)

    logger.info("Predicting for scenarios: Quiet, Normal, Chaos...")
    fcst_quiet = model_instance.model.predict(future_quiet)
    fcst_normal = model_instance.model.predict(future_normal)
    fcst_chaos = model_instance.model.predict(future_chaos)

    return fcst_quiet, fcst_normal, fcst_chaos


def visualize_scenarios(
    df_history: pd.DataFrame,
    fcst_quiet: pd.DataFrame,
    fcst_normal: pd.DataFrame,
    fcst_chaos: pd.DataFrame,
    periods: int = 1,
):
    """
    Visualizes the historical data and the scenario predictions.
    """
    logger.info("--- Debugging Visualization Data ---")
    logger.info(
        f"df_history['n_tweets'] describe:\n{df_history['n_tweets'].describe()}"
    )
    logger.info(f"fcst_quiet['yhat'] describe:\n{fcst_quiet['yhat'].describe()}")
    logger.info(f"fcst_normal['yhat'] describe:\n{fcst_normal['yhat'].describe()}")
    logger.info(f"fcst_chaos['yhat'] describe:\n{fcst_chaos['yhat'].describe()}")

    plt.figure(figsize=(12, 7))

    # Limit historical data to the last N days for better visualization of recent trends
    n_historical_days_to_plot = 60  # Plot last 60 days of history
    df_history_recent = df_history.tail(n_historical_days_to_plot)

    # Plot historical data
    plt.plot(
        df_history_recent.index,
        df_history_recent["n_tweets"],
        "k-",
        label="Histórico Real",
    )

    # Ensure Prophet forecast 'ds' columns are timezone-aware UTC for consistent plotting
    if fcst_quiet["ds"].dt.tz is None:
        fcst_quiet["ds"] = fcst_quiet["ds"].dt.tz_localize("UTC")
        fcst_normal["ds"] = fcst_normal["ds"].dt.tz_localize("UTC")
        fcst_chaos["ds"] = fcst_chaos["ds"].dt.tz_localize("UTC")
    elif fcst_quiet["ds"].dt.tz != timezone.utc:
        fcst_quiet["ds"] = fcst_quiet["ds"].dt.tz_convert("UTC")
        fcst_normal["ds"] = fcst_normal["ds"].dt.tz_convert("UTC")
        fcst_chaos["ds"] = fcst_chaos["ds"].dt.tz_convert("UTC")

    # Extract relevant predictions
    future_dates = fcst_quiet["ds"].tail(periods)
    pred_quiet = fcst_quiet["yhat"].tail(periods)
    pred_normal = fcst_normal["yhat"].tail(periods)
    pred_chaos = fcst_chaos["yhat"].tail(periods)

    logger.info(
        f"Last real date: {df_history.index[-1]}, tweets: {df_history['n_tweets'].iloc[-1]}"
    )
    logger.info(f"First future date: {future_dates.iloc[0]}")
    logger.info(f"First quiet prediction: {pred_quiet.iloc[0]}")
    logger.info(f"First normal prediction: {pred_normal.iloc[0]}")
    logger.info(f"First chaos prediction: {pred_chaos.iloc[0]}")

    # Plot future predictions
    plt.plot(
        future_dates,
        pred_quiet,
        color="green",
        linestyle="--",
        marker="o",
        label="Escenario Tranquilo",
    )
    plt.plot(
        future_dates,
        pred_normal,
        color="blue",
        linestyle="--",
        marker="o",
        label="Escenario Normal",
    )
    plt.plot(
        future_dates,
        pred_chaos,
        color="red",
        linestyle="--",
        marker="o",
        label="Escenario Caos",
    )

    # Connect last historical point to first predicted point
    last_real_date = df_history_recent.index[-1]  # Use recent history for connection
    last_real_tweets = df_history_recent["n_tweets"].iloc[-1]

    plt.plot(
        [last_real_date, future_dates.iloc[0]],
        [last_real_tweets, pred_quiet.iloc[0]],
        "g--",
    )
    plt.plot(
        [last_real_date, future_dates.iloc[0]],
        [last_real_tweets, pred_normal.iloc[0]],
        "b--",
    )
    plt.plot(
        [last_real_date, future_dates.iloc[0]],
        [last_real_tweets, pred_chaos.iloc[0]],
        "r--",
    )

    plt.title(
        f"Proyección de Tuits de Elon Musk: Análisis de Escenarios (Próximos {periods} Días)"
    )
    plt.xlabel("Fecha")
    plt.ylabel("Volumen Diario de Tuits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    logger.info("--- Starting Scenario Simulation ---")

    # 1. Load and Process Data
    df_tweets_raw = load_unified_data()
    feature_engineer = FeatureEngineer()
    df_processed_features = feature_engineer.process_data(df_tweets_raw)

    # Prepare df for Prophet training: 'ds' and 'y'
    df_train = df_processed_features.copy()
    df_train["ds"] = df_train.index  # Create 'ds' column from index
    df_train["y"] = df_train["n_tweets"]  # Create 'y' column from 'n_tweets'
    df_train = df_train.reset_index(drop=True)  # Reset to integer index

    logger.info(f"Columns before timezone check: {df_train.columns.tolist()}")  # DEBUG

    # Ensure 'ds' is timezone-naive for Prophet
    if df_train["ds"].dt.tz is not None:
        df_train["ds"] = df_train["ds"].dt.tz_localize(None)

    # 2. Train the Model
    # Here we are explicitly adding the regressors used for scenarios
    # In a real setup, this would come from the best_hyperparameters.json or a predefined config.
    # For now, we will use all available features from process_data for the training
    # This might include features that will be zero in the future, like spacex_future_launch_count for non-launch days.

    # We need to filter regressors to those actually present in df_train to avoid Prophet errors
    available_regressors = [
        reg for reg in ALL_SCENARIO_REGRESSORS if reg in df_train.columns
    ]

    model_for_scenarios = ProphetInferenceModel(regressors=available_regressors)
    model_for_scenarios.train(df_train)

    # --- NEW: Print Regressor Coefficients (Suggestion A) ---
    from prophet.utilities import regressor_coefficients

    coeffs = regressor_coefficients(model_for_scenarios.model)
    logger.info("\n--- Prophet Regressor Coefficients ---")
    logger.info(coeffs.sort_values("coef", key=abs, ascending=False).to_string())
    logger.info("------------------------------------")

    # 3. Run Scenarios
    # The scenarios were originally written for weekly predictions.
    # Our data is now daily. The scenarios define values for news_vol_log and avg_sentiment.
    # We will simulate for 7 days (1 week).
    periods_to_predict = 7
    fcst_quiet, fcst_normal, fcst_chaos = run_scenarios(
        model_for_scenarios, df_processed_features, periods=periods_to_predict
    )

    # 4. Visualize Results
    visualize_scenarios(
        df_processed_features,
        fcst_quiet,
        fcst_normal,
        fcst_chaos,
        periods=periods_to_predict,
    )

    logger.info("--- Scenario Simulation Completed ---")


if __name__ == "__main__":
    main()
