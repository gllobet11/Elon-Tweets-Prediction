# --- Path Configuration ---
import os
import sys

# Resolve the project root and add it to the system path
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import pandas as pd
from prophet import Prophet
from loguru import logger
import matplotlib.pyplot as plt

try:
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
except ImportError as e:
    logger.error(f"A critical import failed: {e}")
    sys.exit(1)

# --- Configuration ---
TUNED_HPS_FILE = os.path.join(project_root, "tuned_hyperparameters.json")
OUTPUT_PLOT_PATH = os.path.join(project_root, "prophet_components.png")


def train_model_from_hps(all_features_df: pd.DataFrame, hps: dict) -> Prophet:
    """Trains a Prophet model using a given set of hyperparameters."""
    logger.info(f"Training model with features: {hps['features']}")

    prophet_df = all_features_df.copy()
    prophet_df.index.name = "ds"
    prophet_df = prophet_df.reset_index().rename(columns={"n_tweets": "y"})

    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    regressors = hps.get("features", [])
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    prophet_params = hps.get("prophet_params", {})

    m = Prophet(
        growth="linear",
        yearly_seasonality=False,  # Set to False as weekly_seasonality is True
        weekly_seasonality=True,
        daily_seasonality=False,
        **prophet_params,
    )
    if regressors:
        for reg in regressors:
            m.add_regressor(reg)

    m.fit(prophet_df)
    return m


def visualize_prophet_components():
    logger.info("📊 Starting Prophet components visualization...")

    if not os.path.exists(TUNED_HPS_FILE):
        logger.error(
            f"Tuned hyperparameters file not found at {TUNED_HPS_FILE}. Please run Optuna tuner first."
        )
        sys.exit(1)

    with open(TUNED_HPS_FILE, "r") as f:
        hps = json.load(f)

    logger.info(f"Loaded hyperparameters: {hps}")

    # Load and prepare data
    df_tweets = load_unified_data()
    all_features_df = FeatureEngineer().process_data(df_tweets)

    # Train the model
    model = train_model_from_hps(all_features_df, hps)

    # Generate a forecast DataFrame for plotting components
    # Prophet's plot_components needs a forecast dataframe
    future = model.make_future_dataframe(periods=30)  # Forecast 30 days into the future

    # Add regressors to the future dataframe
    regressors = hps.get("features", [])
    if regressors:
        # Merge with all_features_df to get historical regressor values
        # Prepare all_features_df for merging: ensure 'ds' is timezone-naive
        all_features_df_for_merge = all_features_df.reset_index().rename(
            columns={"index": "ds"}
        )
        if all_features_df_for_merge["ds"].dt.tz is not None:
            all_features_df_for_merge["ds"] = all_features_df_for_merge[
                "ds"
            ].dt.tz_localize(None)

        future = future.merge(
            all_features_df_for_merge[["ds"] + regressors], on="ds", how="left"
        )
        # For future dates, fill with 0 or a reasonable default (e.g., last known value, mean)
        # For simplicity, filling NaNs with 0 for future regressors.
        future[regressors] = future[regressors].fillna(0)

    forecast = model.predict(future)

    # Plot components
    fig = model.plot_components(forecast)
    plt.savefig(OUTPUT_PLOT_PATH)
    logger.success(f"Prophet components plot saved to: {OUTPUT_PLOT_PATH}")
    print(f"\nFind your Prophet components plot at: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    visualize_prophet_components()
