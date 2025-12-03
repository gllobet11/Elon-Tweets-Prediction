"""
generate_historical_performance.py

This script generates historical performance data by running a backtest over a specified number of weeks.
It adapts the backtesting logic from the financial_optimizer.py script to produce a clean CSV file
containing the model's predictions (y_pred) and the actual outcomes (y_true) for each week.

This data is then used by the main dashboard to display historical performance.
"""

import glob
import os
import pickle # Added this
import sys
from datetime import timedelta

import pandas as pd
from prophet import Prophet

# --- Path Configuration ---
# This ensures that the script can be run from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Get WEEKS_TO_VALIDATE from the central settings file
    from config.settings import WEEKS_TO_VALIDATE
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.utils import get_last_complete_friday
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error during import: {e}")
    print(
        "Please ensure you are running this script from the project's root directory.",
    )
    sys.exit(1)

# --- Configuration ---
OUTPUT_PATH = os.path.join(
    project_root, "data", "processed", "historical_performance.csv",
)


def generate_backtest_predictions(weeks_to_validate: int, end_date=None):
    """
    Generates backtest predictions using the best Prophet model.
    """
    print("⚙️  Generating backtest predictions...")
    
    # 1. Load the best Prophet model
    model_files = glob.glob("best_prophet_model_*.pkl")
    if not model_files:
        raise FileNotFoundError(
            "No se encontró ningún archivo de modelo Prophet (.pkl). Ejecuta `tools/models_evals.py` primero para entrenar y guardar el modelo.",
        )
    latest_model_path = max(model_files, key=os.path.getmtime)
    with open(latest_model_path, "rb") as f:
        model_package = pickle.load(f)
    
    m = model_package['model']
    best_config_regressors = model_package['regressors']
    
    print(f"   -> Model '{model_package.get('model_name', 'Unknown')}' loaded from '{os.path.basename(latest_model_path)}'.")
    print(f"   -> Using regressors: {best_config_regressors}")

    # 2. Load and process features
    df_tweets = load_unified_data()

    # Filter data based on end_date for testability
    if end_date:
        print(f"   -> Filtering data up to {end_date}")
        df_tweets = df_tweets[df_tweets["created_at"].dt.date <= end_date]

    all_features = FeatureEngineer().process_data(df_tweets)

    # Prepare Prophet DataFrame format
    prophet_df = all_features.reset_index().rename(
        columns={"date": "ds", "n_tweets": "y"},
    )
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    # Ensure all regressors are present and filled
    if best_config_regressors:
        for col in [r for r in best_config_regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[best_config_regressors] = prophet_df[best_config_regressors].fillna(0)

    # 3. Setup walk-forward validation
    last_data_date = all_features.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted(
        [last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)],
    )

    predictions_data = []
    if not validation_fridays:
        print("   -> No validation Fridays found. Skipping prediction generation.")
        return pd.DataFrame()

    print(
        f"   -> Validating {len(validation_fridays)} weeks from {validation_fridays[0].date()}",
    )

    for friday_date in validation_fridays:
        week_start = friday_date

        df_train_current = prophet_df[prophet_df["ds"] < week_start]
        test_dates = pd.date_range(week_start, periods=7, freq="D")

        if len(df_train_current) < 90:
            print(f"   ⚠️ Insufficient data for {friday_date.date()}, skipping.")
            continue

        # Prepare future DataFrame for prediction
        future = pd.DataFrame({"ds": test_dates})
        if best_config_regressors:
            future = future.merge(
                prophet_df[["ds"] + best_config_regressors], on="ds", how="left",
            ).fillna(0)

        # Make prediction
        forecast = m.predict(future)
        
        # Merge predictions with true values and capture uncertainty intervals
        result_week = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            prophet_df[["ds", "y"]], on="ds", how="left",
        )
        result_week["week_start_date"] = friday_date
        predictions_data.append(result_week)

    if not predictions_data:
        raise ValueError("Could not generate backtest predictions.")

    df_pred_all = pd.concat(predictions_data)

    # Aggregate to weekly sums
    df_weekly = (
        df_pred_all.groupby("week_start_date")
        .agg(
            y_true=("y", "sum"),
            y_pred=("yhat", "sum"),
            y_pred_lower=("yhat_lower", "sum"),
            y_pred_upper=("yhat_upper", "sum"),
        )
        .dropna()
    )

    # Filter out weeks that might be incomplete (if actual y_true < 100)
    df_weekly = df_weekly[df_weekly["y_true"] > 100] # Use original filter

    df_weekly.index.name = "week_start_date"

    print(f"✅ Predictions generated for {len(df_weekly)} valid weeks.")
    return df_weekly.reset_index()


if __name__ == "__main__":
    try:
        print("Starting historical performance generation...")
        df_performance = generate_backtest_predictions(
            weeks_to_validate=WEEKS_TO_VALIDATE,
        )

        # Save to CSV
        df_performance.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Historical performance data saved to {OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback

        traceback.print_exc()
