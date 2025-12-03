"""
generate_historical_performance.py

This script generates historical performance data by running a backtest over a specified number of weeks.
It adapts the backtesting logic from the financial_optimizer.py script to produce a clean CSV file
containing the model's predictions (y_pred) and the actual outcomes (y_true) for each week.

This data is then used by the main dashboard to display historical performance.
"""

import os
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
    Generates backtest predictions using the best Prophet model (Dynamic_AR).
    This function is adapted from financial_optimizer.py.

    Args:
        weeks_to_validate (int): The number of past weeks to generate predictions for.
        end_date (date, optional): The reference end date for the backtest.
                                   If None, uses the latest data available. Defaults to None.
    """
    print("⚙️  Generating backtest predictions...")
    df_tweets = load_unified_data()

    # Filter data based on end_date for testability
    if end_date:
        print(f"   -> Filtering data up to {end_date}")
        df_tweets = df_tweets[df_tweets["created_at"].dt.date <= end_date]

    all_features = FeatureEngineer().process_data(df_tweets)

    if "momentum" not in all_features.columns:
        roll_3 = all_features["n_tweets"].rolling(3).mean().shift(1)
        roll_7 = all_features["n_tweets"].rolling(7).mean().shift(1)
        all_features["momentum"] = (roll_3 - roll_7).fillna(0)

    last_data_date = all_features.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted(
        [last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)],
    )

    regressors = ["lag_1", "last_burst", "roll_sum_7", "momentum"]

    prophet_df = all_features.reset_index().rename(
        columns={"date": "ds", "n_tweets": "y"},
    )
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    for col in [r for r in regressors if r not in prophet_df.columns]:
        prophet_df[col] = 0.0
    prophet_df[regressors] = prophet_df[regressors].fillna(0)

    predictions = []
    if not validation_fridays:
        print("   -> No validation Fridays found. Skipping prediction generation.")
        return pd.DataFrame()

    print(
        f"   -> Validating {len(validation_fridays)} weeks from {validation_fridays[0].date()}",
    )

    for friday_date in validation_fridays:
        week_start = friday_date

        df_train = prophet_df[prophet_df["ds"] < week_start]
        test_dates = pd.date_range(week_start, periods=7, freq="D")

        if len(df_train) < 90:
            continue

        m = Prophet(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        for reg in regressors:
            m.add_regressor(reg)
        m.fit(df_train)  # Suppress Stan output

        future = pd.DataFrame({"ds": test_dates})
        future = future.merge(
            prophet_df[["ds"] + regressors], on="ds", how="left",
        ).fillna(0)

        forecast = m.predict(future)
        result_week = forecast[["ds", "yhat"]].merge(
            prophet_df[["ds", "y"]], on="ds", how="left",
        )
        # Assign the week's starting date as an ID for proper grouping
        result_week["week_start_date"] = friday_date
        predictions.append(result_week)

    if not predictions:
        raise ValueError("Could not generate backtest predictions.")

    df_pred = pd.concat(predictions)

    # Group by the explicitly assigned week_start_date, not a Grouper
    df_weekly = (
        df_pred.groupby("week_start_date")
        .agg(y_true=("y", "sum"), y_pred=("yhat", "sum"))
        .dropna()
    )

    # Filter out weeks that might be incomplete
    df_weekly = df_weekly[df_weekly["y_true"] > 100]

    # Rename index to be clear it's the week start date
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
