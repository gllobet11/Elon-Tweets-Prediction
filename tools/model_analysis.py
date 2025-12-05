# --- Path Configuration ---
import os
import sys
# Resolve the project root and add it to the system path
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from prophet import Prophet
from itertools import product

try:
    from tabulate import tabulate
    from src.utils.prophet_utils import extract_prophet_coefficients
    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE, ALPHA_CANDIDATES
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
except ImportError as e:
    print(f"A critical import failed: {e}")
    print("Please ensure the project root is correctly added to the PYTHONPATH.")
    sys.exit(1)


# --- HYPERPARAMETERS & CONFIG ---
FINAL_WINNING_FEATURES = ["lag_1", "roll_sum_7", "momentum", "last_burst"]
BASELINE_FEATURES = ["lag_1", "roll_sum_7", "momentum", "last_burst"]
CANDIDATE_FEATURES = [
    "is_high_regime",
    "regime_intensity",
    "is_regime_change",
    "is_weekend",
    "dow",
    "cv_7",
]
TARGET_FEATURES = ["lag_1", "roll_sum_7", "momentum", "last_burst"]
BINS_CONFIG_LIST = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]
dist_candidates = ["nbinom"]


def display_prophet_regressor_coefficients():
    """
    Loads the latest saved Prophet model and displays the coefficients 
    using the centralized utility function.
    """
    # 1. Find the latest model file
    model_files = [
        f
        for f in os.listdir(project_root)
        if f.startswith("best_prophet_model_") and f.endswith(".pkl")
    ]

    if not model_files:
        print(
            "No Prophet model files found. Please ensure 'tools/model_analysis.py --task train_and_evaluate' has been run.",
        )
        return

    # Sort to get the latest model
    model_files.sort(
        key=lambda x: datetime.strptime(x, "best_prophet_model_%Y%m%d.pkl"),
        reverse=True,
    )
    latest_model_file = os.path.join(project_root, model_files[0])

    print(f"Loading latest model: {latest_model_file}")
    with open(latest_model_file, "rb") as f:
        model_package = pickle.load(f)

    m = model_package["model"]
    regressors = model_package.get("regressors", [])

    if not regressors:
        print("The loaded model was trained without any extra regressors.")
        return

    # ---------------------------------------------------------
    # REFACTORING: Use the centralized utility function
    # ---------------------------------------------------------
    print(f"Extracting coefficients for: {regressors}")
    
    df_coeffs = extract_prophet_coefficients(m, regressors)

    if not df_coeffs.empty:
        print("\n--- Prophet Regressor Coefficients (Feature Importance) ---")
        # Rename for display consistency
        df_coeffs.rename(columns={'Abs_Coefficient': 'Magnitude'}, inplace=True)
        print(tabulate(df_coeffs, headers="keys", tablefmt="simple_grid", showindex=False, floatfmt=".4f"))
        print("\nNote: A higher absolute coefficient indicates a stronger linear relationship.")
        print("Positive = correlates with higher tweet volume, Negative = correlates with lower volume.")
    else:
        print("Could not extract coefficients. The model might lack the 'beta' parameter or components.")


def run_forward_selection():
    logger.info("🚀 Starting Forward Feature Selection (Greedy Approach)")

    # 1. Load Data
    df_tweets = load_unified_data()
    all_features_df = FeatureEngineer().process_data(df_tweets)

    # 2. Baseline Evaluation
    logger.info(f"Evaluating Baseline: {BASELINE_FEATURES}")
    baseline_loss, _ = evaluate_model_cv(
        all_features_df=all_features_df,
        regressors=BASELINE_FEATURES,
        weeks_to_validate=WEEKS_TO_VALIDATE,
        alpha_candidates=ALPHA_CANDIDATES,
        dist_candidates=dist_candidates,
        bins_config=BINS_CONFIG_LIST,
    )
    logger.success(f"📉 Baseline Log Loss: {baseline_loss:.4f}")

    best_loss = baseline_loss
    best_features = BASELINE_FEATURES.copy()

    # 3. Iterative Testing
    for feature in CANDIDATE_FEATURES:
        current_features = best_features + [feature]
        logger.info(
            f"Testing candidate: + {feature} (Current features: {current_features})",
        )

        try:
            current_loss, _ = evaluate_model_cv(
                all_features_df=all_features_df,
                regressors=current_features,
                weeks_to_validate=WEEKS_TO_VALIDATE,
                alpha_candidates=ALPHA_CANDIDATES,
                dist_candidates=dist_candidates,
                bins_config=BINS_CONFIG_LIST,
            )

            improvement = best_loss - current_loss

            if improvement > 0.005:
                logger.success(
                    f"✅ KEEP {feature}: Improved Loss by {improvement:.4f} (New: {current_loss:.4f})",
                )
                best_loss = current_loss
                best_features.append(feature)
            else:
                logger.warning(
                    f"❌ DROP {feature}: No significant improvement (Diff: {improvement:.4f})",
                )

        except Exception as e:
            logger.error(f"Error testing {feature}: {e}")

    # 4. Final Verdict
    logger.info("=" * 30)
    logger.info(f"🏆 FINAL WINNING MODEL FEATURES (Loss: {best_loss:.4f})")
    logger.info(best_features)
    logger.info("=" * 30)


def tune_prophet_hyperparameters(
    df_tweets: pd.DataFrame,
    features: list,
    weeks_to_validate: int,
    alpha_candidates: list,
    dist_candidates: list,
    bins_config: list,
) -> dict:
    logger.info("🚀 Starting Prophet Hyperparameter Tuning (Walk-Forward Validation)")

    all_features_df = FeatureEngineer().process_data(df_tweets)

    prophet_param_grid = {
        "changepoint_prior_scale": [0.005, 0.01, 0.05, 0.1],
        "seasonality_prior_scale": [0.1, 1.0, 5.0, 10.0],
    }

    best_loss = float("inf")
    best_params = {}

    prophet_param_combinations = [
        dict(zip(prophet_param_grid.keys(), v))
        for v in product(*prophet_param_grid.values())
    ]

    total_combinations = (
        len(prophet_param_combinations) * len(alpha_candidates) * len(dist_candidates)
    )
    logger.info(f"Total combinations to evaluate: {total_combinations}")

    current_combination_num = 0

    # --- PREPARE DATA ONCE (Timezone Naive) ---
    df_prophet_format = all_features_df.reset_index().rename(
        columns={"index": "ds", "n_tweets": "y"}
    )

    # Ensure 'ds' is timezone-naive UTC
    if df_prophet_format["ds"].dt.tz is None:
        df_prophet_format["ds"] = df_prophet_format["ds"].dt.tz_localize("UTC")
    else:
        df_prophet_format["ds"] = df_prophet_format["ds"].dt.tz_convert("UTC")
    
    # Remove timezone info so it merges correctly with Prophet's internal naive dates
    df_prophet_format["ds"] = df_prophet_format["ds"].dt.tz_localize(None)
    # ------------------------------------------

    for prophet_params in prophet_param_combinations:
        for dist_type in dist_candidates:
            alphas_to_test = alpha_candidates if dist_type == "nbinom" else [None]

            for alpha in alphas_to_test:
                current_combination_num += 1
                logger.info(
                    f"Evaluating combination {current_combination_num}/{total_combinations}: Prophet: {prophet_params}, Dist: {dist_type}, Alpha: {alpha}"
                )

                last_data_date = all_features_df.index.max()
                last_complete_friday = get_last_complete_friday(last_data_date)
                validation_fridays = sorted(
                    [
                        last_complete_friday - timedelta(weeks=i)
                        for i in range(weeks_to_validate)
                    ]
                )

                predictions = []

                for friday_date in validation_fridays:
                    week_start, week_end = friday_date, friday_date + timedelta(days=6)
                    
                    # Use the pre-cleaned df_prophet_format
                    df_train = df_prophet_format[df_prophet_format["ds"] < week_start]
                    test_dates = pd.date_range(week_start, week_end, freq="D")

                    if len(df_train) < 90:
                        logger.warning(
                            f"   ⚠️ Insufficient data for {friday_date.date()}"
                        )
                        continue

                    try:
                        m = Prophet(
                            growth="linear",
                            yearly_seasonality=False,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            **prophet_params,
                        )
                        for reg in features:
                            m.add_regressor(reg)
                        m.fit(df_train)

                        future = pd.DataFrame({"ds": test_dates})
                        
                        # --- FIX 1: Merge with the pre-cleaned dataframe ---
                        if features:
                            future = future.merge(
                                df_prophet_format[["ds"] + features],
                                on="ds",
                                how="left",
                            ).fillna(0)
                        # -------------------------------------------------

                        forecast = m.predict(future)
                        
                        # --- FIX 2: Merge with the pre-cleaned dataframe ---
                        result_week = forecast[["ds", "yhat"]].merge(
                            df_prophet_format[["ds", "y"]],
                            on="ds",
                            how="left",
                        )
                        # -------------------------------------------------

                        for _, row in result_week.iterrows():
                            predictions.append(
                                {
                                    "ds": row["ds"],
                                    "y_pred": max(0, row["yhat"]),
                                    "y_true": row["y"],
                                    "week_start": friday_date,
                                }
                            )
                    except Exception as e:
                        logger.error(
                            f"   ❌ Error in week {friday_date.date()} with Prophet params {prophet_params}, Dist {dist_type}, Alpha {alpha}: {e}"
                        )

                if not predictions:
                    current_loss = float("inf")
                else:
                    results_df = pd.DataFrame(predictions).set_index("ds")
                    weekly_agg = (
                        results_df.dropna()
                        .groupby("week_start")
                        .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
                        .reset_index()
                    )

                    log_losses = []
                    for _, week in weekly_agg.iterrows():
                        mu, y_true = week["y_pred"], week["y_true"]
                        try:
                            probs = DistributionConverter.get_bin_probabilities(
                                mu_remainder=mu,
                                current_actuals=0,
                                model_type=dist_type,
                                alpha=alpha,
                                bins_config=bins_config,
                            )
                        except ValueError:
                            continue

                        correct_bin = get_bin_for_value(y_true, bins_config)
                        prob_correct = (
                            (probs.get(correct_bin, 0) + 1e-9) if correct_bin else 1e-9
                        )
                        log_losses.append(-np.log(prob_correct))

                    current_loss = np.mean(log_losses) if log_losses else float("inf")

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = {
                        "prophet_params": prophet_params,
                        "distribution": dist_type,
                        "alpha": alpha,
                        "log_loss": best_loss,
                        "features": features,
                    }
                    logger.success(
                        f"🏆 New best found! Log Loss: {best_loss:.4f} with params: {best_params}"
                    )

    logger.info("=" * 30)
    logger.info(f"✨ Tuning complete. Best Log Loss: {best_loss:.4f}")
    logger.info(f"Parameters: {best_params}")
    logger.info("=" * 30)

    # Save best params to a file
    import json

    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    logger.success("💾 Best hyperparameters saved to best_hyperparameters.json")

    return best_params
def get_last_complete_friday(last_data_date: pd.Timestamp) -> datetime:
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    return last_possible_forecast_start - timedelta(days=days_since_friday)


def run_weekly_walk_forward(
    all_features_df: pd.DataFrame,
    regressors: list,
    validation_fridays: list,
) -> pd.DataFrame:
    # 1. Create a copy to avoid SettingWithCopy warnings
    prophet_df = all_features_df.copy()

    # 2. Robustly prepare 'ds' and 'y' columns
    if "ds" not in prophet_df.columns:
        prophet_df.index.name = "ds"
        prophet_df = prophet_df.reset_index()

    if "n_tweets" in prophet_df.columns and "y" not in prophet_df.columns:
        prophet_df = prophet_df.rename(columns={"n_tweets": "y"})

    # --- CRITICAL FIX FOR PROPHET: REMOVE TIMEZONE ---
    # Prophet requires 'ds' to be naive (no timezone).
    # We convert to UTC first (to be safe), then strip the timezone info.
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Ensure it is UTC first
    if prophet_df["ds"].dt.tz is None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize("UTC")
    else:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_convert("UTC")

    # Now Make Naive (Remove Timezone info, keeping the UTC time)
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    # -------------------------------------------------

    # 4. Prepare Regressors
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    predictions = []

    # Clean validation dates for display
    display_dates = []
    for d in validation_fridays:
        # specific handling to safely format regardless of input type
        if hasattr(d, "strftime"):
            display_dates.append(d.strftime("%Y-%m-%d"))
        else:
            display_dates.append(str(d))

    logger.info(f"   -> Validating on weeks starting: {display_dates}")

    import pytz

    for friday_date in validation_fridays:
        # --- SYNC TIMEZONES ---
        # We must convert the validation date (week_start) to Naive UTC
        # to match the prophet_df['ds'] format we created above.

        # 1. Ensure aware UTC
        if friday_date.tzinfo is None:
            friday_date_aware = friday_date.replace(tzinfo=pytz.utc)
        else:
            friday_date_aware = friday_date.astimezone(pytz.utc)

        # 2. Make Naive
        week_start = friday_date_aware.replace(tzinfo=None)
        week_end = week_start + timedelta(days=6)

        # Filter training data (Now safe because both are Naive)
        df_train = prophet_df[prophet_df["ds"] < week_start]
        test_dates = pd.date_range(week_start, week_end, freq="D")

        if len(df_train) < 90:
            logger.warning(f"   ⚠️ Insufficient data for {week_start.date()}")
            continue

        try:
            m = Prophet(
                growth="linear",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=0.1,
            )
            for reg in regressors:
                m.add_regressor(reg)

            # FIT (Safe: df_train['ds'] is naive)
            m.fit(df_train)

            # PREDICT
            future = pd.DataFrame({"ds": test_dates})
            # Ensure future is naive
            if future["ds"].dt.tz is not None:
                future["ds"] = future["ds"].dt.tz_localize(None)

            if regressors:
                # Merge works because both 'future' and 'prophet_df' are naive
                future = future.merge(
                    prophet_df[["ds"] + regressors], on="ds", how="left"
                ).fillna(0)

            forecast = m.predict(future)

            # Merge results (Safe: both naive)
            result_week = forecast[["ds", "yhat"]].merge(
                prophet_df[["ds", "y"]], on="ds", how="left"
            )

            for _, row in result_week.iterrows():
                predictions.append(
                    {
                        "ds": row["ds"],
                        "y_pred": max(0, row["yhat"]),
                        "y_true": row["y"],
                        "week_start": friday_date,  # Keep original 'friday_date' for tracking
                    }
                )
        except Exception as e:
            logger.error(f"   ❌ Error in week {week_start.date()}: {e}")

    return pd.DataFrame(predictions).set_index("ds") if predictions else pd.DataFrame()


def get_bin_for_value(value: float, bins_config: list) -> str | None:
    for label, lower, upper in bins_config:
        if lower <= value < upper:
            return label
    return None


def evaluate_model_cv(
    all_features_df: pd.DataFrame,
    regressors: list,
    weeks_to_validate: int,
    alpha_candidates: list,
    dist_candidates: list,
    bins_config: list,
) -> tuple[float, dict]:
    last_data_date = all_features_df.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted(
        [last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)]
    )

    metrics = []
    results_df = run_weekly_walk_forward(
        all_features_df, regressors, validation_fridays
    )

    if results_df.empty:
        logger.warning("   ❌ No prediction results for the current configuration.")
        return float("inf"), {}

    weekly_agg = (
        results_df.dropna()
        .groupby("week_start")
        .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
        .reset_index()
    )

    for dist_type in dist_candidates:
        alphas_to_test = alpha_candidates if dist_type == "nbinom" else [None]
        for alpha in alphas_to_test:
            log_losses = []
            for _, week in weekly_agg.iterrows():
                mu, y_true = week["y_pred"], week["y_true"]
                try:
                    probs = DistributionConverter.get_bin_probabilities(
                        mu_remainder=mu,
                        current_actuals=0,
                        model_type=dist_type,
                        alpha=alpha,
                        bins_config=bins_config,
                    )
                except ValueError as e:
                    logger.error(
                        f"     - Error generating probabilities for {dist_type}: {e}"
                    )
                    continue

                correct_bin = get_bin_for_value(y_true, bins_config)
                prob_correct = (
                    (probs.get(correct_bin, 0) + 1e-9) if correct_bin else 1e-9
                )
                loss = -np.log(prob_correct)
                log_losses.append(loss)

            if log_losses:
                metrics.append(
                    {
                        "Distribution": dist_type,
                        "Alpha": alpha if alpha is not None else "N/A",
                        "Avg Log Loss": np.mean(log_losses),
                        "Weeks Validated": len(log_losses),
                    }
                )

    if not metrics:
        logger.error("❌ No metrics were generated for the configuration.")
        return float("inf"), {}

    best_metric = pd.DataFrame(metrics).sort_values("Avg Log Loss").iloc[0]
    return best_metric["Avg Log Loss"], best_metric.to_dict()


def train_best_model(all_features_df: pd.DataFrame, best_config: dict) -> dict:
    logger.info(
        f"\n🏆 Training final model: {best_config.get('name', 'Tuned Model')} with distribution {best_config.get('distribution')} and alpha {best_config.get('alpha'):.4f}"
    )

    # --- FIX START: Preparar el DataFrame para Prophet ---
    prophet_df = all_features_df.copy()
    prophet_df.index.name = "ds"
    prophet_df = prophet_df.reset_index()
    prophet_df = prophet_df.rename(columns={"n_tweets": "y"})
    # --- FIX END ---

    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    regressors = best_config["regressors"]
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    prophet_params = best_config.get(
        "prophet_params",
        {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 0.1},
    )

    m = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        **prophet_params,
    )
    for reg in regressors:
        m.add_regressor(reg)
    m.fit(prophet_df)

    return {
        "model": m,
        "model_name": best_config.get("name", "Tuned Model"),
        "regressors": regressors,
        "trained_on": prophet_df["ds"].max(),
        "training_samples": len(prophet_df),
        "metrics": best_config.get("metrics", {}),
    }


def compare_prophet_feature_sets_weekly(weeks_to_validate: int = WEEKS_TO_VALIDATE):
    logger.info(
        f"\n{'=' * 80}\n   MODEL AND DISTRIBUTION VALIDATION (LAST {weeks_to_validate} WEEKS)   \n{'=' * 80}\n"
    )
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    if "momentum" not in all_features.columns:
        logger.warning("   ⚠️ Calculating 'momentum'...")
        roll_3 = all_features["n_tweets"].rolling(3).mean().shift(1)
        roll_7 = all_features["n_tweets"].rolling(7).mean().shift(1)
        all_features["momentum"] = (roll_3 - roll_7).fillna(0)

    model_candidates = {
        "Baseline": [],
        "Dynamic_AR": ["lag_1", "last_burst", "roll_sum_7", "momentum"],
        "Final_Selected_Model": FINAL_WINNING_FEATURES,
    }

    all_evaluation_metrics = []
    all_historical_predictions = []
    best_metrics_dict = {}  # Initialize best_metrics_dict

    for model_name, regressors in model_candidates.items():
        logger.info(
            f"\n🔍 Evaluating Regression Model: {model_name} with regressors: {regressors}"
        )

        last_data_date = all_features.index.max()
        last_complete_friday = get_last_complete_friday(last_data_date)
        validation_fridays = sorted(
            [
                last_complete_friday - timedelta(weeks=i)
                for i in range(weeks_to_validate)
            ]
        )

        results_df = run_weekly_walk_forward(
            all_features, regressors, validation_fridays
        )

        if results_df.empty:
            logger.warning(f"   ❌ No prediction results for {model_name}.")
            continue

        weekly_agg = (
            results_df.dropna()
            .groupby("week_start")
            .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
            .reset_index()
        )
        weekly_agg["model"] = model_name
        all_historical_predictions.append(weekly_agg)

        avg_log_loss, best_config_for_model = evaluate_model_cv(
            all_features_df=all_features,
            regressors=regressors,
            weeks_to_validate=weeks_to_validate,
            alpha_candidates=ALPHA_CANDIDATES,
            dist_candidates=dist_candidates,
            bins_config=BINS_CONFIG_LIST,
        )
        if avg_log_loss != float("inf"):
            all_evaluation_metrics.append(
                {
                    "Model": model_name,
                    "Distribution": best_config_for_model["Distribution"],
                    "Alpha": best_config_for_model["Alpha"],
                    "Avg Log Loss": avg_log_loss,
                    "Weeks Validated": weeks_to_validate,
                    "Regressors": regressors,
                }
            )

        if not all_evaluation_metrics:
            logger.error("❌ No evaluation metrics were generated.")

            return

        # Save all historical predictions for visualization

        if all_historical_predictions:
            df_all_preds = pd.concat(all_historical_predictions)

            pivot_df = df_all_preds.pivot(
                index="week_start", columns="model", values=["y_pred", "y_true"]
            )

            y_true_df = pivot_df["y_true"].iloc[:, 0].rename("y_true")

            y_pred_df = pivot_df["y_pred"]

            final_preds_df = pd.concat([y_true_df, y_pred_df], axis=1)

            final_preds_df.columns = ["y_true"] + [
                f"y_pred_{col}" for col in y_pred_df.columns
            ]

            output_path_all = os.path.join(
                project_root,
                "data",
                "processed",
                "all_models_historical_performance.csv",
            )

            final_preds_df.to_csv(output_path_all)

            logger.success(f"💾 All model predictions saved to: {output_path_all}")

            # --- Save Final_Selected_Model predictions to historical_performance.csv ---

            best_metrics_series = (
                pd.DataFrame(all_evaluation_metrics).sort_values("Avg Log Loss").iloc[0]
            )

            best_metrics_dict = (
                best_metrics_series.to_dict()
            )  # best_metrics_dict is now guaranteed to be set here

            best_model_name_for_hp = best_metrics_dict["Model"]

            y_pred_col_name = f"y_pred_{best_model_name_for_hp}"

            if y_pred_col_name in final_preds_df.columns:
                historical_performance_df = final_preds_df[
                    ["y_true", y_pred_col_name]
                ].copy()

                historical_performance_df.rename(
                    columns={y_pred_col_name: "y_pred"}, inplace=True
                )

                historical_performance_df.index.name = "week_start_date"

                output_path_single = os.path.join(
                    project_root, "data", "processed", "historical_performance.csv"
                )

                historical_performance_df.to_csv(output_path_single)

                logger.success(
                    f"💾 {best_model_name_for_hp} predictions saved to: {output_path_single}"
                )

            else:
                logger.error(
                    f"Could not save {best_model_name_for_hp} predictions to historical_performance.csv: column '{y_pred_col_name}' not found."
                )

    logger.info(f"\n{'=' * 80}\n   VALIDATION RESULTS (Log Loss)   \n{'=' * 80}")
    df_metrics = pd.DataFrame(all_evaluation_metrics).sort_values("Avg Log Loss")
    logger.info(
        "\n"
        + tabulate(
            df_metrics,
            headers="keys",
            tablefmt="simple_grid",
            floatfmt=".4f",
            showindex=False,
        )
    )

    best_metrics_series = df_metrics.iloc[0]
    best_metrics_dict = best_metrics_series.to_dict()

    best_text = f"🏆 BEST COMBINATION: Model '{best_metrics_dict['Model']}' with Distribution '{best_metrics_dict['Distribution']}'"
    if best_metrics_dict["Distribution"] == "nbinom":
        best_text += f" and Alpha = {best_metrics_dict['Alpha']:.4f}"
    best_text += f"\n   • Average Log Loss: {best_metrics_dict['Avg Log Loss']:.4f}"

    best_model_name = best_metrics_dict["Model"]
    best_model_pred_col = f"y_pred_{best_model_name}"
    if (
        "y_true" in final_preds_df.columns
        and best_model_pred_col in final_preds_df.columns
    ):
        mae = (
            (final_preds_df["y_true"] - final_preds_df[best_model_pred_col])
            .abs()
            .mean()
        )
        best_text += f"\n   • Backtest MAE: {mae:.2f} tweets"
        best_metrics_dict["MAE"] = mae
    else:
        logger.warning(
            f"Could not calculate MAE. Column '{best_model_pred_col}' not found."
        )

    logger.success(f"\n{best_text}\n")

    best_config = {
        "name": best_metrics_dict["Model"],
        "regressors": best_metrics_dict["Regressors"],
        "distribution": best_metrics_dict["Distribution"],
        "alpha": best_metrics_dict["Alpha"]
        if best_metrics_dict["Distribution"] == "nbinom"
        else None,
        "metrics": best_metrics_dict,
    }

    final_model_package = train_best_model(all_features, best_config)
    final_model_package["best_distribution"] = best_config["distribution"]
    if best_config["distribution"] == "nbinom":
        final_model_package["best_alpha"] = best_config["alpha"]

    model_filename = f"best_prophet_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(final_model_package, f)

    logger.success(f"\n💾 Model saved: {model_filename}")
    logger.info(f"   • Distribution: {final_model_package['best_distribution']}")
    if "best_alpha" in final_model_package:
        logger.info(f"   • Best Alpha: {final_model_package['best_alpha']:.4f}")


def train_with_tuned_hyperparameters():
    logger.info("🚀 Training model with the best tuned hyperparameters...")
    import json

    try:
        with open("best_hyperparameters.json", "r") as f:
            tuned_params = json.load(f)
    except FileNotFoundError:
        logger.error(
            "`best_hyperparameters.json` not found. Please run the `tune_hyperparameters` task first."
        )
        return

    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    best_config = {
        "name": "Tuned_Model",
        "regressors": tuned_params["features"],
        "distribution": tuned_params["distribution"],
        "alpha": tuned_params.get("alpha"),
        "prophet_params": tuned_params.get("prophet_params", {}),
        "metrics": {"log_loss": tuned_params.get("log_loss")},
    }

    final_model_package = train_best_model(all_features, best_config)
    final_model_package["best_distribution"] = best_config["distribution"]
    if best_config["distribution"] == "nbinom":
        final_model_package["best_alpha"] = best_config["alpha"]

    model_filename = f"best_prophet_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(final_model_package, f)

    logger.success(f"\n💾 Tuned model saved: {model_filename}")
    logger.info(f"   • Distribution: {final_model_package['best_distribution']}")
    if "best_alpha" in final_model_package:
        logger.info(f"   • Best Alpha: {final_model_package['best_alpha']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Model analysis tools.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "display_feature_importance",
            "run_forward_selection",
            "tune_hyperparameters",
            "train_and_evaluate",
            "train_with_tuned_hps",
        ],
        help="The model analysis task to execute.",
    )
    args = parser.parse_args()

    if args.task == "display_feature_importance":
        display_prophet_regressor_coefficients()
    elif args.task == "run_forward_selection":
        run_forward_selection()
    elif args.task == "tune_hyperparameters":
        df_tweets_data = load_unified_data()
        best_tuned_params = tune_prophet_hyperparameters(
            df_tweets=df_tweets_data,
            features=TARGET_FEATURES,
            weeks_to_validate=WEEKS_TO_VALIDATE,
            alpha_candidates=ALPHA_CANDIDATES,
            dist_candidates=dist_candidates,
            bins_config=BINS_CONFIG_LIST,
        )
        logger.success(f"Best hyperparameters found: {best_tuned_params}")
    elif args.task == "train_and_evaluate":
        compare_prophet_feature_sets_weekly()
    elif args.task == "train_with_tuned_hps":
        train_with_tuned_hyperparameters()


if __name__ == "__main__":
    main()
