# --- Path Configuration ---
import os
import sys

# Resolve the project root and add it to the system path
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pickle
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from prophet import Prophet
import optuna
from tabulate import tabulate
import optuna
from optuna.samplers import TPESampler

try:
    from src.utils.prophet_utils import extract_prophet_coefficients
    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
    from tools.model_analysis import get_last_complete_friday, get_bin_for_value
except ImportError as e:
    logger.error(f"A critical import failed: {e}")
    logger.error("Please ensure the project root is correctly added to the PYTHONPATH.")
    sys.exit(1)


# --- CONFIGURATION ---
# Features to use for the model tuning
TARGET_FEATURES = ["lag_1", "roll_sum_7", "momentum", "last_burst", "is_weekend"]
BINS_CONFIG_LIST = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]


def run_weekly_walk_forward_for_optuna(
    all_features_df: pd.DataFrame,
    regressors: list,
    validation_fridays: list,
    prophet_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simplified version of walk-forward validation for Optuna.
    Takes prophet parameters directly.
    Returns predictions for validation weeks AND the last week of training for each split.
    """
    prophet_df = all_features_df.copy()
    prophet_df.index.name = "ds"
    prophet_df = prophet_df.reset_index().rename(columns={"n_tweets": "y"})

    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    validation_predictions = []
    training_last_week_predictions = []

    for friday_date in validation_fridays:
        week_start = friday_date.replace(tzinfo=None)
        week_end = week_start + timedelta(days=6)

        df_train = prophet_df[prophet_df["ds"] < week_start]
        test_dates = pd.date_range(week_start, week_end, freq="D")

        if len(df_train) < 90:
            continue

        try:
            m = Prophet(
                growth="linear",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                **prophet_params,
            )
            for reg in regressors:
                m.add_regressor(reg)

            m.fit(df_train)

            # --- Get Validation Predictions ---
            future = pd.DataFrame({"ds": test_dates})
            if regressors:
                future = future.merge(
                    prophet_df[["ds"] + regressors], on="ds", how="left"
                ).fillna(0)

            forecast_validation = m.predict(future)

            result_week_validation = forecast_validation[["ds", "yhat"]].merge(
                prophet_df[["ds", "y"]], on="ds", how="left"
            )

            for _, row in result_week_validation.iterrows():
                validation_predictions.append(
                    {
                        "ds": row["ds"],
                        "y_pred": max(0, row["yhat"]),
                        "y_true": row["y"],
                        "week_start": friday_date,
                    }
                )

            # --- Get Training (last week) Predictions ---
            # Define the last week of the training data
            train_end_date = df_train["ds"].max()
            train_start_date = train_end_date - timedelta(days=6)

            # Ensure last_train_week_data has features for prediction
            last_train_week_data = prophet_df[
                (prophet_df["ds"] >= train_start_date)
                & (prophet_df["ds"] <= train_end_date)
            ].copy()

            if not last_train_week_data.empty:
                forecast_training_last_week = m.predict(last_train_week_data)

                result_week_training = forecast_training_last_week[
                    ["ds", "yhat"]
                ].merge(prophet_df[["ds", "y"]], on="ds", how="left")

                for _, row in result_week_training.iterrows():
                    training_last_week_predictions.append(
                        {
                            "ds": row["ds"],
                            "y_pred": max(0, row["yhat"]),
                            "y_true": row["y"],
                            "week_start": friday_date,  # Associate with the current validation week's context
                        }
                    )
        except Exception as e:
            logger.warning(f"Error in week {week_start.date()}: {e}")

    return (
        (
            pd.DataFrame(validation_predictions).set_index("ds")
            if validation_predictions
            else pd.DataFrame()
        ),
        (
            pd.DataFrame(training_last_week_predictions).set_index("ds")
            if training_last_week_predictions
            else pd.DataFrame()
        ),
    )


def objective(trial, df_features):
    """
    Función objetivo para Optuna.
    Usa la infraestructura existente (evaluate_model_cv) para no perder lógica.
    """
    # 1. Sugerir Hiperparámetros
    params = {
        "changepoint_prior_scale": trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5, log=True
        ),
        "seasonality_prior_scale": trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10.0, log=True
        ),
        # Puedes añadir más si quieres, ej: 'holidays_prior_scale'
    }

    dist_type = trial.suggest_categorical("distribution", ["nbinom", "poisson"])

    # Manejo de Alpha para nbinom
    if dist_type == "nbinom":
        # Optuna elige UN alpha específico para esta prueba
        alpha = trial.suggest_float("alpha", 0.01, 0.2)
        alpha_candidates = [alpha]
    else:
        alpha_candidates = [None]

    # 2. Ejecutar Evaluación (Reutilizando tu función robusta)
    # Pasamos listas de 1 elemento para dist y alpha porque Optuna ya decidió cuál probar
    try:
        avg_loss, _ = evaluate_model_cv(
            all_features_df=df_features,
            regressors=FINAL_WINNING_FEATURES,  # Usamos la lista ganadora global
            weeks_to_validate=WEEKS_TO_VALIDATE,
            alpha_candidates=alpha_candidates,
            dist_candidates=[dist_type],
            bins_config=BINS_CONFIG_LIST,
            prophet_params=params,
        )
    except Exception as e:
        # Si Prophet falla (ej. matriz singular), podar el trial
        logger.warning(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

    return avg_loss


def run_optuna_tuning(n_trials=50):
    logger.info(f"🚀 Starting Optuna Tuning ({n_trials} trials)...")

    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    # Crear estudio
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))

    # Optimizar
    # Usamos lambda para pasar el dataframe fijo
    study.optimize(lambda trial: objective(trial, all_features), n_trials=n_trials)

    # Resultados
    best_trial = study.best_trial
    logger.success(f"\n🏆 Best Trial Loss: {best_trial.value:.4f}")
    logger.info("Best Params:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    # Guardar JSON
    best_params_formatted = {
        "prophet_params": {
            "changepoint_prior_scale": best_trial.params["changepoint_prior_scale"],
            "seasonality_prior_scale": best_trial.params["seasonality_prior_scale"],
        },
        "distribution": best_trial.params["distribution"],
        "alpha": best_trial.params.get("alpha"),
        "log_loss": best_trial.value,
        "features": FINAL_WINNING_FEATURES,
    }

    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params_formatted, f, indent=4)
    logger.success("💾 Saved best_hyperparameters.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    # Añadir argumento opcional para número de trials
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    args = parser.parse_args()

    logger.info("🚀 Starting Hyperparameter Tuning with Optuna")
    logger.info(f"Number of trials: {args.trials}")
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Storage: {args.storage}")

    # Load and prepare data
    df_tweets = load_unified_data()
    all_features_df = FeatureEngineer().process_data(df_tweets)

    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    # Wrap objective to pass static data
    objective_with_data = lambda trial: objective(
        trial, all_features_df, TARGET_FEATURES
    )

    # Run the optimization
    study.optimize(
        objective_with_data, n_trials=args.trials, timeout=1200
    )  # 20 minutes timeout

    # --- Results ---
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    logger.info("\n" + "=" * 30)
    logger.info("✨ OPTIMIZATION COMPLETE ✨")
    logger.info(f"Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    logger.success(f"\n🏆 Best trial (minimized validation log loss):")
    logger.success(f"  Value (Avg Validation Log Loss): {best_trial.value:.4f}")
    if "avg_log_loss_training" in best_trial.user_attrs:
        logger.success(
            f"  Corresponding Avg Training Log Loss: {best_trial.user_attrs['avg_log_loss_training']:.4f}"
        )
    logger.info("\n  Best Parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    - {key}: {value}")

    # --- Collect and Display All Complete Trials' Metrics ---
    logger.info("\n" + "=" * 50)
    logger.info("📊 Detailed Results for Complete Trials")
    logger.info("=" * 50)

    trial_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_info = {
                "Trial #": trial.number,
                "Avg_Validation_Log_Loss": trial.value,
                "Avg_Training_Log_Loss": trial.user_attrs.get(
                    "avg_log_loss_training", np.nan
                ),
                **trial.params,
            }
            trial_data.append(trial_info)

    if trial_data:
        df_trials = pd.DataFrame(trial_data)
        df_trials = df_trials.sort_values(by="Avg_Validation_Log_Loss").reset_index(
            drop=True
        )

        logger.info("\nTop 10 Trials (sorted by Avg Validation Log Loss):")
        logger.info(
            "\n"
            + tabulate(
                df_trials.head(10),
                headers="keys",
                tablefmt="simple_grid",
                floatfmt=".4f",
                showindex=False,
            )
        )

        # Visualize Training vs Validation Log Loss
        if not df_trials.empty and "Avg_Training_Log_Loss" in df_trials.columns:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                data=df_trials,
                x="Avg_Training_Log_Loss",
                y="Avg_Validation_Log_Loss",
                hue="distribution",
                size="Avg_Validation_Log_Loss",
                sizes=(20, 200),
                alpha=0.7,
            )
            min_val = min(
                df_trials["Avg_Training_Log_Loss"].min(),
                df_trials["Avg_Validation_Log_Loss"].min(),
            )
            max_val = max(
                df_trials["Avg_Training_Log_Loss"].max(),
                df_trials["Avg_Validation_Log_Loss"].max(),
            )
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Ideal Generalization (Train=Validation)",
            )
            plt.title("Optuna Trial: Avg Training Log Loss vs. Avg Validation Log Loss")
            plt.xlabel("Average Training Log Loss")
            plt.ylabel("Average Validation Log Loss")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Distribution")
            plt.tight_layout()
            plt.savefig("optuna_train_vs_validation_log_loss.png")
            logger.success(
                "💾 Saved Optuna train vs. validation log loss plot to optuna_train_vs_validation_log_loss.png"
            )
            plt.show()

    else:
        logger.info("No complete trials to display detailed results.")

    # --- Standardize and Save Results ---
    # Create the standardized dictionary
    standardized_hps = {
        "prophet_params": {
            "changepoint_prior_scale": best_trial.params.get("changepoint_prior_scale"),
            "seasonality_prior_scale": best_trial.params.get("seasonality_prior_scale"),
        },
        "distribution": best_trial.params.get("distribution"),
        "alpha": best_trial.params.get(
            "alpha"
        ),  # Will be None if distribution is not 'nbinom'
        "log_loss": best_trial.value,
        "features": TARGET_FEATURES,
    }

    output_filename = "tuned_hyperparameters.json"
    with open(output_filename, "w") as f:
        json.dump(standardized_hps, f, indent=4)
    logger.success(
        f"\n💾 Best hyperparameters saved in standardized format to: {output_filename}"
    )


if __name__ == "__main__":
    main()
