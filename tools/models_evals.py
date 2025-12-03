"""
models_evals.py

Este script se encarga de evaluar diferentes configuraciones de modelos de Prophet
para predecir la cantidad semanal de tweets. Utiliza un enfoque de validación
walk-forward para simular el rendimiento del modelo a lo largo del tiempo.

El objetivo principal es identificar la configuración de modelo (conjunto de regresores)
y el parámetro de dispersión (`alpha` para la distribución Negative Binomial)
que minimizan el Log Loss promedio.

El script genera:
- Un resumen tabular de las métricas de rendimiento para cada configuración evaluada.
- Un gráfico comparativo de los Log Loss para las diferentes combinaciones.
- Guarda el modelo con la mejor configuración en un archivo `.pkl` para su uso posterior
  en la optimización financiera y el dashboard de producción.
"""

import logging
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet
from tabulate import tabulate

# --- SUPRESIÓN DE LOGS ---
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# --- Path Configuration & Imports ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE  # Moved here
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)

# --- HYPERPARAMETERS ---
ALPHA_CANDIDATES = [0.01, 0.05, 0.10, 0.15, 0.20]
FINAL_WINNING_FEATURES = [
    "lag_1",
    "roll_sum_7",
    "momentum",
    "last_burst",
    "is_high_regime",
    "is_regime_change",
]


def get_last_complete_friday(last_data_date: pd.Timestamp) -> datetime:
    """
    Encuentra el último viernes completo que puede iniciar una ventana de pronóstico de 7 días,
    asegurando que se disponga de datos de verdad fundamental.
    """
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    last_possible_friday = last_possible_forecast_start - timedelta(
        days=days_since_friday,
    )

    return last_possible_friday


def run_weekly_walk_forward(
    all_features_df: pd.DataFrame, regressors: list, validation_fridays: list,
) -> pd.DataFrame:
    """
    Simula predicciones semanales utilizando un enfoque de validación walk-forward.
    """
    prophet_df = all_features_df.reset_index().rename(
        columns={"date": "ds", "n_tweets": "y"},
    )
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    predictions = []
    logger.info(
        f"   -> Validando semanas: {[d.strftime('%Y-%m-%d') for d in validation_fridays]}",
    )

    for friday_date in validation_fridays:
        week_start, week_end = friday_date, friday_date + timedelta(days=6)
        df_train = prophet_df[prophet_df["ds"] < week_start]
        test_dates = pd.date_range(week_start, week_end, freq="D")

        if len(df_train) < 90:
            logger.warning(f"   ⚠️ Insuficientes datos para {friday_date.date()}")
            continue

        try:
            m = Prophet(
                growth="linear",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            for reg in regressors:
                m.add_regressor(reg)
            m.fit(df_train)

            future = pd.DataFrame({"ds": test_dates})
            if regressors:
                future = future.merge(
                    prophet_df[["ds"] + regressors], on="ds", how="left",
                ).fillna(0)

            forecast = m.predict(future)
            result_week = forecast[["ds", "yhat"]].merge(
                prophet_df[["ds", "y"]], on="ds", how="left",
            )

            for _, row in result_week.iterrows():
                predictions.append(
                    {
                        "ds": row["ds"],
                        "y_pred": max(0, row["yhat"]),
                        "y_true": row["y"],
                        "week_start": friday_date,
                    },
                )
        except Exception as e:
            logger.error(f"   ❌ Error en semana {friday_date.date()}: {e}")

    return pd.DataFrame(predictions).set_index("ds") if predictions else pd.DataFrame()


def train_best_model(all_features_df: pd.DataFrame, best_config: dict) -> dict:
    """
    Entrena el mejor modelo Prophet final con la configuración óptima.
    """
    logger.info(
        f"\n🏆 Entrenando modelo final: {best_config['name']} con distribución {best_config['distribution']} y alpha {best_config['alpha']:.4f}",
    )

    prophet_df = all_features_df.reset_index().rename(
        columns={"date": "ds", "n_tweets": "y"},
    )
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    regressors = best_config["regressors"]
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)

    m = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    for reg in regressors:
        m.add_regressor(reg)
    m.fit(prophet_df)

    return {
        "model": m,
        "model_name": best_config["name"],
        "regressors": regressors,
        "trained_on": prophet_df["ds"].max(),
        "training_samples": len(prophet_df),
        "metrics": best_config["metrics"],
    }


def get_bin_for_value(value: float, bins_config: list) -> str | None:
    """
    Determina en qué bin cae un valor.
    """
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
    """
    Evaluates a Prophet model with a specific set of regressors using walk-forward validation.
    Returns the best average Log Loss and its corresponding configuration (distribution, alpha).
    """
    last_data_date = all_features_df.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted(
        [last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)],
    )

    metrics = []

    results_df = run_weekly_walk_forward(
        all_features_df, regressors, validation_fridays,
    )
    if results_df.empty:
        logger.warning(
            "   ❌ Sin resultados de predicción para la configuración actual.",
        )
        return float("inf"), {}  # Return a very high log loss

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
                    logger.error(f"     - Error al generar probs para {dist_type}: {e}")
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
                    },
                )

    if not metrics:
        logger.error("❌ No se generaron métricas para la configuración.")
        return float("inf"), {}

    best_metric = pd.DataFrame(metrics).sort_values("Avg Log Loss").iloc[0]
    return best_metric["Avg Log Loss"], best_metric.to_dict()


def compare_prophet_feature_sets_weekly(weeks_to_validate: int = WEEKS_TO_VALIDATE):
    """
    Realiza una comparación semanal walk-forward de diferentes configuraciones de modelos.
    """
    logger.info(
        f"\n{'=' * 80}\n   VALIDACIÓN DE DISTRIBUCIÓN Y MODELOS (ÚLTIMAS {weeks_to_validate} SEMANAS)   \n{'=' * 80}\n",
    )

    logger.info("📡 Cargando y procesando datos...")
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    if "momentum" not in all_features.columns:
        logger.warning("   ⚠️ Calculando 'momentum'...")
        roll_3 = all_features["n_tweets"].rolling(3).mean().shift(1)
        roll_7 = all_features["n_tweets"].rolling(7).mean().shift(1)
        all_features["momentum"] = (roll_3 - roll_7).fillna(0)


# Global Candidates for Distribution and Bins
dist_candidates = ["nbinom", "poisson"]
bins_config = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]


def compare_prophet_feature_sets_weekly(weeks_to_validate: int = WEEKS_TO_VALIDATE):
    """
    Realiza una comparación semanal walk-forward de diferentes configuraciones de modelos.
    """
    logger.info(
        f"\n{'=' * 80}\n   VALIDACIÓN DE DISTRIBUCIÓN Y MODELOS (ÚLTIMAS {weeks_to_validate} SEMANAS)   \n{'=' * 80}\n",
    )

    logger.info("📡 Cargando y procesando datos...")
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    if "momentum" not in all_features.columns:
        logger.warning("   ⚠️ Calculando 'momentum'...")
        roll_3 = all_features["n_tweets"].rolling(3).mean().shift(1)
        roll_7 = all_features["n_tweets"].rolling(7).mean().shift(1)
        all_features["momentum"] = (roll_3 - roll_7).fillna(0)

    # Simplified model candidates for this script, feature_selector.py will manage more complex evaluations
    model_candidates = {
        "Baseline": [],
        "Dynamic_AR": ["lag_1", "last_burst", "roll_sum_7", "momentum"],
        "Final_Selected_Model": FINAL_WINNING_FEATURES,
    }

    all_evaluation_metrics = []

    for model_name, regressors in model_candidates.items():
        logger.info(
            f"\n🔍 Evaluando Modelo de Regresión: {model_name} con regressors: {regressors}",
        )
        avg_log_loss, best_config_for_model = evaluate_model_cv(
            all_features_df=all_features,
            regressors=regressors,
            weeks_to_validate=weeks_to_validate,
            alpha_candidates=ALPHA_CANDIDATES,
            dist_candidates=dist_candidates,
            bins_config=bins_config,
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
                },
            )

    if not all_evaluation_metrics:
        logger.error("❌ No se generaron métricas de evaluación.")
        return

    logger.info(f"\n{'=' * 80}\n   RESULTADOS DE VALIDACIÓN (Log Loss)   \n{'=' * 80}")
    df_metrics = pd.DataFrame(all_evaluation_metrics).sort_values("Avg Log Loss")
    logger.info(
        "\n"
        + tabulate(
            df_metrics,
            headers="keys",
            tablefmt="simple_grid",
            floatfmt=".4f",
            showindex=False,
        ),
    )

    best = df_metrics.iloc[0]

    best_text = f"🏆 MEJOR COMBINACIÓN: Modelo '{best['Model']}' con Distribución '{best['Distribution']}'"
    if best["Distribution"] == "nbinom":
        best_text += f" y Alpha = {best['Alpha']:.4f}"
    best_text += f"\n   • Log Loss Promedio: {best['Avg Log Loss']:.4f}"
    logger.success(f"\n{best_text}\n")

    # Save the best model
    best_config = {
        "name": best["Model"],
        "regressors": best["Regressors"],
        "distribution": best["Distribution"],
        "alpha": best["Alpha"] if best["Distribution"] == "nbinom" else None,
        "metrics": best.to_dict(),
    }

    final_model_package = train_best_model(all_features, best_config)

    final_model_package["best_distribution"] = best_config["distribution"]
    if best_config["distribution"] == "nbinom":
        final_model_package["best_alpha"] = best_config["alpha"]

    model_filename = f"best_prophet_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(final_model_package, f)

    logger.success(f"\n💾 Modelo guardado: {model_filename}")
    logger.info(f"   • Distribución: {final_model_package['best_distribution']}")
    if "best_alpha" in final_model_package:
        logger.info(f"   • Mejor Alpha: {final_model_package['best_alpha']:.4f}")

    # --- Plotting removed to simplify this core evaluation script ---
    # The feature_selector.py script will handle plotting if needed for its specific purpose.


if __name__ == "__main__":
    compare_prophet_feature_sets_weekly(weeks_to_validate=WEEKS_TO_VALIDATE)
