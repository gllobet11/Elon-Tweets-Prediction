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
from optuna.samplers import TPESampler
from itertools import product
from joblib import Parallel, delayed
from tabulate import tabulate

try:
    from src.utils.prophet_utils import extract_prophet_coefficients
    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE, ALPHA_CANDIDATES
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
except ImportError as e:
    logger.error(f"A critical import failed: {e}")
    sys.exit(1)


# --- CONFIGURATION ---
# Esta es la lista que dió 2.0718.
FINAL_WINNING_FEATURES = [
    "lag_1",
    "roll_sum_7",
    "momentum",
    "last_burst",
    "is_weekend",
    "is_sunday",
    "event_spacex_hype",
    "event_tsla_crash",
    "event_viral_news",
    "event_negative_news_spike",
    "poly_max_prob",
]

# Opción A: Añadirlo como candidato para ver si el selector lo elige
CANDIDATE_FEATURES = [
    "is_sunday",
    "is_high_regime",
    "overheat_ratio",
   # Eventos Externos (SpaceX, Tesla, News)
    "event_spacex_hype",
    "event_tsla_crash",
    "event_tsla_pump",
    "event_viral_news",
    "event_negative_news_spike",
    "news_momentum",
    
    # Features de Polymarket
    "poly_implied_mean_tweets",  # La predicción del mercado (Consenso)
    "poly_daily_vol",            # Miedo / Incertidumbre del mercado
    "poly_entropy",              # Confusión (dispersión de apuestas)
    "poly_max_prob",             # Convicción del mercado (Prob del bin ganador)
]

# Opción B: Forzarlo en el Baseline si confías mucho en él
BASELINE_FEATURES = [
    "lag_1",
    "roll_sum_7",
    "momentum",
    "last_burst",
    "is_weekend",
]
# Target for Hyperparameter Tuning
TARGET_FEATURES = FINAL_WINNING_FEATURES
BINS_CONFIG_LIST = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]
dist_candidates = ["nbinom"]


# --- HELPER FUNCTIONS ---


def prepare_prophet_data(df_features: pd.DataFrame) -> pd.DataFrame:
    df_prophet = df_features.copy()
    df_prophet.index.name = "ds"
    df_prophet = df_prophet.reset_index()
    df_prophet = df_prophet.rename(columns={"n_tweets": "y"})

    if df_prophet["ds"].dt.tz is None:
        pass
    else:
        df_prophet["ds"] = df_prophet["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

    return df_prophet


def get_last_complete_friday(last_data_date: pd.Timestamp) -> datetime:
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    return last_possible_forecast_start - timedelta(days=days_since_friday)


def extract_holidays_from_features(df_features: pd.DataFrame) -> pd.DataFrame:
    holidays_list = []
    event_cols = [c for c in df_features.columns if c.startswith("event_")]

    for col in event_cols:
        event_dates = df_features.loc[df_features[col] == 1, "ds"]
        if len(event_dates) > 0:
            temp_df = pd.DataFrame(
                {
                    "holiday": col,
                    "ds": event_dates,
                    "lower_window": 0,
                    "upper_window": 1,
                }
            )
            holidays_list.append(temp_df)

    if holidays_list:
        return pd.concat(holidays_list, ignore_index=True)
    else:
        return pd.DataFrame()


def get_bin_for_value(value: float, bins_config: list) -> str | None:
    for label, lower, upper in bins_config:
        if lower <= value < upper:
            return label
    return None


# --- CORE TRAINING LOGIC ---


def train_single_fold(
    week_start: datetime,
    df_prophet: pd.DataFrame,
    regressors: list,
    prophet_params: dict,
) -> dict | None:

    # Check de seguridad
    missing_cols = [r for r in regressors if r not in df_prophet.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}", "week": week_start}

    week_end = week_start + timedelta(days=6)
    df_train = df_prophet[df_prophet["ds"] < week_start]

    if len(df_train) < 90:
        return None

    # --- CLASIFICACIÓN DE FEATURES ---
    event_features = [r for r in regressors if r.startswith("event_")]
    continuous_features = [r for r in regressors if not r.startswith("event_")]

    # --- HOLIDAYS ---
    holidays_df_all = extract_holidays_from_features(df_prophet)

    if not holidays_df_all.empty and event_features:
        holidays_df_active = holidays_df_all[
            holidays_df_all["holiday"].isin(event_features)
        ].copy()
    else:
        holidays_df_active = None

    try:
        m = Prophet(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df_active,
            **prophet_params,
        )

        for reg in continuous_features:
            m.add_regressor(reg)

        m.fit(df_train)

        future_dates = pd.date_range(week_start, week_end, freq="D")
        future = pd.DataFrame({"ds": future_dates})

        if regressors:
            future = future.merge(
                df_prophet[["ds"] + regressors], on="ds", how="left"
            ).fillna(0)

        forecast = m.predict(future)

        result_week = forecast[["ds", "yhat"]].merge(
            df_prophet[["ds", "y"]], on="ds", how="left"
        )

        preds = []
        for _, row in result_week.iterrows():
            preds.append(
                {
                    "ds": row["ds"],
                    "y_pred": max(0, row["yhat"]),
                    "y_true": row["y"],
                    "week_start": week_start,
                }
            )
        return preds

    except Exception as e:
        return {"error": str(e), "week": week_start}


def run_parallel_walk_forward(
    all_features_df: pd.DataFrame,
    regressors: list,
    weeks_to_validate: int,
    prophet_params: dict = None,
) -> pd.DataFrame:
    if prophet_params is None:
        prophet_params = {
            "changepoint_prior_scale": 0.1,
            "seasonality_prior_scale": 0.1,
        }

    df_prophet = prepare_prophet_data(all_features_df)
    last_data_date = df_prophet["ds"].max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted(
        [last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)]
    )

    results_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(train_single_fold)(friday, df_prophet, regressors, prophet_params)
        for friday in validation_fridays
    )

    all_predictions = []
    for res in results_list:
        if res is None:
            continue
        if isinstance(res, dict) and "error" in res:
            logger.warning(f"Error in parallel fold: {res['error']}")
            continue
        all_predictions.extend(res)

    if not all_predictions:
        return pd.DataFrame()
    return pd.DataFrame(all_predictions).set_index("ds")


def calculate_metrics_from_predictions(
    results_df: pd.DataFrame,
    bins_config: list,
    alpha_candidates: list,
    dist_candidates: list,
) -> tuple[float, dict]:
    if results_df.empty:
        return float("inf"), {}

    weekly_agg = (
        results_df.dropna()
        .groupby("week_start")
        .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
        .reset_index()
    )

    best_loss = float("inf")
    best_config = {}

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
                best_config = {
                    "Distribution": dist_type,
                    "Alpha": alpha,
                    "Avg Log Loss": best_loss,
                    "Weeks Validated": len(log_losses),
                }

    return best_loss, best_config


def evaluate_model_cv(
    all_features_df: pd.DataFrame,
    regressors: list,
    weeks_to_validate: int,
    alpha_candidates: list,
    dist_candidates: list,
    bins_config: list,
    prophet_params: dict = None,
) -> tuple[float, dict]:

    results_df = run_parallel_walk_forward(
        all_features_df=all_features_df,
        regressors=regressors,
        weeks_to_validate=weeks_to_validate,
        prophet_params=prophet_params,
    )

    return calculate_metrics_from_predictions(
        results_df, bins_config, alpha_candidates, dist_candidates
    )


# --- OPTUNA OPTIMIZATION ---


def objective(trial, df_features, regressors):
    # 1. Prophet Params (Flexibilidad de tendencia)
    params = {
        "changepoint_prior_scale": trial.suggest_float(
            "changepoint_prior_scale", 0.05, 0.5, log=True 
        ),
        "seasonality_prior_scale": trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10.0, log=True
        ),
    }

    # 2. Selección de Distribución
    # Poisson es muy rígida (Media = Varianza). 
    # NBinom permite Varianza > Media (Sobredispersión).
    dist_type = trial.suggest_categorical("distribution", ["nbinom", "poisson"])

    if dist_type == "nbinom":
        # ANTES: alpha = trial.suggest_float("alpha", 0.01, 0.2)
        
        # AHORA: Rango expandido y logarítmico para cubrir desde 0.001 hasta 5.0
        alpha = trial.suggest_float("alpha", 0.001, 5.0, log=True)
        alpha_candidates = [alpha]
    else:
        alpha_candidates = [None]

    try:
        avg_loss, _ = evaluate_model_cv(
            all_features_df=df_features,
            regressors=regressors,
            weeks_to_validate=WEEKS_TO_VALIDATE,
            alpha_candidates=alpha_candidates,
            dist_candidates=[dist_type],
            bins_config=BINS_CONFIG_LIST,
            prophet_params=params,
        )
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

    return avg_loss


def run_optuna_tuning(n_trials=50):
    logger.info(f"🚀 Starting Optuna Tuning ({n_trials} trials)...")

    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    # --- CONFIGURACIÓN DE BASE DE DATOS PERSISTENTE ---
    storage_name = "sqlite:///optuna_tuning.db"
    study_name = "prophet-tuning-study"

    # load_if_exists=True permite pausar y reanudar el entrenamiento
    study = optuna.create_study(
        direction="minimize", 
        sampler=TPESampler(seed=42),
        storage=storage_name,      # <--- ESTO GUARDA EL ARCHIVO .db
        study_name=study_name,     # <--- ESTO LE DA EL NOMBRE QUE BUSCAS
        load_if_exists=True        # <--- ESTO EVITA ERRORES SI YA EXISTE
    )

    study.optimize(
        lambda trial: objective(trial, all_features, FINAL_WINNING_FEATURES),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    logger.success(f"\n🏆 Best Trial Loss: {best_trial.value:.4f}")
    logger.info("Best Params:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

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


def display_prophet_regressor_coefficients():
    import glob
    
    logger.info("🔍 Searching for the latest trained model...")

    search_patterns = [
        os.path.join(project_root, "best_prophet_model_*.pkl"),
        "best_prophet_model_*.pkl"
    ]
    
    list_of_files = []
    for pattern in search_patterns:
        list_of_files.extend(glob.glob(pattern))
    
    list_of_files = list(set(list_of_files))

    if not list_of_files:
        logger.error("❌ No trained model (.pkl) found.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    logger.info(f"📂 Loading model from: {latest_file}")

    try:
        with open(latest_file, "rb") as f:
            model_data = pickle.load(f)
        
        model = model_data.get("model")
        # --- CORRECCIÓN: Recuperamos la lista de regresores guardada ---
        regressor_names = model_data.get("regressors", [])

        if not model:
            logger.error("❌ Invalid model file.")
            return

        if not regressor_names:
            logger.warning("⚠️ No regressors found in model metadata. Trying to extract from model object...")
            # Fallback: intentar sacarlos del objeto Prophet si no están en el diccionario
            regressor_names = list(model.extra_regressors.keys())

        logger.info(f"📊 Extracting Coefficients for {len(regressor_names)} features...")
        
        # --- CORRECCIÓN: Pasamos el segundo argumento ---
        df_coeffs = extract_prophet_coefficients(model, regressor_names)

        if df_coeffs.empty:
            logger.warning("⚠️ The model has no extra regressors.")
            return

        df_coeffs["Abs_Value"] = df_coeffs["Value"].abs()
        df_coeffs = df_coeffs.sort_values(by="Abs_Value", ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print(f"🧠 FEATURE IMPORTANCE REPORT")
        print(f"📅 Model Date: {model_data.get('trained_on', 'Unknown')}")
        print("="*60)
        
        cols_to_show = ["Regressor", "Value"]
        print(tabulate(df_coeffs[cols_to_show], headers="keys", tablefmt="simple_grid", floatfmt="+.4f"))
        
        print("-" * 60)
        if "best_distribution" in model_data:
            print(f"📉 Distribution : {model_data['best_distribution']}")
        if "best_alpha" in model_data:
            print(f"🔧 Alpha (Disp) : {model_data.get('best_alpha', 0):.4f}")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"❌ Failed to display coefficients: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def run_forward_selection():
    logger.info("🚀 Starting Forward Feature Selection (Parallelized)")
    df_tweets = load_unified_data()
    all_features_df = FeatureEngineer().process_data(df_tweets)

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

    for feature in CANDIDATE_FEATURES:
        current_features = best_features + [feature]
        logger.info(f"Testing candidate: + {feature}")
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
                    f"✅ KEEP {feature}: Improved Loss by {improvement:.4f} (New: {current_loss:.4f})"
                )
                best_loss = current_loss
                best_features.append(feature)
            else:
                logger.warning(f"❌ DROP {feature}: Diff {improvement:.4f}")
        except Exception as e:
            logger.error(f"Error testing {feature}: {e}")

    logger.info("=" * 30)
    logger.info(f"🏆 FINAL WINNING MODEL FEATURES (Loss: {best_loss:.4f})")
    logger.info(best_features)
    logger.info("=" * 30)



def train_best_model(all_features_df: pd.DataFrame, best_config: dict) -> dict:
    prophet_df = prepare_prophet_data(all_features_df)
    regressors = best_config["regressors"]

    # --- LOGICA HOLIDAYS TAMBIÉN AQUÍ PARA EL MODELO FINAL ---
    event_features = [r for r in regressors if r.startswith("event_")]
    continuous_features = [r for r in regressors if not r.startswith("event_")]

    holidays_df_all = extract_holidays_from_features(prophet_df)
    if not holidays_df_all.empty and event_features:
        holidays_df_active = holidays_df_all[
            holidays_df_all["holiday"].isin(event_features)
        ].copy()
    else:
        holidays_df_active = None
    # --------------------------------------------------------

    prophet_params = best_config.get(
        "prophet_params",
        {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 0.1},
    )

    m = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df_active,
        **prophet_params,
    )
    for reg in continuous_features:
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
    logger.info(f"🚀 GENERATING ROBUST BACKTEST (LAST {weeks_to_validate} WEEKS)")
    
    # 1. Cargar Configuración Optimizada (El "Cerebro" Nuevo)
    tuned_config = {}
    try:
        with open("best_hyperparameters.json", "r") as f:
            tuned_config = json.load(f)
        logger.info("✅ Using Optimized Hyperparameters from Optuna.")
    except FileNotFoundError:
        logger.warning("⚠️ Optuna config not found. Falling back to default params.")

    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    # 2. Definir el Modelo Ganador
    # Si existe config optimizada, esa es nuestra apuesta. Si no, usamos la lista manual.
    final_features = tuned_config.get("features", FINAL_WINNING_FEATURES)
    final_params = tuned_config.get("prophet_params", {})
    
    # 3. Ejecutar Backtest Honesto (Walk-Forward)
    logger.info(f"⏳ Running Walk-Forward Validation on Best Model...")
    results_df = run_parallel_walk_forward(
        all_features, 
        final_features, 
        weeks_to_validate, 
        prophet_params=final_params
    )

    if results_df.empty:
        logger.error("❌ No predictions generated.")
        return

    # 4. Calcular Métricas
    avg_log_loss, best_dist = calculate_metrics_from_predictions(
        results_df, BINS_CONFIG_LIST, ALPHA_CANDIDATES, dist_candidates
    )
    
    print("\n" + "="*40)
    print(f"📉 BACKTEST RESULTS (Last {weeks_to_validate} Weeks)")
    print(f"   Avg Log Loss: {avg_log_loss:.4f}")
    print(f"   Model Alpha : {best_dist.get('Alpha')}")
    print("="*40 + "\n")

    # 5. GUARDAR HISTÓRICO LIMPIO (Solución del KeyError)
    # Agregamos por semana para tener una sola fila por semana
    weekly_agg = (
        results_df.dropna()
        .groupby("week_start")
        .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
        .reset_index()
    )
    
    # Renombrar para estándar estricto
    weekly_agg.rename(columns={"week_start": "week_start_date"}, inplace=True)
    
    # Asegurar que solo existen estas 3 columnas
    final_csv = weekly_agg[["week_start_date", "y_true", "y_pred"]]

    output_path = os.path.join(project_root, "data", "processed", "historical_performance.csv")
    final_csv.to_csv(output_path, index=False)
    
    logger.success(f"💾 Clean history saved to: {output_path}")
    logger.info("👉 Ready for Financial Optimizer.")


def train_with_tuned_hyperparameters():
    logger.info("🚀 Training model with the best tuned hyperparameters...")
    try:
        with open("best_hyperparameters.json", "r") as f:
            tuned_params = json.load(f)
    except FileNotFoundError:
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


# --- MAIN ---
def main():
    # Definimos el texto de ayuda para que se vea bonito en la terminal
    task_help_text = """Tarea a ejecutar:
    
    [Análisis]
    - display_feature_importance : Muestra los coeficientes del último modelo entrenado.
    - run_forward_selection      : Algoritmo 'Greedy' para seleccionar las mejores variables.
    - train_and_evaluate         : Compara el modelo actual vs Baseline (Walk-Forward CV).
    
    [Optimización y Entrenamiento]
    - run_optuna                 : Busca los mejores hiperparámetros con IA (Optuna).
    - train_with_tuned_hps       : Entrena y guarda el modelo final usando el JSON de Optuna.
    """

    parser = argparse.ArgumentParser(
        description="Elon Tweets Prediction - Model Analysis & Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,  # Esto permite saltos de línea en el help
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "display_feature_importance",
            "run_forward_selection",
            "run_optuna",
            "train_and_evaluate",
            "train_with_tuned_hps",
        ],
        help=task_help_text,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Número de intentos para la optimización de Optuna (default: 50).",
    )

    args = parser.parse_args()

    # --- Ejecución de Tareas ---
    if args.task == "display_feature_importance":
        display_prophet_regressor_coefficients()

    elif args.task == "run_forward_selection":
        run_forward_selection()

    elif args.task == "run_optuna":
        run_optuna_tuning(n_trials=args.trials)

    elif args.task == "train_and_evaluate":
        compare_prophet_feature_sets_weekly(weeks_to_validate=WEEKS_TO_VALIDATE)

    elif args.task == "train_with_tuned_hps":
        train_with_tuned_hyperparameters()


if __name__ == "__main__":
    main()
