import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# --- Path Configuration ---
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# IMPORTANTE: Reutilizamos la lógica robusta que ya creaste
from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer
from config.settings import WEEKS_TO_VALIDATE

# Importamos la función que YA sabe manejar Holidays y evitar Leakage
from tools.model_analysis import run_parallel_walk_forward

# Rutas de los archivos JSON a comparar
COMPARISON_CONFIGS = {
    "Optuna_Best": "best_hyperparameters.json",
    # "Grid_Search": "old_hyperparameters.json" # Si tuvieras otro para comparar
}

OUTPUT_PLOT_PATH = os.path.join(
    project_root, "data", "reporting", "model_comparison.png"
)


def load_params(filename):
    path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    logger.info("🚀 Starting Model Visualization & Comparison...")

    # 1. Cargar Datos (Una sola vez)
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)

    combined_results = []

    # 2. Ejecutar Walk-Forward para cada configuración
    for label, filename in COMPARISON_CONFIGS.items():
        params = load_params(filename)
        if not params:
            continue

        logger.info(f"Evaluating config: {label} ...")

        # Extraer configuración del JSON estandarizado
        features = params.get("features", [])
        prophet_params = params.get("prophet_params", {})

        # --- AQUÍ ESTÁ LA CLAVE ---
        # Usamos run_parallel_walk_forward importado de model_analysis.
        # Esto garantiza que se usa la lógica correcta de Holidays vs Regressors
        # y que NO hay Data Leakage (porque entrena semana a semana).
        results_df = run_parallel_walk_forward(
            all_features_df=all_features,
            regressors=features,
            weeks_to_validate=WEEKS_TO_VALIDATE,
            prophet_params=prophet_params,
        )

        if not results_df.empty:
            # Agregamos por semana para limpiar el gráfico
            weekly = (
                results_df.groupby("week_start")
                .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
                .reset_index()
            )

            weekly["Model"] = label
            combined_results.append(weekly)

    if not combined_results:
        logger.error("No results generated.")
        return

    # 3. Consolidar Datos
    df_final = pd.concat(combined_results)

    # GUARDAR PARA FINANCIAL OPTIMIZER
    # Filtrar para el modelo "Optuna_Best" y guardar en historical_performance.csv
    financial_df = df_final[df_final["Model"] == "Optuna_Best"][
        ["week_start", "y_true", "y_pred"]
    ]
    historical_performance_path = os.path.join(
        project_root, "data", "processed", "historical_performance.csv"
    )
    financial_df.to_csv(historical_performance_path, index=False)
    logger.success(
        f"✅ Historical performance for financial optimization saved to: {historical_performance_path}"
    )

    # 4. Visualizar
    plt.figure(figsize=(14, 7))

    # Pintar la Realidad (Solo una vez)
    # Usamos el primer modelo para sacar los datos reales (son los mismos para todos)
    first_model_data = combined_results[0]
    sns.lineplot(
        data=first_model_data,
        x="week_start",
        y="y_true",
        label="REALIDAD (Tweets Reales)",
        color="black",
        linewidth=2.5,
        marker="o",
    )

    # Pintar las Predicciones de los modelos
    sns.lineplot(
        data=df_final,
        x="week_start",
        y="y_pred",
        hue="Model",
        style="Model",
        markers=True,
        dashes=True,
    )

    plt.title(
        f"Validación Backtest: Realidad vs Modelos (Últimas {WEEKS_TO_VALIDATE} Semanas)",
        fontsize=16,
    )
    plt.ylabel("Total Tweets Semanales", fontsize=12)
    plt.xlabel("Fecha Inicio Semana", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Leyenda", loc="upper left")

    # Guardar
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH)
    logger.success(f"📊 Gráfico guardado en: {OUTPUT_PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
