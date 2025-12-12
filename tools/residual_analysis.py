import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Path Configuration ---
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.model_analysis import (
    run_parallel_walk_forward,
    prepare_prophet_data,
    extract_holidays_from_features,
)
from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer
from config.settings import WEEKS_TO_VALIDATE


def analyze_residuals():
    print("🚀 Entrenando modelo actual para extraer residuos...")

    # 1. Cargar y Procesar
    df_raw = load_unified_data()
    engineer = FeatureEngineer()
    all_features = engineer.process_data(df_raw)

    # 2. Definir el modelo "Casi Ganador" actual
    # Usamos las features que dieron 2.13
    current_features = [
        "lag_1",
        "roll_sum_7",
        "momentum",
        "last_burst",
        "event_tsla_crash",
    ]

    # 3. Obtener predicciones (Backtest)
    results = run_parallel_walk_forward(
        all_features, current_features, WEEKS_TO_VALIDATE
    )

    if results.empty:
        print("❌ No se generaron predicciones.")
        return

    # 4. Calcular Residuos
    # Residuo = Realidad - Predicción
    # Positivo = El modelo se quedó corto (Elon tuiteó más de lo esperado)
    # Negativo = El modelo se pasó (Elon tuiteó menos)
    results["residual"] = results["y_true"] - results["y_pred"]

    # Unir con las features originales para buscar correlaciones
    # Necesitamos las features en 'results', hacemos merge por fecha
    df_prophet = prepare_prophet_data(all_features)
    analysis_df = results.merge(df_prophet, on="ds", how="left", suffixes=("", "_orig"))

    print("\n🔍 TOP CORRELACIONES CON EL ERROR (RESIDUO):")
    print(
        "Si una variable tiene alta correlación, significa que explica lo que el modelo NO sabe."
    )

    # Seleccionar solo columnas numéricas y excluir las obvias
    numeric_cols = analysis_df.select_dtypes(include=np.number).columns
    corrs = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["residual"])
        .sort_values(ascending=False)
    )

    print(corrs.drop(["y_true", "y_pred", "residual", "y"]).head(10))
    print("\n👇 VARIABLES QUE HACEN QUE EL MODELO SOBRESTIME (Correlación Negativa):")
    print(corrs.drop(["y_true", "y_pred", "residual", "y"]).tail(5))

    # 5. Visualizar el Error en Días Específicos
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="is_sunday", y="residual", data=analysis_df)
    plt.title("¿Falla el modelo sistemáticamente los domingos?")
    plt.axhline(0, color="r", linestyle="--")
    plt.show()

    # Visualizar error vs News
    if "news_momentum" in analysis_df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x="news_momentum", y="residual", data=analysis_df, alpha=0.5)
        plt.title(
            "Residuos vs News Momentum (¿Nos quedamos cortos cuando hay muchas noticias?)"
        )
        plt.axhline(0, color="r", linestyle="--")
        plt.show()


if __name__ == "__main__":
    analyze_residuals()
