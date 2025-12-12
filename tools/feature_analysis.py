import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# --- Path Configuration ---
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer

# Configuración de Estilo
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_event_impact_boxplot(df, target_col, event_col, title):
    """
    Muestra si la media de tweets cambia significativamente cuando ocurre el evento.
    """
    plt.figure()

    # Filtrar solo eventos relevantes (evitar clases desbalanceadas extremas en visualización)
    # Pero para boxplot queremos ver 0 vs 1

    sns.boxplot(x=event_col, y=target_col, data=df, showfliers=False)

    # Calcular medias
    mean_0 = df[df[event_col] == 0][target_col].mean()
    mean_1 = df[df[event_col] == 1][target_col].mean()

    plt.title(
        f"{title}\nMedia sin evento: {mean_0:.1f} vs Media con evento: {mean_1:.1f}"
    )
    plt.tight_layout()
    plt.show()


def plot_lagged_cross_correlation(df, target_col, feature_col, max_lag=14):
    """
    Muestra la correlación cruzada para ver si la reacción de Elon es retardada.
    """
    plt.figure()

    # Normalizar para xcorr
    target = (df[target_col] - df[target_col].mean()) / df[target_col].std()
    feature = (df[feature_col] - df[feature_col].mean()) / df[feature_col].std()

    lags, c, line, b = plt.xcorr(
        feature, target, maxlags=max_lag, usevlines=True, normed=True, lw=2
    )
    plt.title(f"Cross-Correlation: {feature_col} vs {target_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Imprimir el mejor lag
    best_lag_idx = np.argmax(np.abs(c))
    print(
        f" > [{feature_col}] Max Correlation at Lag: {lags[best_lag_idx]} (Corr: {c[best_lag_idx]:.3f})"
    )


def plot_continuous_relationship(df, target_col, feature_col):
    """
    Scatter plot con regresión para ver no-linealidades.
    """
    plt.figure()
    sns.regplot(
        x=feature_col,
        y=target_col,
        data=df,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
    )
    plt.title(f"Relación Directa: {feature_col} vs {target_col}")
    plt.tight_layout()
    plt.show()


def analyze_tesla_thresholds(df):
    """
    Analiza si hay un 'tipping point' en los retornos de Tesla.
    """
    if "tsla_returns" not in df.columns:
        print("Tesla returns not found.")
        return

    plt.figure()
    # Binning de retornos
    df["tsla_ret_bin"] = pd.cut(df["tsla_returns"], bins=20)
    agg = df.groupby("tsla_ret_bin", observed=True)["n_tweets"].mean()

    agg.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Promedio de Tweets según rango de Retorno de Tesla (Binning)")
    plt.ylabel("Avg Tweets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print("🔄 Cargando y procesando datos...")
    df_raw = load_unified_data()
    # Usamos FeatureEngineer para obtener las métricas base (volatilidad, proximidad, etc)
    engineer = FeatureEngineer()
    df = engineer.process_data(df_raw)

    # Target (quitamos el shift para el análisis visual directo correlacional)
    # Queremos ver: Dado el evento HOY, ¿cuántos tweets hubo HOY?
    # (Ya luego nos preocupamos del lag en el modelo)
    target = "n_tweets"

    # --- 1. ANÁLISIS DE SPACEX ---
    print("\n🚀 Analizando SpaceX...")
    # Recuperamos la columna continua si se perdió al hacer features
    # Si 'spacex_launch_proximity' es 0, es lanzamiento.
    if "spacex_launch_proximity" in df.columns:
        # Lag Analysis: ¿Anticipa o Reacciona?
        # Invertimos proximidad para que "Cerca" sea un valor alto para la correlación
        df["spacex_inv_prox"] = 14 - df["spacex_launch_proximity"]
        plot_lagged_cross_correlation(df, target, "spacex_inv_prox")

        # Boxplot del día del lanzamiento
        df["is_launch_day"] = (df["spacex_launch_proximity"] == 0).astype(int)
        plot_event_impact_boxplot(
            df, target, "is_launch_day", "Impacto Día Lanzamiento SpaceX"
        )

    # --- 2. ANÁLISIS DE TESLA ---
    print("\n🚗 Analizando Tesla...")
    if "tesla_volatility_garch" in df.columns:
        plot_continuous_relationship(df, target, "tesla_volatility_garch")
        plot_lagged_cross_correlation(df, target, "tesla_volatility_garch")

    # Análisis de Retornos (Thresholds)
    # Necesitamos recuperar los retornos crudos si no están en el DF final procesado
    # (FeatureEngineer a veces solo guarda las derivadas)
    # Para este análisis, asumimos que 'tsla_returns' o derivado está disponible o lo recalculamos rápido:
    # Este paso es conceptual, depende de si tu engineer guarda 'tsla_returns'.
    # Si no, coméntalo.

    # --- 3. ANÁLISIS DE NEWS ---
    print("\n📰 Analizando Noticias...")
    if "news_vol_log" in df.columns:
        plot_continuous_relationship(df, target, "news_vol_log")
        # Ver interacción: Noticias negativas
        if "avg_sentiment" in df.columns:
            df["bad_news_intensity"] = df["news_vol_log"] * (
                df["avg_sentiment"] < -0.2
            ).astype(int)
            plot_continuous_relationship(df, target, "bad_news_intensity")


if __name__ == "__main__":
    main()
