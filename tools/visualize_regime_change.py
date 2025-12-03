import os
import sys

import matplotlib.pyplot as plt
import pandas as pd  # Asegurarse de importar pandas
import seaborn as sns

# Añadir el root del proyecto al path para encontrar 'src' y 'config'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer


def visualize_regime_change():
    """
    Carga los datos unificados, calcula el cambio de régimen y genera una
    visualización para explicar por qué se eligió esa fecha.
    """
    print("--- Visualización del Cambio de Régimen ---")

    try:
        # 1. Cargar datos unificados
        df_tweets = load_unified_data()

        # 2. Preparar serie diaria (ya filtrada por df_tweets.created_at.min() en feat_eng)
        df_tweets["date_utc"] = df_tweets["created_at"].dt.floor("D")
        daily_counts = (
            df_tweets.groupby("date_utc")
            .size()
            .rename("n_tweets")
            .to_frame()
            .sort_index()
            .asfreq("D", fill_value=0)
        )

        # 3. Filtrar datos desde 2025 para la visualización
        daily_counts_filtered = daily_counts[daily_counts.index >= "2025-01-01"].copy()

        # 4. Calcular la fecha del cambio de régimen (usando la misma lógica del pipeline)
        # Se calcula sobre todos los datos disponibles, no solo los filtrados, para mantener la coherencia
        feat_eng = FeatureEngineer()
        regime_change_date = feat_eng._calculate_regime_change(daily_counts["n_tweets"])
        print(
            f"\n🗓️ Cambio de Régimen Detectado (basado en todos los datos): {regime_change_date.date()}",
        )

        # 5. Preparar datos para visualización (mensual)
        monthly_counts = (
            daily_counts_filtered["n_tweets"].resample("MS").sum()
        )  # 'MS' para Month Start
        monthly_diffs = monthly_counts.diff()

        # 6. Generar la visualización
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

        # --- Gráfico 1: Tweets Mensuales ---
        ax1.plot(
            monthly_counts.index,
            monthly_counts.values,
            marker="o",
            linestyle="-",
            label="Tweets por Mes",
        )

        # Marcar cambio de régimen si está en el rango visible
        if regime_change_date >= monthly_counts.index.min():
            ax1.axvline(
                regime_change_date,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Cambio de Régimen ({regime_change_date.date()})",
            )

        # Marcar evento DOGE (30 de Mayo de 2025)
        doge_event_date = pd.Timestamp("2025-05-30", tz="UTC")
        if doge_event_date >= monthly_counts.index.min():
            ax1.axvline(
                doge_event_date,
                color="purple",
                linestyle=":",
                linewidth=2,
                label="Evento DOGE (30 May 2025)",
            )

        ax1.set_title("Tweets Mensuales de Elon Musk (Año 2025)", fontsize=16)
        ax1.set_ylabel("Número de Tweets")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        # --- Gráfico 2: Diferencia Mensual ---
        ax2.bar(
            monthly_diffs.index,
            monthly_diffs.values,
            color="gray",
            label="Diferencia Mensual de Tweets",
        )

        # Marcar cambio de régimen si está en el rango visible
        if regime_change_date >= monthly_diffs.index.min():
            ax2.axvline(regime_change_date, color="r", linestyle="--", linewidth=2)

        # Marcar evento DOGE
        if doge_event_date >= monthly_diffs.index.min():
            ax2.axvline(doge_event_date, color="purple", linestyle=":", linewidth=2)

        ax2.set_title(
            "Diferencia Mensual en el Número de Tweets (Month-over-Month)", fontsize=16,
        )
        ax2.set_ylabel("Cambio en el Número de Tweets")
        ax2.set_xlabel("Fecha")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print(
            "\n📊 Gráfico generado. La línea roja marca el inicio del nuevo régimen, y la línea morada punteada el evento DOGE.",
        )

    except Exception as e:
        print(f"\n❌ Ocurrió un error fatal durante la visualización: {e}")


if __name__ == "__main__":
    visualize_regime_change()
