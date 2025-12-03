import os
import sys

import pandas as pd

# Añadir el root del proyecto al path para encontrar 'src' y 'config'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer


def verify_feature_engineering():
    """
    Script de verificación para el pipeline de Feature Engineering y el split de datos.
    Carga datos, procesa features y simula el split para validar la corrección del
    error de comparación de timezones.
    """
    print("--- Verificación de Feature Engineering y Data Split ---")

    try:
        # 1. Cargar Datos Unificados
        print("📡 Cargando datos unificados...")
        df_tweets = load_unified_data()
        if df_tweets.empty:
            raise ValueError("El DataFrame de tweets está vacío.")
        print("✅ Datos unificados cargados.")

        # 2. Generación de Features
        print("\n⚙️ Ejecutando FeatureEngineer.process_data...")
        feat_eng = FeatureEngineer()
        all_features = feat_eng.process_data(df_tweets)
        print("✅ Generación de todas las features completada.")
        print(f"  -> Shape del DataFrame de features: {all_features.shape}")

        # 3. Simulación del Data Split (el punto de fallo anterior)
        print("\n🔪 Simulando el split de datos con fecha de mercado...")

        # Crear una fecha de mercado falsa, pero realista (timezone-aware)
        market_start_date_aware = (
            pd.Timestamp("2025-11-25 17:00:00")
            .tz_localize("America/New_York")
            .tz_convert("UTC")
        )
        print(f"  -> Fecha de mercado (aware) simulada: {market_start_date_aware}")

        # La corrección clave: convertirla a timezone-naive para la comparación
        market_start_date_naive = market_start_date_aware.tz_localize(None)
        print(
            f"  -> Fecha de mercado (naive) para comparación: {market_start_date_naive}",
        )

        # Realizar el split
        train_features = all_features[all_features.index < market_start_date_naive]
        predict_features = all_features.iloc[[-1]]

        print("\n✅ Split de datos realizado con éxito.")
        print(f"  -> Shape de train_features: {train_features.shape}")
        print(f"  -> Shape de predict_features: {predict_features.shape}")

        if train_features.empty or predict_features.empty:
            print("⚠️ Advertencia: Uno de los DataFrames resultantes está vacío.")
        else:
            print(
                "\n✅ Verificación completada con éxito. La corrección del split funciona.",
            )

    except Exception as e:
        import traceback

        print(f"\n❌ Ocurrió un error fatal durante la verificación: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    verify_feature_engineering()
