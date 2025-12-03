import os
import sys

# Añadir la raíz del proyecto al path para poder importar desde src y config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer


def merge_and_process_for_verification():
    """
    Carga, unifica, procesa las features y guarda ambos resultados en
    archivos CSV para una depuración detallada.
    """
    print("🚀 Iniciando la fusión y procesamiento para verificación...")

    try:
        # --- 1. Cargar y Unificar ---
        df_unified = load_unified_data()

        # --- Debugging: Inspect df_curr_raw ---
        # Esto requiere una modificación temporal en unified_feed.py para exponer df_curr_raw
        # O un llamado directo a XTrackerIngestor aquí para propósitos de depuración
        print("\n--- DEBUG: Inspecting df_curr_raw from XTrackerIngestor ---")
        # xtracker_ingestor = XTrackerIngestor(data_path=os.path.join(project_root, 'data', 'raw', 'elonmusk-Elon_Musk___tweets_November_25___December_2__2025_-tweets.csv'))
        # df_curr_raw_debug = xtracker_ingestor.load_and_clean_data()
        # if df_curr_raw_debug is not None:
        #     print(f"Loaded {len(df_curr_raw_debug)} tweets from current week CSV for debug.")
        #     print("Head of df_curr_raw_debug:")
        #     print(df_curr_raw_debug.head())
        #     print("Tail of df_curr_raw_debug:")
        #     print(df_curr_raw_debug.tail())
        # else:
        #     print("df_curr_raw_debug is None.")
        # print("--- END DEBUG ---")

        output_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)

        # Guardar el resultado de la unificación
        unified_path = os.path.join(output_dir, "merged_tweets.csv")
        df_unified.to_csv(unified_path, index=False, encoding="utf-8")
        print(f"📄 Archivo de tweets unificados guardado en: {unified_path}")

        # --- 2. Procesar Features ---
        print("\n⚙️  Generando features desde los datos unificados...")
        feature_engineer = FeatureEngineer()
        all_features = feature_engineer.process_data(df_unified)

        # Guardar el resultado del feature engineering
        features_path = os.path.join(output_dir, "verified_features.csv")
        all_features.to_csv(features_path, index=True, encoding="utf-8")
        print(f"📄 Archivo de features verificado guardado en: {features_path}")

        print("\n✅ Verificación completada con éxito.")
        print(
            "🔍 Ahora puedes inspeccionar 'merged_tweets.csv' y 'verified_features.csv'.",
        )

    except Exception as e:
        print(f"❌ Error durante el proceso de verificación: {e}")


if __name__ == "__main__":
    merge_and_process_for_verification()
