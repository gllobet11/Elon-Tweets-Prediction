import sys
import os
import pandas as pd

# Path Configuration
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
except Exception as e:
    print(f"--- ERROR FATAL EN LA CONFIGURACIÓN INICIAL ---")
    import traceback
    print(f"Error: {e}")
    sys.exit(1)

def check_max_data_date():
    """
    Loads unified data, processes it through FeatureEngineer, and prints the maximum date
    in the resulting feature DataFrame.
    """
    print("📡 Cargando y procesando datos para verificar la fecha máxima...")
    try:
        df_tweets = load_unified_data()
        feat_eng = FeatureEngineer()
        all_features = feat_eng.process_data(df_tweets)
        
        if not all_features.empty:
            max_date = all_features.index.max()
            print(f"\n✅ La fecha más reciente en los datos procesados es: {max_date.date()}")
        else:
            print("❌ El DataFrame de características procesadas está vacío.")
            
    except Exception as e:
        print(f"\n❌ Ocurrió un error al verificar la fecha máxima de los datos: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    check_max_data_date()

