import pandas as pd
import os
from loguru import logger

def load_unified_data():
    """
    Lee los datos de tweets pre-procesados y unificados desde el archivo CSV.
    Este archivo es generado por el script `run_ingest.py`.

    Retorna:
        pd.DataFrame: Un DataFrame GRANULAR (una fila por tweet) ordenado por fecha,
                      o un DataFrame vacío si el archivo no se encuentra.
    """
    # --- Construcción de rutas ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'merged_elon_tweets.csv')

    logger.info(f"Attempting to load pre-processed data from: {processed_data_path}")

    if not os.path.exists(processed_data_path):
        logger.error(f"Pre-processed data file not found at '{processed_data_path}'.")
        logger.error("Please run the `run_ingest.py` script first to generate the unified data file.")
        return pd.DataFrame()

    try:
        # Cargar los datos sin parsear fechas inicialmente
        df = pd.read_csv(processed_data_path)

        # Convertir a datetime de forma robusta, forzando errores a NaT (Not a Time) y estableciendo UTC
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)

        # Eliminar filas donde la fecha no se pudo convertir (si las hay)
        rows_before = len(df)
        df.dropna(subset=['created_at'], inplace=True)
        rows_after = len(df)
        
        if rows_before > rows_after:
            logger.warning(f"Se eliminaron {rows_before - rows_after} filas por tener fechas inválidas.")

        if df.empty:
            logger.error("El DataFrame está vacío después de procesar las fechas.")
            return pd.DataFrame()

        logger.success(f"Successfully loaded and processed {len(df)} tweets from pre-processed file.")
        logger.info(f"Data range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
        
        return df

    except Exception as e:
        logger.error(f"Failed to load or process data from '{processed_data_path}'. Error: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Para pruebas rápidas
    df_test = load_unified_data()
    if not df_test.empty:
        print("Test load successful.")
        print(df_test.head())
        print(df_test.info())