import pandas as pd
import os
import glob
from datetime import datetime, timezone
from loguru import logger
import sys

# Ensure project root is in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Conditional import based on module location
try:
    # If run_ingest.py is in the root
    from src.ingestion.xtracker_feed import XTrackerIngestor, snowflake_to_dt_utc
    from src.ingestion.auto_ingestor import AutoIngestor
except ImportError:
    # If run_ingest.py is in src/ingestion
    from ingestion.xtracker_feed import XTrackerIngestor, snowflake_to_dt_utc
    from ingestion.auto_ingestor import AutoIngestor


# --- Path Configuration ---
project_root = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(project_root, 'data', 'raw')
processed_dir = os.path.join(project_root, 'data', 'processed')
merged_output_path = os.path.join(processed_dir, 'merged_elon_tweets.csv')

# Ensure processed directory exists
os.makedirs(processed_dir, exist_ok=True)

# Define MARKET_KEYWORDS for AutoIngestor
# These keywords should match the market you want to analyze
MARKET_KEYWORDS = ["elon musk", "tweets", "november 25", "december 2"] # Example, should be dynamic or configured


def run_data_ingestion():
    logger.info("Starting data ingestion process.")

    # --- PASO 0: Auto-Ingesta del CSV más reciente ---
    logger.info("Executing Auto-Ingestor to download the latest CSV...")
    try:
        auto_ingestor = AutoIngestor(keywords=MARKET_KEYWORDS, download_path=raw_dir)
        auto_ingestor.run()
        logger.success("Auto-ingestion completed successfully.")
    except Exception as e:
        logger.warning(f"Auto-ingestion process failed. Continuing with existing local files. Error: {e}")

    # --- PASO 1: Unificación de Datos ---
    hist_path = os.path.join(raw_dir, 'elonmusk.csv')
    search_pattern = os.path.join(raw_dir, '*Elon_Musk*tweets*.csv')

    logger.info("Starting unification of data sources...")

    # Cargar Datos Históricos
    if not os.path.exists(hist_path):
        logger.error(f"Historical file not found: {hist_path}")
        return pd.DataFrame()
    
    try:
        df_hist = pd.read_csv(hist_path, sep=",", engine="python", on_bad_lines="skip", quotechar='"', escapechar="\\")
        df_hist.columns = [c.strip().lower() for c in df_hist.columns]
        
        df_hist['id'] = df_hist['id'].astype(str).str.replace(r'\D', '', regex=True)
        df_hist = df_hist[df_hist['id'].str.len() > 0].copy()
        df_hist['id'] = df_hist['id'].astype('int64')
        df_hist['created_at'] = df_hist['id'].apply(snowflake_to_dt_utc)
        
        df_hist = df_hist[['id', 'text', 'created_at']]
        logger.info(f"Loaded {len(df_hist)} historical tweets.")
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

    # Cargar Datos de la Semana Actual (el que se acaba de descargar)
    df_curr = pd.DataFrame()
    list_of_files = glob.glob(search_pattern)
    
    list_of_files.sort(key=os.path.getmtime, reverse=True)

    if list_of_files:
        current_week_file = list_of_files[0]
        logger.info(f"Current week file found: {os.path.basename(current_week_file)}")
        
        ingestor = XTrackerIngestor(data_path=current_week_file)
        df_curr = ingestor.load_and_clean_data()
        
        if not df_curr.empty:
            logger.info(f"Loaded {len(df_curr)} new tweets.")
            logger.info(f"(New data range: {df_curr['created_at'].min()} to {df_curr['created_at'].max()})")
    else:
        logger.warning(f"No new files found in: {raw_dir}")

    # Unificar y Deduplicar
    df_unified = pd.concat([df_hist, df_curr], ignore_index=True)
    df_unified.drop_duplicates(subset=['id'], keep='last', inplace=True)
    df_unified.sort_values('created_at', inplace=True)
    df_unified.reset_index(drop=True, inplace=True)

    # Filtrar por fecha
    filter_date = pd.Timestamp('2025-01-01', tz=timezone.utc)
    df_unified = df_unified[df_unified['created_at'] >= filter_date].copy()
    
    logger.info(f"TOTAL UNIFIED (filtered from {filter_date.date()}): {len(df_unified)} unique tweets.")
    logger.info(f"Total Range (filtered): {df_unified['created_at'].min().date()} to {df_unified['created_at'].max().date()}")

    # --- PASO 2: Guardar datos unificados ---
    df_unified.to_csv(merged_output_path, index=False)
    logger.success(f"Unified data saved to {merged_output_path}")

    return df_unified

if __name__ == "__main__":
    run_data_ingestion()
