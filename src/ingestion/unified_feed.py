import os

import pandas as pd
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
    processed_data_path = os.path.join(
        project_root, "data", "processed", "merged_elon_tweets.csv",
    )

    logger.info(f"Attempting to load pre-processed data from: {processed_data_path}")

    if not os.path.exists(processed_data_path):
        logger.error(f"Pre-processed data file not found at '{processed_data_path}'.")
        logger.error(
            "Please run the `run_ingest.py` script first to generate the unified data file.",
        )
        return pd.DataFrame()

    try:
        # 1. Cargar datos
        # IMPORTANTE: Forzamos 'id' a string para evitar problemas de precisión o notación científica
        # low_memory=False ayuda si el archivo crece mucho
        df = pd.read_csv(processed_data_path, dtype={"id": str}, low_memory=False)

        # 2. Conversión de fechas ROBUSTA (El arreglo del bug)
        # format='mixed' permite que convivan fechas estilo "2025-11-27 20:00:00" (Histórico)
        # con fechas estilo "2025-12-03T15:00:00" (API nueva) sin generar NaT.
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True, format="mixed",
        )

        # 3. Limpieza
        rows_before = len(df)
        df.dropna(subset=["created_at"], inplace=True)
        rows_after = len(df)

        if rows_before > rows_after:
            # Si esto ocurre ahora, realmente son datos corruptos, no un error de formato
            logger.warning(
                f"Se eliminaron {rows_before - rows_after} filas por tener fechas inválidas.",
            )

        if df.empty:
            logger.error("El DataFrame está vacío después de procesar las fechas.")
            return pd.DataFrame()

        # 4. Asegurar ordenamiento (buena práctica para series temporales)
        df = df.sort_values("created_at").reset_index(drop=True)

        logger.success(
            f"Successfully loaded and processed {len(df)} tweets from pre-processed file.",
        )
        logger.info(
            f"Data range: {df['created_at'].min().date()} to {df['created_at'].max().date()}",
        )

        return df

    except Exception as e:
        logger.error(
            f"Failed to load or process data from '{processed_data_path}'. Error: {e}",
        )
        return pd.DataFrame()


if __name__ == "__main__":
    # Para pruebas rápidas
    df_test = load_unified_data()
    if not df_test.empty:
        print("Test load successful.")
        print(
            df_test.tail(),
        )  # Imprimimos el final para ver si llegan los datos de Diciembre
        print(df_test.info())
