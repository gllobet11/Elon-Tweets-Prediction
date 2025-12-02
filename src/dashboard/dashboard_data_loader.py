"""
dashboard_data_loader.py

Este módulo se encarga de cargar todos los datos y artefactos necesarios para el dashboard
de Streamlit. Centraliza la lógica para obtener datos históricos de tweets, cargar modelos
predictivos entrenados y parámetros de riesgo optimizados, así como para obtener información
actual del mercado desde Polymarket.

Proporciona una interfaz limpia para que el script principal del dashboard acceda a todos
los insumos de datos requeridos sin manejar los detalles de la implementación.
"""

import pandas as pd
from datetime import datetime
import os
import glob
import pickle
import re
from loguru import logger

# --- Project-specific Imports ---
try:
    from config.bins_definition import MARKET_BINS
    from config.settings import MARKET_KEYWORDS # NEW IMPORT
    from src.ingestion.unified_feed import load_unified_data
    from src.ingestion.poly_feed import PolymarketFeed
except ImportError as e:
    logger.error(f"Error importing modules in dashboard_data_loader: {e}")
    # Depending on context, might re-raise or handle differently


class DashboardDataLoader:
    """
    Clase para cargar todos los datos y artefactos requeridos por el dashboard.
    """
    def __init__(self):
        """
        Inicializa el cargador de datos.
        """
        self.market_keywords = MARKET_KEYWORDS
        self.poly_feed = PolymarketFeed()

    def load_and_prepare_tweets_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Carga los datos unificados de tweets y los prepara en un formato granular
        y series diarias de conteos.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Una tupla que contiene:
                - granular_df (pd.DataFrame): DataFrame de tweets con granularidad original.
                - daily_series (pd.Series): Serie de conteos diarios de tweets.
        """
        logger.info("Ejecutando carga y preparación de datos de tweets...")
        df_tweets = load_unified_data()
        granular_df = df_tweets.copy()
        
        daily_counts = (
            granular_df.groupby(granular_df['created_at'].dt.floor('D'))
            .size()
            .rename('n_tweets')
            .to_frame()
        )
        full_idx = pd.date_range(start=daily_counts.index.min(), end=daily_counts.index.max(), freq='D')
        daily_series = daily_counts.reindex(full_idx, fill_value=0)
        daily_series.index.name = 'date'
        
        return granular_df, daily_series

    def load_prophet_model(self) -> dict:
        """
        Busca el último modelo Prophet pre-entrenado guardado (.pkl) y lo carga.

        Raises:
            FileNotFoundError: Si no se encuentra ningún archivo de modelo Prophet.

        Returns:
            dict: Un diccionario que contiene el modelo Prophet y sus metadatos.
        """
        model_files = glob.glob('best_prophet_model_*.pkl')
        if not model_files:
            raise FileNotFoundError("No se encontró ningún archivo de modelo Prophet (.pkl). Ejecuta `models_evals.py`.")
        latest_model_path = max(model_files, key=os.path.getmtime)
        with open(latest_model_path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info(f"Modelo '{model_data.get('model_name', 'Unknown')}' cargado desde '{os.path.basename(latest_model_path)}'.")
        return model_data

    def load_risk_parameters(self) -> dict:
        """
        Carga los parámetros de riesgo óptimos (alpha y kelly_fraction) desde 'risk_params.pkl'.

        Si el archivo no se encuentra, devuelve valores por defecto y emite una advertencia.

        Returns:
            dict: Un diccionario con 'alpha' y 'kelly' (óptimos o por defecto).
        """
        try:
            with open('risk_params.pkl', 'rb') as f:
                risk_params = pickle.load(f)
            logger.info(f"Parámetros de riesgo cargados. Alpha: {risk_params['alpha']:.4f}, Kelly: {risk_params['kelly']:.2f}.")
            return risk_params
        except FileNotFoundError:
            logger.warning("`risk_params.pkl` no encontrado. Usando valores por defecto (alpha=0.2, kelly=0.1). Ejecuta `financial_optimizer.py`.")
            return {'alpha': 0.2, 'kelly': 0.1}

    def fetch_market_data(self) -> dict:
        """
        Obtiene los detalles y el estado actual del mercado de Polymarket basado en las palabras clave configuradas.

        Extrae las fechas de inicio y fin del mercado de la descripción del mismo y
        obtiene los precios de los bins disponibles.

        Raises:
            ValueError: Si no se encuentran detalles del mercado o no se pueden extraer las fechas.

        Returns:
            dict: Un diccionario con información detallada del mercado, incluyendo
                  pregunta, fechas, snapshot de precios y configuración de bins.
        """
        market_details = self.poly_feed.get_market_details(keywords=self.market_keywords)
        if not market_details:
            raise ValueError(f"No se encontraron detalles del mercado para las keywords: {self.market_keywords}")
            
        description = market_details.get('description', '')
        match = re.search(r'from (.*? ET) to (.*? ET)', description)
        if not match:
            raise ValueError("No se pudieron extraer las fechas del mercado de la descripción.")
            
        start_date_str, end_date_str = match.groups()
        start_clean = start_date_str.replace(" ET", "").strip()
        end_clean = end_date_str.replace(" ET", "").strip()
        
        # This assumes current year if not specified in the description.
        # This might need to be more robust for year changes.
        if str(datetime.now().year) not in start_clean:
            start_clean = f"{start_clean}, {datetime.now().year}"
            
        market_start_date = pd.to_datetime(start_clean).tz_localize('America/New_York').tz_convert('UTC')
        market_end_date = pd.to_datetime(end_clean).tz_localize('America/New_York').tz_convert('UTC')
        
        updated_bins = self.poly_feed.fetch_market_ids_automatically(keywords=self.market_keywords, bins_dict=MARKET_BINS)
        market_snapshot = self.poly_feed.get_all_bins_prices(updated_bins)

        return {
            'market_details': market_details,
            'market_question': market_details.get('question', 'Título No Encontrado'),
            'market_start_date': market_start_date,
            'market_end_date': market_end_date,
            'market_snapshot': market_snapshot,
            'updated_bins': updated_bins,
            'bins_config': [(k, v['lower'], v['upper']) for k, v in MARKET_BINS.items()]
        }