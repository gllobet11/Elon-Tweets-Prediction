import glob
import os
import pickle

import pandas as pd
from loguru import logger

# --- Project-specific Imports ---
try:
    from config.bins_definition import MARKET_BINS
    from config.settings import MARKET_KEYWORDS
    from src.ingestion.poly_feed import PolymarketFeed
    from src.ingestion.unified_feed import load_unified_data
except ImportError as e:
    logger.error(f"Error importing modules in dashboard_data_loader: {e}")


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

        if df_tweets.empty:
            logger.error(
                "El DataFrame está VACÍO después de load_unified_data(). No se puede continuar.",
            )
            # In a real app, you might want to stop or show an error message.
            # For now, we'll return empty structures to prevent a crash downstream,
            # though this will likely lead to other errors.
            return pd.DataFrame(), pd.Series(dtype="int64")

        # Fuerza la conversión a UTC y maneja errores para asegurar que el groupby funcione
        df_tweets["created_at"] = pd.to_datetime(
            df_tweets["created_at"], utc=True, errors="coerce",
        )
        # -------------------------------------

        granular_df = df_tweets.copy()

        # Elimina filas con fecha inválida (NaT) que rompen el groupby
        granular_df = granular_df.dropna(subset=["created_at"])

        daily_counts = (
            granular_df.groupby(granular_df["created_at"].dt.floor("D"))
            .size()
            .rename("n_tweets")
            .to_frame()
        )
        full_idx = pd.date_range(
            start=daily_counts.index.min(), end=daily_counts.index.max(), freq="D",
        )
        daily_series = daily_counts.reindex(full_idx, fill_value=0)
        daily_series.index.name = "date"

        logger.info(
            f"Datos cargados: {len(granular_df)} tweets, {len(daily_series)} días",
        )

        return granular_df, daily_series

    # ... resto de los métodos sin cambios ...

    def load_prophet_model(self) -> dict:
        """
        Busca el último modelo Prophet pre-entrenado guardado (.pkl) y lo carga.
        """
        model_files = glob.glob("best_prophet_model_*.pkl")
        if not model_files:
            raise FileNotFoundError(
                "No se encontró ningún archivo de modelo Prophet (.pkl). Ejecuta `models_evals.py`.",
            )
        latest_model_path = max(model_files, key=os.path.getmtime)
        with open(latest_model_path, "rb") as f:
            model_data = pickle.load(f)
        logger.info(
            f"Modelo '{model_data.get('model_name', 'Unknown')}' cargado desde '{os.path.basename(latest_model_path)}'.",
        )
        return model_data

    def load_risk_parameters(self) -> dict:
        """
        Carga los parámetros de riesgo óptimos (alpha y kelly_fraction) desde 'risk_params.pkl'.
        """
        try:
            with open("risk_params.pkl", "rb") as f:
                risk_params = pickle.load(f)
            logger.info(
                f"Parámetros de riesgo cargados. Alpha: {risk_params['alpha']:.4f}, Kelly: {risk_params['kelly']:.2f}.",
            )
            return risk_params
        except FileNotFoundError:
            logger.warning(
                "`risk_params.pkl` no encontrado. Usando valores por defecto (alpha=0.2, kelly=0.1). Ejecuta `financial_optimizer.py`.",
            )
            return {"alpha": 0.2, "kelly": 0.1}

    def fetch_market_data(self) -> dict:
        """
        Obtiene los detalles y el estado actual del mercado de Polymarket.
        """
        market_details = self.poly_feed.get_market_details(
            keywords=self.market_keywords,
        )
        if not market_details:
            raise ValueError(
                f"No se encontraron detalles del mercado para las keywords: {self.market_keywords}",
            )

        description = market_details.get("description", "")
        market_start_date, market_end_date = self.poly_feed.get_market_dates(
            description,
        )
        if not market_start_date or not market_end_date:
            raise ValueError("Could not extract dates from the market description.")

        updated_bins = self.poly_feed.fetch_market_ids_automatically(
            keywords=self.market_keywords, bins_dict=MARKET_BINS,
        )
        market_snapshot = self.poly_feed.get_all_bins_prices(updated_bins)

        return {
            "market_details": market_details,
            "market_question": market_details.get("question", "Título No Encontrado"),
            "market_start_date": market_start_date,
            "market_end_date": market_end_date,
            "market_snapshot": market_snapshot,
            "updated_bins": updated_bins,
            "bins_config": [
                (k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()
            ],
        }

    def load_historical_performance(self) -> pd.DataFrame:
        """
        Loads historical performance data (predictions vs actuals) from a CSV file.
        """
        history_path = os.path.join("data", "processed", "historical_performance.csv")
        if not os.path.exists(history_path):
            logger.warning(
                f"Historical performance file not found at {history_path}. Run `tools/generate_historical_performance.py`.",
            )
            return pd.DataFrame()

        df_history = pd.read_csv(history_path)
        df_history["week_start_date"] = pd.to_datetime(df_history["week_start_date"])
        df_history = df_history.set_index("week_start_date").sort_index()
        df_history.index = df_history.index.tz_localize("UTC")
        logger.info(f"Loaded {len(df_history)} historical performance weeks.")
        return df_history
