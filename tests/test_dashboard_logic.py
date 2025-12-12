from unittest.mock import patch, MagicMock, Mock
import pytest
from unittest import mock
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil import tz 
import numpy as np

logger = mock.Mock() 

# --- Zona Horaria ---
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except ImportError:
    try:
        ET_TZ = tz.gettz("America/New_York")
        if ET_TZ is None:
            ET_TZ = timezone(timedelta(hours=-5)) 
    except ImportError:
        ET_TZ = timezone(timedelta(hours=-5)) 

class DashboardLogicProcessor:
    def __init__(self):
        pass

    def _get_current_time_et(self):
        """Wrapper para mockear fácilmente en tests."""
        return datetime.now(ET_TZ)

    def _to_timestamp(self, dt):
        """Convierte datetime a pd.Timestamp preservando timezone."""
        return pd.Timestamp(dt) if dt.tzinfo else pd.Timestamp(dt, tz=ET_TZ)

    def _get_hybrid_prediction(self, prophet_model, all_features_df: pd.DataFrame, days_forward: int = 7) -> tuple[pd.DataFrame, dict]:
        return pd.DataFrame(), {"sum_of_predictions": 0.0}

    def calculate_trading_opportunities(
        self,
        prophet_model,
        optimal_alpha: float,
        optimal_kelly: float,
        market_info: dict,
        granular_data: pd.DataFrame,
        all_features_df: pd.DataFrame,  
        bankroll: float,
        historical_performance_df: pd.DataFrame,
        selected_strategy: str,
        simple_mode: str = "BLOCK",
        simple_bet_pct: float = 0.10,
    ) -> tuple[pd.DataFrame, dict]:
        
        market_start_date = market_info.get("market_start_date")
        
        today = self._get_current_time_et()
        
        # ✅ FIX: Asegurar TZ correctamente
        if market_start_date.tzinfo is None:
            market_start_date = market_start_date.replace(tzinfo=ET_TZ) 
        
        # Preparar granular_data
        if not isinstance(granular_data.index, pd.DatetimeIndex):
            if "created_at" in granular_data.columns:
                granular_data = granular_data.set_index("created_at")
        
        # ✅ FIX CRÍTICO: Datos YA SON UTC → tz_convert, NO tz_localize
        if granular_data.index.tz is None:
            # Asumir UTC si naive (estándar en APIs)
            granular_data.index = granular_data.index.tz_localize('UTC')
        elif str(granular_data.index.tz) != str(ET_TZ):
            # Convertir a ET si está en otra TZ
            granular_data.index = granular_data.index.tz_convert(ET_TZ)

        today_ts = self._to_timestamp(today)
        market_start_ts = self._to_timestamp(market_start_date)
        
        actuals_df = granular_data[
            (granular_data.index >= market_start_ts) & (granular_data.index < today_ts)
        ]
        sum_of_actuals = len(actuals_df)
        
        _, pred_metrics = self._get_hybrid_prediction(prophet_model, all_features_df, 0)
        
        metrics = {
            "sum_of_actuals": sum_of_actuals,
            "sum_of_predictions": pred_metrics["sum_of_predictions"],
            "weekly_total_prediction": sum_of_actuals,
            "remaining_days_fraction": 0.0
        }
        return pd.DataFrame(), metrics

# --- Fixtures ---
MARKET_START = datetime(2025, 12, 5, 12, 0, 0)

@pytest.fixture
def base_data_frame():
    """Datos en UTC equivalentes a ET times."""
    data = [
        # T1: 11:59:59 AM ET = 16:59:59 UTC → EXCLUIDO
        (datetime(2025, 12, 5, 16, 59, 59), 1), 
        # T2: 12:00:00 PM ET = 17:00:00 UTC → INCLUIDO si today > 12:00ET
        (datetime(2025, 12, 5, 17, 0, 0), 1),  
        # T3: 12:00:01 PM ET = 17:00:01 UTC → INCLUIDO si today > 12:00:01ET
        (datetime(2025, 12, 5, 17, 0, 1), 1),  
    ]
    df = pd.DataFrame(data, columns=["created_at", "n_tweets"])
    return df.set_index("created_at")["n_tweets"].to_frame()

@pytest.fixture
def market_info_fixture():
    return {
        "market_start_date": MARKET_START,
        "market_end_date": datetime(2025, 12, 12, 12, 0, 0),
        "bins_config": [],
        "market_snapshot": {}
    }

@pytest.fixture
def processor():
    return DashboardLogicProcessor()

@pytest.fixture
def test_args(market_info_fixture):
    return {
        'prophet_model': Mock(), 
        'optimal_alpha': 0.1, 
        'optimal_kelly': 0.5,
        'market_info': market_info_fixture, 
        'granular_data': None,  # Inyectado por test
        'all_features_df': pd.DataFrame(), 
        'bankroll': 1000.0,
        'historical_performance_df': pd.DataFrame(), 
        'selected_strategy': "",
    }

# --- Tests ---
def test_calculate_trading_opportunities_timezone_boundary(processor, base_data_frame, test_args):
    test_args['granular_data'] = base_data_frame.copy()
    
    # Debug: Verificar conversión
    print("Datos originales (UTC):")
    print(base_data_frame.index)
    
    # CASO 1: today = 12:00:00 ET → 0 actuals (ninguno < 12:00ET)
    with patch.object(processor, '_get_current_time_et') as mock_time:
        mock_time.return_value = datetime(2025, 12, 5, 12, 0, 0, tzinfo=ET_TZ)
        _, metrics = processor.calculate_trading_opportunities(**test_args)
        print(f"Caso 1 today={mock_time.return_value}, actuals={metrics['sum_of_actuals']}")
        assert metrics["sum_of_actuals"] == 0

    # CASO 2: today = 12:00:00.000001 ET → 1 actual (T2=12:00:00ET incluido)
    with patch.object(processor, '_get_current_time_et') as mock_time:
        mock_time.return_value = datetime(2025, 12, 5, 12, 0, 0, 1, tzinfo=ET_TZ)
        _, metrics = processor.calculate_trading_opportunities(**test_args)
        print(f"Caso 2 today={mock_time.return_value}, actuals={metrics['sum_of_actuals']}")
        assert metrics["sum_of_actuals"] == 1

    # CASO 3: today = 12:00:01 ET → 1 actual (T2 incluido, T3 excluido)
    with patch.object(processor, '_get_current_time_et') as mock_time:
        mock_time.return_value = datetime(2025, 12, 5, 12, 0, 1, tzinfo=ET_TZ)
        _, metrics = processor.calculate_trading_opportunities(**test_args)
        print(f"Caso 3 today={mock_time.return_value}, actuals={metrics['sum_of_actuals']}")
        assert metrics["sum_of_actuals"] == 1

    # CASO 4: today = 12:00:01.000001 ET → 2 actuals (T2+T3)
    with patch.object(processor, '_get_current_time_et') as mock_time:
        mock_time.return_value = datetime(2025, 12, 5, 12, 0, 1, 1, tzinfo=ET_TZ)
        _, metrics = processor.calculate_trading_opportunities(**test_args)
        print(f"Caso 4 today={mock_time.return_value}, actuals={metrics['sum_of_actuals']}")
        assert metrics["sum_of_actuals"] == 2
