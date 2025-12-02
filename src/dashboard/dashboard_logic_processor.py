"""
dashboard_logic_processor.py

Este módulo encapsula la lógica de negocio central del dashboard de Streamlit.
Se encarga de procesar los datos cargados para calcular métricas clave (KPIs),
realizar las predicciones híbridas del modelo, y determinar las oportunidades
de trading basadas en probabilidades y el criterio de Kelly.

Su objetivo es mantener la lógica de cálculo separada de la capa de presentación (UI)
para facilitar el mantenimiento, las pruebas y la reutilización del código.
"""

import pandas as pd
from loguru import logger
from datetime import datetime
from pandas.io.formats.style import Styler

# --- Project-specific Imports ---
try:
    from config.bins_definition import MARKET_BINS
    from src.strategy.prob_math import DistributionConverter
    from src.strategy.hybrid_predictor import get_hybrid_prediction
except ImportError as e:
    logger.error(f"Error importing modules in dashboard_logic_processor: {e}")

class DashboardLogicProcessor:
    """
    Clase para procesar la lógica de negocio del dashboard.
    """
    def __init__(self):
        """
        Inicializa el procesador de lógica.
        """
        pass

    def calculate_kpis(self, daily_data: pd.Series) -> dict:
        """
        Calcula una serie de Indicadores Clave de Rendimiento (KPIs).
        """
        six_months_ago = daily_data.index.max() - pd.DateOffset(months=6)
        data_last_6_months = daily_data[daily_data.index >= six_months_ago].copy()
        
        current_month_str = datetime.now().strftime('%B')
        month_data = daily_data[daily_data.index.month == datetime.now().month]
        monthly_mean = month_data['n_tweets'].mean()
        
        last_7d = daily_data.tail(7).copy()
        mean_7d = last_7d['n_tweets'].mean()
        std_7d = last_7d['n_tweets'].std()
        threshold = mean_7d + std_7d
        outlier_days = last_7d[last_7d['n_tweets'] > threshold].shape[0]
        yesterday_val = daily_data['n_tweets'].iloc[-2]
        deviation = yesterday_val - monthly_mean

        return {
            'monthly_mean': monthly_mean,
            'yesterday_val': yesterday_val,
            'deviation': deviation,
            'outlier_days': outlier_days,
            'std_7d': std_7d,
            'current_month_str': current_month_str,
            'data_last_6_months': data_last_6_months
        }

    def calculate_trading_opportunities(self, prophet_model, optimal_alpha: float, optimal_kelly: float,
                                        market_info: dict, granular_data: pd.DataFrame, bankroll: float) -> tuple[pd.DataFrame, dict]:
        """
        Calcula las oportunidades de trading.
        """
        market_start_date = market_info['market_start_date']
        market_end_date = market_info['market_end_date']
        bins_config = market_info['bins_config']
        market_snapshot = market_info['market_snapshot']

        (weekly_total_prediction,
         sum_of_actuals,
         sum_of_predictions,
         remaining_days_fraction) = get_hybrid_prediction(prophet_model, market_start_date, market_end_date, granular_data)
        
        model_probabilities = DistributionConverter.get_bin_probabilities(
            mu_remainder=sum_of_predictions,
            current_actuals=sum_of_actuals,
            alpha=optimal_alpha,
            bins_config=bins_config
        )

        opportunities = []
        for bin_label, bin_data in MARKET_BINS.items():
            valuation = market_snapshot.get(bin_label, {'mid_price': 0.0})
            mkt_price = valuation.get('mid_price', 0.0)
            my_prob = model_probabilities.get(bin_label, 0.0)
            edge = my_prob - mkt_price
            
            bet_size = DistributionConverter.calculate_kelly_bet(
                my_prob=my_prob,
                market_price=mkt_price,
                bankroll=bankroll,
                kelly_fraction=optimal_kelly
            )

            opportunities.append({
                "Bin": bin_label, 
                "Mkt Price": mkt_price, 
                "My Model": my_prob, 
                "Edge": edge, 
                "Bet Size ($)": bet_size
            })
        
        df_opportunities = pd.DataFrame(opportunities)

        return df_opportunities, {
            'weekly_total_prediction': weekly_total_prediction,
            'sum_of_actuals': sum_of_actuals,
            'sum_of_predictions': sum_of_predictions,
            'remaining_days_fraction': remaining_days_fraction
        }

    def style_opportunities_df(self, df: pd.DataFrame) -> Styler:
        """
        Aplica un formato estilístico a un DataFrame de oportunidades de trading.
        """
        return df.style.format({
            'Mkt Price': '{:.3f}',
            'My Model': '{:.3f}',
            'Edge': '{:+.3f}',
            'Bet Size ($)': '${:,.2f}'
        }).background_gradient(cmap='Blues', subset=['Mkt Price']
        ).background_gradient(cmap='Greens', subset=['My Model']
        ).background_gradient(cmap='RdYlGn', subset=['Edge'], vmin=-0.1, vmax=0.1
        ).bar(subset=['Bet Size ($)'], color='#2E8B57', align='zero')
