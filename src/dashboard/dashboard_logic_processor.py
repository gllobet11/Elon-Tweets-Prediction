"""
dashboard_logic_processor.py

Este módulo encapsula la lógica de negocio central del dashboard de Streamlit.
Se encarga de procesar los datos cargados para calcular métricas clave (KPIs),
realizar las predicciones híbridas del modelo, y determinar las oportunidades
de trading basadas en probabilidades y el criterio de Kelly.

Su objetivo es mantener la lógica de cálculo separada de la capa de presentación (UI)
para facilitar el mantenimiento, las pruebas y la reutilización del código.
"""

from datetime import datetime

import pandas as pd
from loguru import logger
from pandas.io.formats.style import Styler

# --- Project-specific Imports ---
try:
    from config.bins_definition import MARKET_BINS
    from src.strategy.hybrid_predictor import get_hybrid_prediction
    from src.strategy.prob_math import DistributionConverter
    from src.strategy.financial_simple import get_simple_strategy_recommendations, calculate_simple_bias # ADDED
    from src.utils.prophet_utils import extract_prophet_coefficients
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

    def calculate_kpis(self, daily_data: pd.Series) -> dict:
        """
        Calcula una serie de Indicadores Clave de Rendimiento (KPIs).
        """
        six_months_ago = daily_data.index.max() - pd.DateOffset(months=6)
        data_last_6_months = daily_data[daily_data.index >= six_months_ago].copy()

        current_month_str = datetime.now().strftime("%B")
        month_data = daily_data[daily_data.index.month == datetime.now().month]
        monthly_mean = month_data["n_tweets"].mean()

        last_7d = daily_data.tail(7).copy()
        mean_7d = last_7d["n_tweets"].mean()
        std_7d = last_7d["n_tweets"].std()
        threshold = mean_7d + std_7d
        outlier_days = last_7d[last_7d["n_tweets"] > threshold].shape[0]
        yesterday_val = daily_data["n_tweets"].iloc[-2]
        deviation = yesterday_val - monthly_mean

        return {
            "monthly_mean": monthly_mean,
            "yesterday_val": yesterday_val,
            "deviation": deviation,
            "outlier_days": outlier_days,
            "std_7d": std_7d,
            "current_month_str": current_month_str,
            "data_last_6_months": data_last_6_months,
        }

    def calculate_trading_opportunities(
        self,
        prophet_model,
        optimal_alpha: float, # Used by Optimal strategy
        optimal_kelly: float, # Used by Optimal strategy
        market_info: dict,
        granular_data: pd.DataFrame,
        all_features_df: pd.DataFrame,
        bankroll: float,
        historical_performance_df: pd.DataFrame, # NEW: for simple strategy bias
        selected_strategy: str, # NEW: to select strategy
    ) -> tuple[pd.DataFrame, dict]:
        """
        Calculates trading opportunities.
        """
        # Extract necessary information from market_info
        bins_config = market_info["bins_config"]
        market_snapshot = market_info["market_snapshot"]
        market_start_date = market_info["market_start_date"]
        market_end_date = market_info["market_end_date"]

        # --- RE-INTEGRATED HYBRID LOGIC (Common to both strategies) ---
        # 1. Calculate actuals for the elapsed portion of the week
        today = datetime.now(market_start_date.tz)
        
        if not isinstance(granular_data.index, pd.DatetimeIndex):
             granular_data = granular_data.set_index('created_at')

        actuals_df = granular_data[(granular_data.index >= market_start_date) & (granular_data.index < today)]
        sum_of_actuals = len(actuals_df)

        # 2. Get a pure forecast for the remaining days
        remaining_days = max(0, (market_end_date - today).days)
        
        _predictions_df, pred_metrics = get_hybrid_prediction(
            prophet_model=prophet_model,
            all_features_df=all_features_df,
            days_forward=remaining_days
        )
        sum_of_predictions = pred_metrics.get("sum_of_predictions", 0.0)

        # 3. Combine for the final hybrid prediction
        weekly_total_prediction = sum_of_actuals + sum_of_predictions
        
        # --- Strategy Selection ---
        if selected_strategy == "Optimal (Financial Optimizer)":
            logger.info("Using Optimal (Financial Optimizer) strategy.")
            # This is the existing logic
            model_probabilities = DistributionConverter.get_bin_probabilities(
                mu_remainder=sum_of_predictions,
                current_actuals=sum_of_actuals,
                alpha=optimal_alpha,
                bins_config=bins_config,
            )

            opportunities = []
            for bin_label, _, _ in bins_config: # Iterate over ordered bins_config
                valuation = market_snapshot.get(bin_label, {"mid_price": 0.0})
                mkt_price = valuation.get("mid_price", 0.0)
                my_prob = model_probabilities.get(bin_label, 0.0)
                edge = my_prob - mkt_price

                bet_size = DistributionConverter.calculate_kelly_bet(
                    my_prob=my_prob,
                    market_price=mkt_price,
                    bankroll=bankroll,
                    kelly_fraction=optimal_kelly,
                )

                opportunities.append(
                    {
                        "Bin": bin_label,
                        "Mkt Price": mkt_price,
                        "My Model": my_prob,
                        "Edge": edge,
                        "Bet Size ($)": bet_size,
                    },
                )
            df_opportunities = pd.DataFrame(opportunities)

        elif selected_strategy == "Simple Directional (Financial Simple)":
            logger.info("Using Simple Directional (Financial Simple) strategy.")
            # Calculate bias from historical data
            historical_bias = calculate_simple_bias(historical_performance_df)
            
            y_pred_biased = weekly_total_prediction + historical_bias

            # Determine the target bin for the y_pred_biased
            target_bin_idx = -1
            # Iterate through bins_config which is (label, lower, upper)
            for i, (bin_label, bin_lower, bin_upper) in enumerate(bins_config):
                if bin_lower <= y_pred_biased < bin_upper:
                    target_bin_idx = i
                    break
            
            # If y_pred_biased falls outside defined bins, default to first/last
            if target_bin_idx == -1:
                if y_pred_biased < bins_config[0][1]: # Using bin_lower from first bin (bins_config[0] is (label, lower, upper))
                    target_bin_idx = 0
                else:
                    target_bin_idx = len(bins_config) - 1
            
            # Generate a simple probability distribution centered around the target bin
            model_probabilities = {}
            # Initialize all bins to a very small non-zero probability to avoid division by zero or log errors,
            # especially if the chart tries to do something with zero probability.
            # Then assign higher probabilities to target/neighbors.
            default_low_prob = 1e-6 / len(bins_config) # Smallest possible positive prob
            for i, (bin_label, _, _) in enumerate(bins_config):
                model_probabilities[bin_label] = default_low_prob

            # Assign higher probabilities for target and neighbors
            if target_bin_idx != -1: # Ensure a target bin was found
                model_probabilities[bins_config[target_bin_idx][0]] = 0.8 # High probability for the target bin
                
                # Neighbors
                if target_bin_idx > 0:
                    model_probabilities[bins_config[target_bin_idx - 1][0]] = 0.1
                if target_bin_idx < len(bins_config) - 1:
                    model_probabilities[bins_config[target_bin_idx + 1][0]] = 0.1

            # Normalize probabilities to sum to 1.0
            prob_sum = sum(model_probabilities.values())
            if prob_sum > 0: # Should always be > 0 due to default_low_prob
                model_probabilities = {k: v / prob_sum for k, v in model_probabilities.items()}
            else: # Fallback if for some reason sum is 0 (should not happen with default_low_prob)
                model_probabilities = {k: 1.0 / len(bins_config) for k, v in model_probabilities.items()} # Uniform distribution

            # Now, calculate trading opportunities using this simple probability distribution
            opportunities = []
            
            # First, get the trade recommendations from the simple strategy for actual bet sizes.
            current_trade_recs = get_simple_strategy_recommendations(
                y_pred_current=weekly_total_prediction, # Use the hybrid prediction as input
                current_capital=bankroll,
                mode="BLOCK", # Default mode for simple strategy in dashboard
                bet_pct=0.10, # Default bet percentage
                historical_bias=historical_bias,
                market_prices_override=market_snapshot_to_array(market_snapshot, bins_config), # Pass live market prices
            )
            
            for bin_label, bin_lower, bin_upper in bins_config: # Iterate using bins_config
                valuation = market_snapshot.get(bin_label, {"mid_price": 0.0})
                mkt_price = valuation.get("mid_price", 0.0)
                my_prob = model_probabilities.get(bin_label, 0.0) # Use the derived probability
                edge = my_prob - mkt_price

                bet_size_usd = 0.0
                for rec in current_trade_recs:
                    if rec["bin_label"] == bin_label:
                        bet_size_usd = rec["wager_usd"]
                        break

                opportunities.append(
                    {
                        "Bin": bin_label,
                        "Mkt Price": mkt_price,
                        "My Model": my_prob,
                        "Edge": edge,
                        "Bet Size ($)": bet_size_usd,
                    },
                )
            df_opportunities = pd.DataFrame(opportunities)
            
        else:
            logger.error(f"Unknown strategy selected: {selected_strategy}")
            df_opportunities = pd.DataFrame() # Return empty if strategy is unknown

        # --- END OF STRATEGY SELECTION ---
        
        # This part remains common regardless of strategy chosen
        remaining_days_fraction = remaining_days / 7.0
        
        return df_opportunities, {
            "weekly_total_prediction": weekly_total_prediction,
            "sum_of_actuals": sum_of_actuals,
            "sum_of_predictions": sum_of_predictions,
            "remaining_days_fraction": remaining_days_fraction,
        }

    def get_feature_importance(self, model_data: dict) -> pd.DataFrame:
        """
        Extracts feature importance (coefficients) from the Prophet model using a centralized utility.
        """
        model = model_data.get("model")
        regressors = model_data.get("regressors", [])

        if not model or not regressors:
            return pd.DataFrame()

        try:
            # Use the new utility function
            feature_importance_df = extract_prophet_coefficients(model, regressors)
            
            if not feature_importance_df.empty:
                # The utility already sorts by Abs_Coefficient, but let's add the Impact column for coloring
                feature_importance_df['Impact'] = feature_importance_df['Coefficient'].apply(
                    lambda x: 'Positive' if x > 0 else 'Negative'
                )
                # Rename Abs_Coefficient to Magnitude for consistency with charts if needed
                if 'Abs_Coefficient' in feature_importance_df.columns:
                    feature_importance_df.rename(columns={'Abs_Coefficient': 'Magnitude'}, inplace=True)

            return feature_importance_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return pd.DataFrame()

    def style_opportunities_df(self, df: pd.DataFrame) -> Styler:
        """
        Aplica un formato estilístico a un DataFrame de oportunidades de trading.
        """
        return (
            df.style.format(
                {
                    "Mkt Price": "{:.3f}",
                    "My Model": "{:.3f}",
                    "Edge": "{:+.3f}",
                    "Bet Size ($)": "${:,.2f}",
                },
            )
            .background_gradient(cmap="Blues", subset=["Mkt Price"])
            .background_gradient(cmap="Greens", subset=["My Model"])
            .background_gradient(cmap="RdYlGn", subset=["Edge"], vmin=-0.1, vmax=0.1)
            .bar(subset=["Bet Size ($)"], color="#2E8B57", align="zero")
        )

# Helper function for market prices (needed by get_simple_strategy_recommendations)
import numpy as np # Necesario para np.ndarray
def market_snapshot_to_array(market_snapshot: dict, bins_config: list) -> np.ndarray:
    """Converts the market snapshot dict to an array matching the BINS order."""
    prices_array = np.zeros(len(bins_config))
    # bins_config here is a list of (label, lower, upper)
    for i, (bin_label, _, _) in enumerate(bins_config):
        prices_array[i] = market_snapshot.get(bin_label, {}).get("mid_price", 0.0)
    return prices_array
