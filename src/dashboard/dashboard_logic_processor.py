"""
dashboard_logic_processor.py
Encapsula la lógica de negocio, predicciones híbridas y cálculo de oportunidades.
Fix: manejo robusto de timezones (ET) y ds naive para Prophet.
Fix: Conditional Status column handling in style_opportunities_df.
"""
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from loguru import logger
from pandas.io.formats.style import Styler

try:
    from config.bins_definition import MARKET_BINS
    from src.strategy.prob_math import DistributionConverter
    from src.strategy.financial_simple import (
        get_simple_strategy_recommendations,
        calculate_simple_bias,
    )
    from src.utils.prophet_utils import extract_prophet_coefficients
    from src.processing.feature_eng import FeatureEngineer
except ImportError as e:
    logger.error(f"Error importing modules in dashboard_logic_processor: {e}")

# --- Definición de Zona Horaria ET (America/New_York) ---
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
    logger.info("Using 'zoneinfo' for America/New_York TZ.")
except ImportError:
    try:
        from dateutil import tz
        ET_TZ = tz.gettz("America/New_York")
        if ET_TZ is None:
            raise ImportError("dateutil could not find America/New_York.")
        logger.info("Using 'dateutil.tz' for America/New_York TZ.")
    except ImportError:
        logger.warning("Could not load full TZ library. Falling back to fixed UTC-5.")
        # Fallback de emergencia, menos robusto por DST
        ET_TZ = timezone(timedelta(hours=-5))


class DashboardLogicProcessor:
    def __init__(self):
        pass

    # ---------- Helpers de tiempo / TZ ----------

    def _get_current_time_et(self) -> datetime:
        """Wrapper para obtener 'now' en ET, fácil de mockear en tests."""
        return datetime.now(ET_TZ)

    def _to_timestamp(self, dt: datetime) -> pd.Timestamp:
        """Convierte datetime a pd.Timestamp preservando timezone."""
        return pd.Timestamp(dt) if dt.tzinfo else pd.Timestamp(dt, tz=ET_TZ)

    # ---------- KPIs ----------

    def calculate_kpis(self, daily_data: pd.Series) -> dict:
        """Calcula métricas de actividad reciente."""
        if daily_data.empty:
            return {
                "monthly_mean": 0,
                "yesterday_val": 0,
                "deviation": 0,
                "outlier_days": 0,
                "std_7d": 0,
                "current_month_str": "",
                "data_last_6_months": pd.DataFrame(),
            }

        six_months_ago = daily_data.index.max() - pd.DateOffset(months=6)
        data_last_6_months = daily_data[daily_data.index >= six_months_ago].copy()

        now_local = datetime.now()
        current_month_str = now_local.strftime("%B")
        month_data = daily_data[daily_data.index.month == now_local.month]
        monthly_mean = month_data["n_tweets"].mean() if not month_data.empty else 0

        last_7d = daily_data.tail(7).copy()
        mean_7d = last_7d["n_tweets"].mean() if not last_7d.empty else 0
        std_7d = last_7d["n_tweets"].std() if not last_7d.empty else 0

        threshold = mean_7d + std_7d
        outlier_days = last_7d[last_7d["n_tweets"] > threshold].shape[0]

        yesterday_val = daily_data["n_tweets"].iloc[-2] if len(daily_data) > 1 else 0
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

    # ---------- Predicción híbrida ----------

    def _get_hybrid_prediction(
        self,
        prophet_model,
        all_features_df: pd.DataFrame,
        days_forward: int = 7,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Genera predicciones recursivas (Walk-Forward).
        Usa la predicción de hoy como input (lag) para mañana.
        """
        if all_features_df.empty:
            return pd.DataFrame(), {}

        feature_engineer = FeatureEngineer()

        history = all_features_df.copy()
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index)

        predictions = []
        current_df = history.copy()

        for i in range(days_forward):
            if current_df.empty:
                break

            last_date = current_df.index[-1]
            next_date = last_date + timedelta(days=1)

            # 1. Predecir el día siguiente
            row_to_predict = current_df.iloc[[-1]].copy()

            # Preparar fila para Prophet (ds + regresores)
            row_for_prophet = (
                row_to_predict.reset_index()
                .rename(columns={"index": "ds", "created_at": "ds"})
            )
            if "ds" not in row_for_prophet.columns:
                row_for_prophet["ds"] = next_date

            # Asegurar timezone naive para Prophet
            # (Prophet requiere ds sin zona horaria)
            if hasattr(row_for_prophet["ds"].dtype, "tz") and row_for_prophet["ds"].dt.tz is not None:
                row_for_prophet["ds"] = (
                    row_for_prophet["ds"].dt.tz_convert(None).dt.tz_localize(None)
                )

            try:
                forecast = prophet_model.predict(row_for_prophet)
                yhat = forecast["yhat"].values[0]
                yhat_clean = max(0, int(round(yhat)))
            except Exception as e:
                logger.error(f"Prediction error at step {i}: {e}")
                yhat_clean = 0

            predictions.append({"ds": next_date, "y_pred": yhat_clean})

            # 2. Actualizar historia para el siguiente paso (Lags)
            new_row = pd.DataFrame({"n_tweets": [yhat_clean]}, index=[next_date])

            current_tz = current_df.index.tz
            if new_row.index.tz is None and current_tz is not None:
                new_row.index = new_row.index.tz_localize(current_tz)

            combined = pd.concat([current_df, new_row])
            current_df = feature_engineer.process_data(combined)

        predictions_df = pd.DataFrame(predictions)
        total_tweets = predictions_df["y_pred"].sum() if not predictions_df.empty else 0.0

        metrics = {
            "weekly_total_prediction": total_tweets,
            "sum_of_predictions": total_tweets,
            "remaining_days_fraction": days_forward,
        }

        return predictions_df, metrics

    def get_live_tweet_audit(self, granular_data: pd.DataFrame, market_info: dict) -> tuple[int, pd.DataFrame]:
        """
        Calculates the current number of tweets strictly within the market window.
        This serves as a sanity check and audit tool.
        """
        m_start = market_info.get("market_start_date")
        m_end = market_info.get("market_end_date")
        
        # Ensure we have granular data with a datetime index
        if granular_data.empty or not isinstance(granular_data.index, pd.DatetimeIndex):
            return 0, pd.DataFrame()

        # Ensure the index is UTC for comparison
        if granular_data.index.tz is None:
            granular_data.index = granular_data.index.tz_localize("UTC")
        elif str(granular_data.index.tz).lower() != "utc":
            granular_data.index = granular_data.index.tz_convert("UTC")

        audit_df = pd.DataFrame()
        
        if m_start:
            # The market dates from the ingestor are already correct, UTC-based
            audit_df = granular_data[granular_data.index >= m_start].copy()
            if m_end:
                audit_df = audit_df[audit_df.index < m_end]
            
            current_audit_count = len(audit_df)
            return current_audit_count, audit_df
            
        return 0, pd.DataFrame()

    # ---------- Oportunidades de trading ----------

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
        bins_config = market_info.get("bins_config", [])
        market_snapshot = market_info.get("market_snapshot", {})
        market_start_date = market_info.get("market_start_date")
        market_end_date = market_info.get("market_end_date")

        if not market_start_date or not market_end_date:
            return pd.DataFrame(), {}

        # 1. Asegurar límites del mercado en ET
        if market_start_date.tzinfo is None:
            market_start_date = market_start_date.replace(tzinfo=ET_TZ)
        if market_end_date.tzinfo is None:
            market_end_date = market_end_date.replace(tzinfo=ET_TZ)

        # 2. 'today' en ET usando wrapper (mockeable en tests)
        today = self._get_current_time_et()

        logger.info(f"📅 Market Window: {market_start_date} to {market_end_date}")
        logger.info(f"🕒 Current Time : {today}")

        # 3. Asegurar que granular_data tiene índice datetime
        if not isinstance(granular_data.index, pd.DatetimeIndex):
            if "created_at" in granular_data.columns:
                granular_data = granular_data.set_index("created_at")

        # 4. Normalizar granular_data a ET
        #   - Si viene naive → asumimos UTC y luego convertimos a ET
        #   - Si viene en UTC aware → tz_convert a ET
        if granular_data.index.tz is None:
            granular_data.index = granular_data.index.tz_localize("UTC")
        if str(granular_data.index.tz) != str(ET_TZ):
            granular_data.index = granular_data.index.tz_convert(ET_TZ)

        # 5. Filtrar actuals en [market_start_date, today)
        today_ts = self._to_timestamp(today)
        market_start_ts = self._to_timestamp(market_start_date)

        actuals_df = granular_data[
            (granular_data.index >= market_start_ts) &
            (granular_data.index < today_ts)
        ]
        sum_of_actuals = len(actuals_df)
        logger.info(f"🧐 Actuals Counted: {sum_of_actuals}")

        # 5. Días restantes hasta fin de mercado
        diff = market_end_date - today_ts.to_pydatetime()
        remaining_days_float = max(0.0, diff.total_seconds() / 86400.0)
        days_to_predict = int(np.ceil(remaining_days_float))

        # 6. Predicción híbrida
        _predictions_df, pred_metrics = self._get_hybrid_prediction(
            prophet_model=prophet_model,
            all_features_df=all_features_df,
            days_forward=days_to_predict,
        )

        sum_of_predictions = pred_metrics.get("sum_of_predictions", 0.0)
        weekly_total_prediction = sum_of_actuals + sum_of_predictions

        # --- Selección de estrategia ---
        df_opportunities = pd.DataFrame()

        if selected_strategy == "Optimal (Financial Optimizer)":
            model_probabilities = DistributionConverter.get_bin_probabilities(
                mu_remainder=sum_of_predictions,
                current_actuals=sum_of_actuals,
                alpha=optimal_alpha,
                bins_config=bins_config,
                model_type="nbinom",
            )

            opportunities = []
            for bin_label, _, _ in bins_config:
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
                        "Status": valuation.get("status", "UNKNOWN"),
                    }
                )
            df_opportunities = pd.DataFrame(opportunities)

        elif selected_strategy == "Simple Directional (Financial Simple)":
            historical_bias = 0.0
            if not historical_performance_df.empty:
                historical_bias = calculate_simple_bias(historical_performance_df)

            prices_array = np.zeros(len(bins_config))
            for i, (l, _, _) in enumerate(bins_config):
                prices_array[i] = market_snapshot.get(l, {}).get("mid_price", 0.0)

            recommendations = get_simple_strategy_recommendations(
                y_pred_current=weekly_total_prediction,
                current_capital=bankroll,
                mode=simple_mode,
                bet_pct=simple_bet_pct,
                historical_bias=historical_bias,
                market_prices_override=prices_array,
            )

            rec_map = {r["bin_label"]: r for r in recommendations}
            opps_data = []

            for bin_label, _, _ in bins_config:
                rec = rec_map.get(bin_label)
                mkt_p = market_snapshot.get(bin_label, {}).get("mid_price", 0.0)

                if rec:
                    opps_data.append(
                        {
                            "Bin": bin_label,
                            "Mkt Price": mkt_p,
                            "My Model": rec.get("model_prob", 0.0),
                            "Edge": rec.get("edge", 0.0),
                            "Bet Size ($)": rec.get("wager_usd", 0.0),
                        }
                    )
                else:
                    opps_data.append(
                        {
                            "Bin": bin_label,
                            "Mkt Price": mkt_p,
                            "My Model": 0.0,
                            "Edge": 0.0,
                            "Bet Size ($)": 0.0,
                        }
                    )
            df_opportunities = pd.DataFrame(opps_data)

        metrics = {
            "weekly_total_prediction": weekly_total_prediction,
            "sum_of_actuals": sum_of_actuals,
            "sum_of_predictions": sum_of_predictions,
            "remaining_days_fraction": remaining_days_float,
        }

        return df_opportunities, metrics

    # ---------- Estilo para Streamlit ----------

    def style_opportunities_df(self, df: pd.DataFrame) -> Styler:
        """
        FIXED: Only apply Status column styling/hiding if the column exists.
        Simple Directional strategy doesn't include Status column.
        """
        if df.empty:
            return df.style

        cols_to_float = ["Mkt Price", "My Model", "Edge", "Bet Size ($)"]
        for c in cols_to_float:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Only apply no_liquidity style if Status column exists (Optimal strategy)
        if "Status" in df.columns:
            def style_no_liquidity(row):
                """Apply a faded style if the market bin has no liquidity."""
                is_no_liq = "NO_LIQUIDITY" in row.get("Status", "") or "MISSING_ID" in row.get("Status", "")
                return ["color: #666; background-color: #f9f9f9" if is_no_liq else "" for _ in row]

            styler = df.style.apply(style_no_liquidity, axis=1)
        else:
            # Simple strategy - no Status column, no special styling
            styler = df.style
        
        styler = styler.format(
            {
                "Mkt Price": "{:.3f}",
                "My Model": "{:.3f}",
                "Edge": "{:+.3f}",
                "Bet Size ($)": "${:,.2f}",
            }
        )
        
        # Hide the status column from the final output ONLY if it exists
        if "Status" in df.columns:
            styler = styler.hide(subset=["Status"], axis=1)

        # Apply gradients and bars only to columns that exist
        if "Mkt Price" in df.columns:
            styler = styler.background_gradient(cmap="Blues", subset=["Mkt Price"])
        if "My Model" in df.columns:
            styler = styler.background_gradient(cmap="Greens", subset=["My Model"])
        if "Edge" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn", subset=["Edge"], vmin=-0.1, vmax=0.1)
        if "Bet Size ($)" in df.columns:
            styler = styler.bar(subset=["Bet Size ($)"], color="#2E8B57", align="zero")
            
        return styler