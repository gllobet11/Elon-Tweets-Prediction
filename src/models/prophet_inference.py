import contextlib  # Add this import
import logging
import os
import sys

import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress cmdstanpy and prophet warnings for cleaner output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# --- Path Configuration (similar to other scripts) ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.processing.feature_eng import FeatureEngineer
except Exception as e:
    print("--- ERROR FATAL EN LA CONFIGURACIÓN INICIAL DE PROPHET ---")
    import traceback

    print(f"Error: {e}")
    print(traceback.format_exc())
    sys.exit(1)


class ProphetInferenceModel:
    def __init__(self, regressors: list = None):
        """
        Inicializa el modelo Prophet con una lista de regresores.
        :param regressors: Lista de nombres de características a usar como regresores.
        """
        self.regressors = regressors if regressors is not None else []
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.burst_threshold = None  # To be calculated during training

    def _calculate_dynamic_features(
        self,
        y_history: pd.Series,
        last_known_date: pd.Timestamp,
        current_date: pd.Timestamp,
        full_df: pd.DataFrame = None,
    ):
        """
        Calcula características dinámicas para una fecha futura 'current_date'.

        Para características autoregresivas (ej. lags, rolling stats), se usa 'y_history' que
        puede incluir predicciones pasadas para mantener la recursividad.

        Para características derivadas del tiempo (ej. día de la semana, fin de semana), se calculan directamente.

        Para el resto de características ("externas" no autoregresivas), se usa el último valor conocido
        de 'full_df' (Forward Fill) antes de 'last_known_date'.
        """
        recent_history = y_history.iloc[
            -30:
        ]  # Usar un historial reciente para rolling stats

        # Asegurarse de que recent_history no esté vacío antes de acceder a last_val
        last_val = recent_history.iloc[-1] if not recent_history.empty else 0

        feats = {}

        # Features autoregresivas / de comportamiento
        if "lag_1" in self.regressors:
            feats["lag_1"] = last_val

        if "last_burst" in self.regressors:
            if self.burst_threshold is not None:
                feats["last_burst"] = 1 if last_val > self.burst_threshold else 0
            else:
                feats["last_burst"] = (
                    0  # Fallback si no se ha entrenado (debería calcularse)
                )

        if "roll_sum_7" in self.regressors:
            feats["roll_sum_7"] = (
                recent_history.rolling(7).sum().iloc[-1]
                if len(recent_history) >= 7
                else recent_history.sum()
            )

        if "momentum" in self.regressors:
            r3 = (
                recent_history.rolling(3).mean().iloc[-1]
                if len(recent_history) >= 3
                else np.nan
            )
            r7 = (
                recent_history.rolling(7).mean().iloc[-1]
                if len(recent_history) >= 7
                else np.nan
            )
            feats["momentum"] = (
                (r3 - r7) if (not pd.isna(r3) and not pd.isna(r7)) else 0.0
            )

        # Features derivadas del tiempo
        if "dow" in self.regressors:
            feats["dow"] = current_date.dayofweek

        if "is_weekend" in self.regressors:
            feats["is_weekend"] = 1 if current_date.dayofweek >= 5 else 0

        # Features "externas" / no autoregresivas directas (forward-fill con último valor conocido)
        forward_fill_features = [
            "reply_ratio",
            "hour_std_feature",
            "regime_intensity",
            "is_high_regime",
            "is_low_regime",
            "is_regime_change",
            "cv_7",
        ]

        if full_df is not None:
            full_df_sorted = full_df.sort_index()
            for col in forward_fill_features:
                if col in self.regressors:
                    if col in full_df_sorted.columns:
                        val = (
                            full_df_sorted.loc[
                                full_df_sorted.index <= last_known_date, col,
                            ].iloc[-1]
                            if not full_df_sorted.loc[
                                full_df_sorted.index <= last_known_date, col,
                            ].empty
                            else 0
                        )
                        feats[col] = val
                    else:
                        feats[col] = 0

        # Rellenar NaNs que puedan surgir de rolling/shift en history reciente
        for k, v in feats.items():
            if pd.isna(v):
                feats[k] = 0.0

        return feats

    def train(self, df_train: pd.DataFrame, verbose: bool = True):
        """
        Entrena el modelo Prophet con los datos históricos proporcionados.
        :param df_train: DataFrame con columnas 'ds' (fecha), 'y' (n_tweets) y regresores.
        :param verbose: Si es True, imprime mensajes de progreso.
        """
        if verbose:
            print("🧠 Entrenando modelo Prophet...")

        # Ensure 'ds' and 'y' are present
        if "ds" not in df_train.columns or "y" not in df_train.columns:
            raise ValueError(
                "df_train must contain 'ds' (datetime) and 'y' (target) columns.",
            )

        # Ensure 'ds' is timezone-naive
        if df_train["ds"].dt.tz is not None:
            df_train["ds"] = df_train["ds"].dt.tz_localize(None)

        # Initialize Prophet model
        self.model = Prophet(
            growth="linear",
            yearly_seasonality=False,  # Assuming weekly is more relevant for tweets
            weekly_seasonality=True,
            daily_seasonality=False,  # Already handled by day-level data
            changepoint_prior_scale=0.05,  # Default value, can be tuned
            seasonality_mode="multiplicative",  # Can be additive or multiplicative
        )

        # Add regressors
        for regressor in self.regressors:
            if regressor not in df_train.columns:
                if verbose:
                    print(
                        f"Advertencia: Regresor '{regressor}' no encontrado en df_train. Se omitirá.",
                    )
            else:
                self.model.add_regressor(regressor)

        # Use contextlib to suppress output from Prophet/cmdstanpy's fit method
        with (
            contextlib.redirect_stdout(os.devnull),
            contextlib.redirect_stderr(os.devnull),
        ):
            self.model.fit(df_train)
        if verbose:
            print("✅ Modelo Prophet entrenado.")

        # Calculate burst_threshold from the training data for future dynamic feature calculation
        self.burst_threshold = (
            df_train["y"].quantile(0.8) if not df_train["y"].empty else 0
        )

    def predict_single_step(
        self, target_date: pd.Timestamp, all_processed_features_df: pd.DataFrame,
    ) -> tuple[float, float, float]: # Return yhat, yhat_lower, yhat_upper
        """
        Predice 'y' para una única fecha objetivo (target_date) utilizando el modelo Prophet entrenado.
        Las características para target_date se extraen de all_processed_features_df.
        Este método es para la validación walk-forward, donde las features del día a predecir son conocidas.
        :param target_date: Fecha para la que se quiere predecir.
        :param all_processed_features_df: DataFrame con todas las features procesadas, incluyendo la fila para target_date.
        :return: Predicción de 'y' para target_date, yhat_lower, yhat_upper.
        """
        if self.model is None:
            raise ValueError(
                "El modelo Prophet no ha sido entrenado. Llame a .train() primero.",
            )

        if target_date not in all_processed_features_df.index:
            raise ValueError(
                f"target_date {target_date} no encontrada en all_processed_features_df.",
            )

        # 1. Extraer las características para target_date
        features_for_target_date = all_processed_features_df.loc[[target_date]]

        # 2. Construir el DataFrame 'future' para Prophet
        future_df = pd.DataFrame({"ds": [target_date]})

        # 3. Rellenar future_df con los regresores necesarios
        for regressor in self.regressors:
            if regressor in features_for_target_date.columns:
                future_df[regressor] = features_for_target_date[regressor].values[0]
            else:
                # Si un regresor es dinámico (ej. lag_1), debería ser calculado por FeatureEngineer y estar en all_processed_features_df
                # Si no está, podría indicar un problema en el feature engineering o una feature no válida
                print(
                    f"Advertencia: Regresor '{regressor}' no encontrado para target_date {target_date}. Se usará 0.",
                )
                future_df[regressor] = 0

        # Asegurar que 'ds' es timezone-naive
        if future_df["ds"].dt.tz is not None:
            future_df["ds"] = future_df["ds"].dt.tz_localize(None)

        # 4. Predecir
        forecast = self.model.predict(future_df)
        pred_y = max(0, forecast["yhat"].values[0])  # Asegurar no negativos
        pred_y_lower = forecast["yhat_lower"].values[0]
        pred_y_upper = forecast["yhat_upper"].values[0]
        return pred_y, pred_y_lower, pred_y_upper

    def predict_next_week_recursively(
        self,
        historical_df: pd.DataFrame,
        days_to_predict: int = 7,
        full_features_df: pd.DataFrame = None,
    ):
        """
        Realiza una predicción recursiva para los próximos 'days_to_predict' días.
        :param historical_df: DataFrame con el historial (ds, y) para iniciar la recursión.
                              También debe contener los regresores para las últimas fechas.
        :param days_to_predict: Número de días a predecir.
        :param full_features_df: DataFrame completo de features procesadas, usado para forward-fill de features "externas".
        :return: Suma total de tweets predichos para el período.
        """
        if self.model is None:
            raise ValueError(
                "El modelo Prophet no ha sido entrenado. Llame a .train() primero.",
            )

        if historical_df.empty:
            raise ValueError(
                "historical_df no puede estar vacío para predicción recursiva.",
            )

        # Asegurar que el 'ds' es timezone-naive
        if historical_df["ds"].dt.tz is not None:
            historical_df["ds"] = historical_df["ds"].dt.tz_localize(None)

        dynamic_history_y = historical_df[["ds", "y"]].set_index("ds")["y"]

        last_date_in_data = dynamic_history_y.index.max()

        future_dates = pd.date_range(
            start=last_date_in_data + pd.Timedelta(days=1),
            periods=days_to_predict,
            freq="D",
        )

        print(
            f"📅 Prediciendo para el periodo: {future_dates.min().date()} al {future_dates.max().date()}",
        )

        future_preds = []

        for current_date in future_dates:
            # 1. Preparar la fila futura para la predicción
            future_row_data = {"ds": [current_date]}

            # 2. Calcular features dinámicas (autoregresivas) y externas (forward-filled)
            # El 'full_features_df' es el `all_features` que ya tenemos procesado
            computed_feats = self._calculate_dynamic_features(
                y_history=dynamic_history_y,
                last_known_date=last_date_in_data,  # last_date_in_data se mantiene fijo como el último día REAL conocido
                current_date=current_date,  # Pass current_date for date-derived features
                full_df=full_features_df,
            )

            # 3. Rellenar future_row_data con las features calculadas
            for reg in self.regressors:
                future_row_data[reg] = computed_feats.get(reg, 0)

            future_row = pd.DataFrame(future_row_data)

            # 4. Predecir con Prophet
            forecast = self.model.predict(future_row)
            pred_y = max(0, forecast["yhat"].values[0])  # Asegurar no negativos

            future_preds.append(pred_y)

            # 5. Actualizar historial dinámico para la próxima iteración recursiva
            dynamic_history_y = pd.concat(
                [dynamic_history_y, pd.Series([pred_y], index=[current_date])],
            )

            # Update last_date_in_data for _calculate_dynamic_features (recursive logic)
            # This is important: for the next day's prediction, the 'last_known_date' for external features
            # should still be the last REAL date, not the predicted one.
            # Only y_history updates recursively.

            # print(f"   -> {current_date.date()}: {pred_y:.2f} tweets") # Descomentar para verbosidad

        total_predicted = sum(future_preds)
        return total_predicted
