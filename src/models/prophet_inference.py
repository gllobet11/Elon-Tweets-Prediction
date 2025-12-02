import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import os
import sys
import contextlib # Add this import

# Suppress cmdstanpy and prophet warnings for cleaner output
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Path Configuration (similar to other scripts) ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.processing.feature_eng import FeatureEngineer
except Exception as e:
    print(f"--- ERROR FATAL EN LA CONFIGURACIÓN INICIAL DE PROPHET ---")
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
        self.burst_threshold = None # To be calculated during training

    def _calculate_dynamic_features(self, y_history: pd.Series, last_known_date: pd.Timestamp, full_df: pd.DataFrame = None):
        """
        Calcula características dinámicas (lag_1, last_burst, roll_sum_7, momentum, reply_ratio, hour_std_feature)
        basándose en el historial de 'y' (n_tweets) y valores conocidos de 'full_df' para features "externas".
        
        CRÍTICO: Para features que no son puramente autoregresivas (como reply_ratio, hour_std_feature),
        si la fecha a predecir es FUTURA respecto a last_known_date, se usa el último valor conocido (Forward Fill).
        """
        recent_history = y_history.iloc[-30:] # Usar un historial reciente para rolling stats
        last_val = recent_history.iloc[-1]
        
        feats = {}
        
        # 1. Características puramente autoregresivas (calculadas solo desde y_history)
        feats['lag_1'] = last_val

        # last_burst: necesita un umbral que se calcula en el entrenamiento
        if self.burst_threshold is not None:
            feats['last_burst'] = 1 if last_val > self.burst_threshold else 0
        else:
            feats['last_burst'] = 0 # Fallback si no se ha entrenado (debería calcularse)
        
        # Rolling Sum 7
        feats['roll_sum_7'] = recent_history.rolling(7).sum().iloc[-1] if len(recent_history) >=7 else recent_history.sum()
        
        # Momentum (diferencia entre media 3 y 7 días)
        r3 = recent_history.rolling(3).mean().iloc[-1] if len(recent_history) >=3 else np.nan
        r7 = recent_history.rolling(7).mean().iloc[-1] if len(recent_history) >=7 else np.nan
        feats['momentum'] = (r3 - r7) if (not pd.isna(r3) and not pd.isna(r7)) else 0.0

        # 2. Características "externas" (reply_ratio, hour_std_feature)
        # Estas deben ser "conocidas" para el futuro o usar forward fill
        # 'full_df' es el DataFrame de features procesadas hasta 'last_known_date'
        
        # Placeholder for dynamic lookup - this is simplified for now
        # In a full production system, you'd need to compute these for future dates
        # or have a robust way to forward-fill them.
        # For simplicity here, if the feature is not autoregressive, we assume it's 0 or the last known value
        
        # AQUI ES DONDE HAY QUE ASEGURARSE DE QUE LAS COLUMNAS existen en full_df si se van a usar
        # La lógica de Prophet requiere que los futuros valores de los regresores se pasen explícitamente.
        # Para la predicción recursiva, si 'full_df' solo contiene datos históricos,
        # para fechas futuras 'ds > last_known_date', no tendremos valores reales de 'reply_ratio' o 'hour_std_feature'.
        # La sugerencia era usar el último valor conocido (forward fill).
        
        # Obtener el último valor conocido para cada feature no autoregresiva
        last_known_values = {}
        if full_df is not None:
            # Asegurar que el 'full_df' está ordenado por fecha y tiene las columnas
            full_df_sorted = full_df.sort_index()
            for col in ['reply_ratio', 'hour_std_feature']: # Add other non-autoregressive features here
                if col in full_df_sorted.columns:
                    # El valor que se "arrastra" es el del 'last_known_date'
                    val = full_df_sorted.loc[full_df_sorted.index <= last_known_date, col].iloc[-1] if not full_df_sorted.loc[full_df_sorted.index <= last_known_date, col].empty else 0
                    last_known_values[col] = val
                else:
                    last_known_values[col] = 0

        # Asignar a feats, usando forward fill para fechas futuras
        for col in ['reply_ratio', 'hour_std_feature']:
            if col in self.regressors: # Solo si esta feature está en la lista de regresores del modelo
                feats[col] = last_known_values.get(col, 0)
        
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
        if 'ds' not in df_train.columns or 'y' not in df_train.columns:
            raise ValueError("df_train must contain 'ds' (datetime) and 'y' (target) columns.")
        
        # Ensure 'ds' is timezone-naive
        if df_train['ds'].dt.tz is not None:
            df_train['ds'] = df_train['ds'].dt.tz_localize(None)

        # Initialize Prophet model
        self.model = Prophet(
            growth='linear',
            yearly_seasonality=False, # Assuming weekly is more relevant for tweets
            weekly_seasonality=True,
            daily_seasonality=False, # Already handled by day-level data
            changepoint_prior_scale=0.05, # Default value, can be tuned
            seasonality_mode='multiplicative' # Can be additive or multiplicative
        )

        # Add regressors
        for regressor in self.regressors:
            if regressor not in df_train.columns:
                if verbose:
                    print(f"Advertencia: Regresor '{regressor}' no encontrado en df_train. Se omitirá.")
            else:
                self.model.add_regressor(regressor)

        # Use contextlib to suppress output from Prophet/cmdstanpy's fit method
        with contextlib.redirect_stdout(os.devnull), contextlib.redirect_stderr(os.devnull):
            self.model.fit(df_train)
        if verbose:
            print("✅ Modelo Prophet entrenado.")

        # Calculate burst_threshold from the training data for future dynamic feature calculation
        self.burst_threshold = df_train['y'].quantile(0.8) if not df_train['y'].empty else 0


    def predict_single_step(self, target_date: pd.Timestamp, all_processed_features_df: pd.DataFrame):
        """
        Predice 'y' para una única fecha objetivo (target_date) utilizando el modelo Prophet entrenado.
        Las características para target_date se extraen de all_processed_features_df.
        Este método es para la validación walk-forward, donde las features del día a predecir son conocidas.
        :param target_date: Fecha para la que se quiere predecir.
        :param all_processed_features_df: DataFrame con todas las features procesadas, incluyendo la fila para target_date.
        :return: Predicción de 'y' para target_date.
        """
        if self.model is None:
            raise ValueError("El modelo Prophet no ha sido entrenado. Llame a .train() primero.")
        
        if not target_date in all_processed_features_df.index:
            raise ValueError(f"target_date {target_date} no encontrada en all_processed_features_df.")

        # 1. Extraer las características para target_date
        features_for_target_date = all_processed_features_df.loc[[target_date]]
        
        # 2. Construir el DataFrame 'future' para Prophet
        future_df = pd.DataFrame({'ds': [target_date]})
        
        # 3. Rellenar future_df con los regresores necesarios
        for regressor in self.regressors:
            if regressor in features_for_target_date.columns:
                future_df[regressor] = features_for_target_date[regressor].values[0]
            else:
                # Si un regresor es dinámico (ej. lag_1), debería ser calculado por FeatureEngineer y estar en all_processed_features_df
                # Si no está, podría indicar un problema en el feature engineering o una feature no válida
                print(f"Advertencia: Regresor '{regressor}' no encontrado para target_date {target_date}. Se usará 0.")
                future_df[regressor] = 0

        # Asegurar que 'ds' es timezone-naive
        if future_df['ds'].dt.tz is not None:
            future_df['ds'] = future_df['ds'].dt.tz_localize(None)

        # 4. Predecir
        forecast = self.model.predict(future_df)
        pred_y = max(0, forecast['yhat'].values[0]) # Asegurar no negativos
        return pred_y


    def predict_next_week_recursively(self, historical_df: pd.DataFrame, days_to_predict: int = 7, full_features_df: pd.DataFrame = None):
        """
        Realiza una predicción recursiva para los próximos 'days_to_predict' días.
        :param historical_df: DataFrame con el historial (ds, y) para iniciar la recursión.
                              También debe contener los regresores para las últimas fechas.
        :param days_to_predict: Número de días a predecir.
        :param full_features_df: DataFrame completo de features procesadas, usado para forward-fill de features "externas".
        :return: Suma total de tweets predichos para el período.
        """
        if self.model is None:
            raise ValueError("El modelo Prophet no ha sido entrenado. Llame a .train() primero.")
        
        if historical_df.empty:
            raise ValueError("historical_df no puede estar vacío para predicción recursiva.")
        
        # Asegurar que el 'ds' es timezone-naive
        if historical_df['ds'].dt.tz is not None:
            historical_df['ds'] = historical_df['ds'].dt.tz_localize(None)

        dynamic_history_y = historical_df[['ds', 'y']].set_index('ds')['y']
        
        last_date_in_data = dynamic_history_y.index.max()
        
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
        
        print(f"📅 Prediciendo para el periodo: {future_dates.min().date()} al {future_dates.max().date()}")
        
        future_preds = []
        
        for current_date in future_dates:
            # 1. Preparar la fila futura para la predicción
            future_row_data = {'ds': [current_date]}
            
            # 2. Calcular features dinámicas (autoregresivas) y externas (forward-filled)
            # El 'full_features_df' es el `all_features` que ya tenemos procesado
            computed_feats = self._calculate_dynamic_features(
                y_history=dynamic_history_y, 
                last_known_date=last_date_in_data, # last_date_in_data se mantiene fijo como el último día REAL conocido
                full_df=full_features_df
            )
            
            # 3. Rellenar future_row_data con las features calculadas
            for reg in self.regressors:
                future_row_data[reg] = computed_feats.get(reg, 0)
                
            future_row = pd.DataFrame(future_row_data)

            # 4. Predecir con Prophet
            forecast = self.model.predict(future_row)
            pred_y = max(0, forecast['yhat'].values[0]) # Asegurar no negativos

            future_preds.append(pred_y)
            
            # 5. Actualizar historial dinámico para la próxima iteración recursiva
            dynamic_history_y = pd.concat([dynamic_history_y, pd.Series([pred_y], index=[current_date])])
            
            # Update last_date_in_data for _calculate_dynamic_features (recursive logic)
            # This is important: for the next day's prediction, the 'last_known_date' for external features
            # should still be the last REAL date, not the predicted one.
            # Only y_history updates recursively.
            
            # print(f"   -> {current_date.date()}: {pred_y:.2f} tweets") # Descomentar para verbosidad

        total_predicted = sum(future_preds)
        return total_predicted
