import pandas as pd
from loguru import logger
import os
import contextlib

# --- Importaciones de la estructura del proyecto ---
from src.ingestion.unified_feed import load_unified_data

def get_hybrid_prediction(prophet_model, market_start_date, market_end_date, df_tweets):
    """
    Calcula una predicción semanal combinando los tweets reales de los días ya pasados
    con una predicción de Prophet para el tiempo restante.
    Esta versión es precisa con los tiempos, no redondea a días completos.

    Args:
        prophet_model: El objeto del modelo Prophet ya entrenado y cargado.
        market_start_date (pd.Timestamp): Fecha de inicio del mercado (UTC).
        market_end_date (pd.Timestamp): Fecha de fin del mercado (UTC).
        df_tweets (pd.DataFrame): DataFrame granular con todos los tweets.

    Returns:
        tuple: (predicción_total, reales_sumados, predicción_sumada, días_predichos_fraccionales)
    """
    logger.info("🛠️  Iniciando lógica de predicción híbrida precisa...")

    now_utc = pd.Timestamp.now(tz='UTC')

    # 2. Contar tweets reales con precisión
    actuals_mask = (df_tweets['created_at'] >= market_start_date) & (df_tweets['created_at'] < now_utc)
    sum_of_actuals = len(df_tweets[actuals_mask])
    
    logger.info(f" -> Suma de tweets reales (exactos): {sum_of_actuals}")

    # 3. Calcular el tiempo restante a predecir
    sum_of_predictions = 0
    remaining_time = market_end_date - now_utc
    remaining_days_fraction = remaining_time.total_seconds() / (24 * 3600)

    if remaining_days_fraction > 0:
        logger.info(f"🔮 Quedan {remaining_days_fraction:.2f} días fraccionales por predecir.")
        
        # --- FIX: Añadir regresores al future_df ---
        # 1. Identificar los regresores que el modelo necesita.
        # El objeto del modelo tiene un atributo con la lista de regresores con los que fue entrenado.
        model_regressors = list(prophet_model.extra_regressors.keys())
        logger.info(f"El modelo requiere los siguientes regresores: {model_regressors}")

        # 2. Generar todas las features para obtener los valores más recientes de los regresores.
        # Usamos df_tweets que es el DF granular con todos los datos.
        from src.processing.feature_eng import FeatureEngineer
        feature_engineer = FeatureEngineer()
        # Silenciamos la salida de `process_data` para no ensuciar el log de producción
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            all_features_df = feature_engineer.process_data(df_tweets.copy())
        
        # 3. Obtener la última fila de datos con features.
        latest_features = all_features_df.iloc[-1]
        
        # 4. Crear el future_df y añadirle los regresores.
        future_df = prophet_model.make_future_dataframe(periods=7, freq='D')

        if model_regressors:
            for regressor in model_regressors:
                if regressor in latest_features:
                    # Usamos el último valor conocido del regresor para toda la ventana de predicción.
                    regressor_value = latest_features[regressor]
                    future_df[regressor] = regressor_value
                    logger.info(f"   -> Añadiendo regresor '{regressor}' con valor constante: {regressor_value:.4f}")
                else:
                    # Fallback por si un regresor no se pudo calcular (aunque no debería pasar).
                    future_df[regressor] = 0.0
                    logger.warning(f"   -> No se encontró el regresor '{regressor}' en las features. Usando 0 como fallback.")
        # --- FIN DEL FIX ---

        forecast = prophet_model.predict(future_df)
        
        # Tomamos la predicción de los próximos 7 días y calculamos la media
        # 'yhat' es la predicción puntual de Prophet
        avg_daily_prediction_rate = forecast.iloc[-7:]['yhat'].clip(lower=0).mean()
        logger.info(f" -> Tasa de predicción diaria del modelo: {avg_daily_prediction_rate:.2f} tweets/día")

        # 5. Estimar los tweets para el período restante
        sum_of_predictions = avg_daily_prediction_rate * remaining_days_fraction
        logger.info(f" -> Suma de tweets predichos (fraccional): {sum_of_predictions:.2f}")
    else:
        logger.info("✅ No queda tiempo por predecir, usando solo datos reales.")
        remaining_days_fraction = 0

    # 6. Combinar para obtener la predicción total
    total_prediction = sum_of_actuals + sum_of_predictions
    
    return total_prediction, sum_of_actuals, sum_of_predictions, remaining_days_fraction