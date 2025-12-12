import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta

from src.processing.feature_eng import FeatureEngineer

import os
import shutil  # Added for clearing debug directory


def get_hybrid_prediction(
    prophet_model, all_features_df: pd.DataFrame, days_forward: int = 7
) -> tuple[pd.DataFrame, dict]:
    """
    Genera predicciones recursivas (Walk-Forward) asegurando que los lags
    se calculen correctamente para cada día futuro.
    """
    if all_features_df.empty:
        return pd.DataFrame(), {}

    # Inicializamos el ingeniero
    feature_engineer = FeatureEngineer()

    # Trabajamos con una copia del histórico para ir añadiendo el futuro simulado
    # Solo necesitamos las columnas base, el engineer recalculará el resto
    history_expanded = all_features_df.copy()

    # Asegurarnos de que el índice es datetime
    if not isinstance(history_expanded.index, pd.DatetimeIndex):
        history_expanded.index = pd.to_datetime(history_expanded.index)

    predictions = []

    for i in range(days_forward):
        # 1. Determinar la fecha siguiente
        last_date = history_expanded.index[-1]
        next_date = last_date + timedelta(days=1)

        # 2. Crear fila placeholder para el siguiente día
        # Inicializamos con 0 o NaN. Lo importante es que exista el índice.
        # Las features externas (Tesla/SpaceX) se llenarán al llamar a process_data
        # (si hay datos futuros disponibles en los CSVs o APIs)
        new_row = pd.DataFrame(index=[next_date])

        # Concatenamos temporalmente para calcular features
        temp_df = pd.concat([history_expanded, new_row])

        # 3. Recalcular Features (Aquí ocurre la magia de los Lags correctos)
        # process_data tomará el 'n_tweets' de ayer y lo pondrá en 'lag_1' de hoy
        processed_df = feature_engineer.process_data(temp_df)

        # 4. Extraer la fila lista para predecir (la última)
        row_to_predict = processed_df.iloc[[-1]].copy()

        # Preparar para Prophet (ds format)
        row_for_prophet = row_to_predict.reset_index().rename(columns={"index": "ds"})
        if row_for_prophet["ds"].dt.tz is not None:
            row_for_prophet["ds"] = row_for_prophet["ds"].dt.tz_localize(None)

        # 5. Predecir
        forecast = prophet_model.predict(row_for_prophet)
        yhat = forecast["yhat"].values[0]

        # Redondear y asegurar no negativo (tweets son enteros)
        yhat_clean = max(0, int(round(yhat)))

        # 6. Guardar predicción y actualizar histórico "ficticio"
        predictions.append({"ds": next_date, "y_pred": yhat_clean})

        # Escribimos el valor predicho en nuestro histórico expandido
        # para que sirva de base para el lag del siguiente día del bucle
        # (Si no hacemos esto, el siguiente día tendrá un lag de 0)

        # Añadimos la fila al history_expanded con el valor predicho
        # Nota: Solo necesitamos 'n_tweets' para que el feature engineer funcione en la prox vuelta
        new_row_with_val = pd.DataFrame({"n_tweets": [yhat_clean]}, index=[next_date])
        # Aseguramos compatibilidad de columnas si hay más en history
        for col in history_expanded.columns:
            if col not in new_row_with_val.columns:
                new_row_with_val[col] = 0  # O valores por defecto

        history_expanded = pd.concat([history_expanded, new_row_with_val])

    # --- Resultados Finales ---
    predictions_df = pd.DataFrame(predictions)

    total_tweets = predictions_df["y_pred"].sum()
    metrics = {
        "weekly_total_prediction": total_tweets,
        "sum_of_predictions": total_tweets,
        "sum_of_actuals": 0,  # En modo futuro puro es 0
        "remaining_days_fraction": days_forward,
    }

    return predictions_df, metrics
