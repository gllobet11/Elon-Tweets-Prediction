import pandas as pd
import numpy as np
from loguru import logger

from prophet.utilities import regressor_coefficients

def extract_prophet_coefficients(m, regressor_names=None):
    """
    Extrae los coeficientes reales (Beta) usando la utilidad nativa de Prophet.
    Esto maneja automáticamente la des-estandarización y es compatible con
    todas las versiones recientes.
    """
    try:
        # 1. Usamos la función oficial de Prophet para obtener coeficientes
        # Esto devuelve un DataFrame con columnas: 'regressor', 'regressor_mode', 'center', 'coef'
        df_coeffs = regressor_coefficients(m)
        
        # 2. Renombramos para que coincida con lo que espera tu script model_analysis.py
        # 'coef' es el valor real (des-estandarizado), equivalente a 'Value'
        df_coeffs = df_coeffs.rename(columns={"regressor": "Regressor", "coef": "Value"})
        
        # 3. Filtrar solo los regresores que nos interesan (si se pasa la lista)
        if regressor_names:
            # Aseguramos que solo devolvemos los que existen en el modelo para evitar errores
            df_coeffs = df_coeffs[df_coeffs["Regressor"].isin(regressor_names)]
        
        # 4. Devolver solo las columnas necesarias
        return df_coeffs[["Regressor", "Value"]]

    except Exception as e:
        print(f"Error extracting coefficients: {e}")
        return pd.DataFrame(columns=["Regressor", "Value"])