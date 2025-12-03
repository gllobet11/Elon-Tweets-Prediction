# settings.py

"""
Archivo de configuración para almacenar variables globales y parámetros del proyecto.
"""

# ==========================================
# 1. CONFIGURACIÓN DEL MERCADO
# ==========================================

# Palabras clave para identificar el mercado correcto en Polymarket.
# Es crucial que estas palabras clave sean lo suficientemente específicas para
# encontrar un único mercado. Incluir el rango de fechas es una buena práctica.
MARKET_KEYWORDS = ["elon musk", "tweets", "december 2", "december 9"]

# ==========================================
# 2. CONFIGURACIÓN DE BACKTESTING Y OPTIMIZACIÓN
# ==========================================

# Número de semanas para usar en el backtesting (validación cruzada) y la optimización financiera.
# Este valor determina cuántos períodos históricos se simularán.
WEEKS_TO_VALIDATE = 12
