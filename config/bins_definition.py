# ==========================================
# 1. CONFIGURACIÓN DE ESTRATEGIA
# ==========================================
MIN_EDGE_TO_TRADE = 0.20  # 20% de diferencia (alpha) para entrar
MAX_POSITION_SIZE = 100  # USDC por bet

# ==========================================
# 2. GENERADOR DE BINS (140 hasta 500+)
# ==========================================
# Definimos el rango base
START_BIN = 140
END_BIN = 500
STEP = 20

# Diccionario donde almacenaremos la configuración
MARKET_BINS = {}

# A) Generación automática de bins intermedios (140-159, 160-179... 480-499)
for lower_bound in range(START_BIN, END_BIN, STEP):
    upper_bound = lower_bound + STEP

    # Etiqueta estilo Polymarket: "140-159"
    # Ahora formateamos los números con comas para que coincidan con la API
    label = f"{lower_bound:,}-{upper_bound - 1:,}"

    MARKET_BINS[label] = {
        "id": None,  # <--- AQUÍ SE RELLENARÁ AUTOMÁTICAMENTE O A MANO
        "lower": lower_bound,
        "upper": upper_bound,  # Usamos el límite exclusivo para cálculo de CDF
    }

# B) Agregar el Bin Final (500+)
# Formateado con comas también para consistencia
MARKET_BINS["500+"] = {
    "id": None,
    "lower": 500,
    "upper": 10000,  # Un número suficientemente grande para simular infinito en la CDF
}

# ==========================================
# 3. MAPEO DE IDs (MANUAL O API)
# ==========================================
# Si prefieres rellenar a mano los IDs de esta semana, hazlo aquí sobreescribiendo.
# Si usas el script de auto-fetch, esto se ignorará.

# ==========================================
# 4. ORDEN CANÓNICO DE BINS
# ==========================================
# Generamos una lista ordenada de los bins para asegurar consistencia en todo el proyecto.
# Se ordena numéricamente por el límite inferior de cada bin.
BINS_ORDER = sorted(MARKET_BINS.keys(), key=lambda k: MARKET_BINS[k]["lower"])
