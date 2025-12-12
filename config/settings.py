# settings.py

"""
Configuration file for storing global variables and project parameters.
"""

# ==========================================
# 1. MARKET CONFIGURATION
# ==========================================

# Keywords para búsqueda de mercados (fallback)
MARKET_KEYWORDS = ["tweets", "december 5", "december 12"]

# NUEVO: Market Slug para búsqueda directa en CLOB API
# Este es el slug base del mercado actual en Polymarket
# Actualiza esto cada semana con el nuevo mercado
# config/settings.py
MARKET_SLUG = "elon-musk-of-tweets-december-5-december-12"
# ==========================================
# 2. BACKTESTING AND OPTIMIZATION CONFIGURATION
# ==========================================

# Number of weeks to use in backtesting (cross-validation) and financial optimization.
# This value determines how many historical periods will be simulated.
WEEKS_TO_VALIDATE = 12

# Alpha candidates for the Negative Binomial distribution.
# This is a key hyperparameter for modeling the variance of the tweet count.
ALPHA_CANDIDATES = [0.03]
