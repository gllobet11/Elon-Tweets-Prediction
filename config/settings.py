# settings.py

"""
Configuration file for storing global variables and project parameters.
"""

# ==========================================
# 1. MARKET CONFIGURATION
# ==========================================

# Keywords to identify the correct market on Polymarket.
# It's crucial that these keywords are specific enough to find a single market.
# Including the date range is a good practice.
MARKET_KEYWORDS = ["elon musk", "tweets", "december 5", "december 12"]

# ==========================================
# 2. BACKTESTING AND OPTIMIZATION CONFIGURATION
# ==========================================

# Number of weeks to use in backtesting (cross-validation) and financial optimization.
# This value determines how many historical periods will be simulated.
WEEKS_TO_VALIDATE = 12

# Alpha candidates for the Negative Binomial distribution.
# This is a key hyperparameter for modeling the variance of the tweet count.
ALPHA_CANDIDATES = [0.01]
