import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from loguru import logger
import numpy as np

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.strategy.prob_math import DistributionConverter
    from config.bins_definition import MARKET_BINS
    from tools.models_evals import ALPHA_CANDIDATES # Assuming optimal alpha is first in this list
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
HISTORICAL_PERFORMANCE_PATH = os.path.join(project_root, 'data', 'processed', 'historical_performance.csv')
OUTPUT_PLOT_PATH = os.path.join(project_root, 'historical_predictions_plot.png')

# Assuming optimal alpha is the first one in ALPHA_CANDIDATES as per tuning
OPTIMAL_ALPHA = ALPHA_CANDIDATES[0]
OPTIMAL_DISTRIBUTION = 'nbinom' # From tuning results
BINS_CONFIG_LIST = [(k, v['lower'], v['upper']) for k, v in MARKET_BINS.items()]

def calculate_log_loss_for_row(row: pd.Series) -> float:
    """Calculates Log Loss for a single row of predictions."""
    mu_pred = row['y_pred']
    y_true = row['y_true']

    try:
        probs = DistributionConverter.get_bin_probabilities(
            mu_remainder=mu_pred,
            current_actuals=0, # Assuming no current actuals for this historical data
            model_type=OPTIMAL_DISTRIBUTION,
            alpha=OPTIMAL_ALPHA,
            bins_config=BINS_CONFIG_LIST
        )
    except ValueError:
        return np.nan # Return NaN if probability calculation fails

    # Determine the correct bin for y_true
    correct_bin = None
    for label, lower, upper in BINS_CONFIG_LIST:
        if lower <= y_true < upper:
            correct_bin = label
            break
    
    if correct_bin is None:
        return np.nan # y_true outside defined bins

    prob_correct = (probs.get(correct_bin, 0) + 1e-9) # Add epsilon to prevent log(0)
    return -np.log(prob_correct)


def visualize_predictions_and_metrics():
    logger.info("📊 Starting visualization of historical predictions...")

    if not os.path.exists(HISTORICAL_PERFORMANCE_PATH):
        logger.error(f"Historical performance data not found at '{HISTORICAL_PERFORMANCE_PATH}'.")
        logger.error("Please run `tools/generate_historical_performance.py` first.")
        sys.exit(1)

    df_hist = pd.read_csv(HISTORICAL_PERFORMANCE_PATH)
    df_hist['week_start_date'] = pd.to_datetime(df_hist['week_start_date'])
    df_hist = df_hist.sort_values('week_start_date').set_index('week_start_date')

    # Calculate Log Loss for each week
    df_hist['log_loss'] = df_hist.apply(calculate_log_loss_for_row, axis=1)
    avg_log_loss = df_hist['log_loss'].mean()
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((df_hist['y_true'] - df_hist['y_pred'])**2))

    logger.info(f"Visualizing predictions for {len(df_hist)} weeks.")
    logger.info(f"Average Log Loss: {avg_log_loss:.4f}")
    logger.info(f"RMSE: {rmse:.2f}")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_hist, x=df_hist.index, y='y_true', label='Actual Tweets (y_true)', marker='o', color='blue')
    sns.lineplot(data=df_hist, x=df_hist.index, y='y_pred', label='Predicted Tweets (y_pred)', marker='x', linestyle='--', color='orange')
    
    # Add Confidence Bands
    plt.fill_between(df_hist.index, df_hist['y_pred_lower'], df_hist['y_pred_upper'], color='orange', alpha=0.2, label='Confidence Interval')

    plt.title('Historical Tweet Predictions vs. Actuals with Confidence Interval')
    plt.xlabel('Week Start Date')
    plt.ylabel('Number of Tweets')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT_PATH)
    logger.success(f"Plot saved to: {OUTPUT_PLOT_PATH}")
    # plt.show() # Uncomment to display plot interactively

if __name__ == "__main__":
    visualize_predictions_and_metrics()