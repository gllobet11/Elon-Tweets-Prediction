import pandas as pd
from loguru import logger

def analyze_prediction_bias():
    """
    Analyzes the historical performance data to calculate the
    systematic bias (underestimation) of the model's predictions.
    """
    history_path = "data/processed/historical_performance.csv"
    
    try:
        df = pd.read_csv(history_path)
    except FileNotFoundError:
        logger.error(f"File not found: {history_path}. Please run the pipeline to generate it.")
        return

    # Calculate the error for each prediction
    df['error'] = df['y_true'] - df['y_pred']
    
    # Calculate the mean error (the bias)
    mean_bias = df['error'].mean()
    
    logger.info(f"Analysis of prediction bias based on {len(df)} historical weeks:")
    logger.info(f"  - Average 'y_true' (actual tweets): {df['y_true'].mean():.2f}")
    logger.info(f"  - Average 'y_pred' (predicted tweets): {df['y_pred'].mean():.2f}")
    logger.info(f"  - Average Error (y_true - y_pred): {mean_bias:.2f}")
    
    if mean_bias > 0:
        logger.success(f"CONCLUSION: The model systematically underestimates the number of tweets by an average of {mean_bias:.2f} tweets per week.")
        logger.info("This value will be used as the bias correction factor.")
    else:
        logger.warning(f"The model seems to be overestimating. Bias: {mean_bias:.2f}. A different approach may be needed.")
        
    # Save the bias to a file for the optimizer to use
    with open("bias_correction.txt", "w") as f:
        f.write(str(mean_bias))

if __name__ == "__main__":
    analyze_prediction_bias()
