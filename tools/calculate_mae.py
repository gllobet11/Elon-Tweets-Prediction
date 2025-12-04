import pandas as pd
import os

# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
performance_file = os.path.join(project_root, 'data', 'processed', 'all_models_historical_performance.csv')
best_model_column = 'y_pred_Final_Selected_Model'

# --- Calculation ---
try:
    df = pd.read_csv(performance_file)
    
    if 'y_true' in df.columns and best_model_column in df.columns:
        # Drop rows where there might be missing values
        df_clean = df[['y_true', best_model_column]].dropna()
        
        # Calculate MAE
        mae = (df_clean['y_true'] - df_clean[best_model_column]).abs().mean()
        
        print("\n--- Backtest Performance Metrics for Best Model ---")
        print(f"Model: '{best_model_column.replace('y_pred_', '')}'")
        print(f"Mean Absolute Error (MAE): {mae:.2f} tweets")
        print("\nThis means that, on average, the model's weekly predictions during the backtest period were off by about " + f"{mae:.2f} tweets.")
        
    else:
        print(f"Error: Required columns ('y_true' or '{best_model_column}') not found in the performance file.")

except FileNotFoundError:
    print(f"Error: The performance file was not found at '{performance_file}'.")
    print("Please ensure the model evaluation pipeline has been run successfully.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

