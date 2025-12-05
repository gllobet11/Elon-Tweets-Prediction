import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta

from src.processing.feature_eng import FeatureEngineer

import os
import shutil # Added for clearing debug directory

def get_hybrid_prediction(
    prophet_model,
    all_features_df: pd.DataFrame,
    days_forward: int = 7,
    debug_mode: bool = False,
    debug_path: str = "debug_dumps"
) -> tuple[pd.DataFrame, dict]:
    """
    Generates predictions using a recursive walk-forward strategy.
    For each future day, it predicts the value, then uses that value
    to compute the features for the next day.
    """
    if all_features_df.empty:
        logger.error("Feature DataFrame is empty. Cannot predict.")
        return pd.DataFrame(), {}

    if debug_mode:
        logger.info(f"Debug mode enabled. Snapshots will be saved to: {debug_path}")
        if os.path.exists(debug_path):
            shutil.rmtree(debug_path) # Clear previous debug dumps
        os.makedirs(debug_path, exist_ok=True)
    
    # Initialize a FeatureEngineer instance
    feature_engineer = FeatureEngineer()

    # Initialize a FeatureEngineer instance
    feature_engineer = FeatureEngineer()
    
    # Start with the most recent historical data as the seed
    latest_features = all_features_df.iloc[[-1]].copy()
    
    all_predictions = []
    
    # Use a copy of the historical data that we can extend with new predictions
    extended_history = all_features_df.copy()

    for i in range(days_forward):
        # 1. Prepare the DataFrame for this step's prediction
        # The features for the day we want to predict are based on the last row of our history.
        # Prophet expects the 'ds' column to be the date it is predicting for.
        prediction_date = latest_features.index[0] + timedelta(days=1)
        
        current_step_df = latest_features.copy()
        current_step_df = current_step_df.reset_index(drop=True) # Reset index to avoid issues
        current_step_df['ds'] = prediction_date # Set ds to the date we are predicting

        # Ensure 'ds' is timezone-naive for Prophet
        if current_step_df["ds"].dt.tz is not None:
            current_step_df["ds"] = current_step_df["ds"].dt.tz_localize(None)

        # 2. Predict for the next day
        forecast = prophet_model.predict(current_step_df)
        predicted_tweets = forecast["yhat"].iloc[0]
        
        # Store the prediction
        prediction_date = latest_features.index[0] + timedelta(days=1)
        
        # --- DEBUG SNAPSHOT: Predicted Tweets Value ---
        if debug_mode:
            step_num = f"{i:03d}"
            with open(f"{debug_path}/step_{step_num}_00_prediction_details.txt", "w") as f:
                f.write(f"Iteration: {i}\n")
                f.write(f"Prediction Date: {prediction_date}\n")
                f.write(f"Raw Prophet yhat: {predicted_tweets}\n")
                f.write(f"Rounded n_tweets: {max(0, int(round(predicted_tweets)))}\n")
            
            # Save the input to Prophet for this step
            current_step_df.to_csv(f"{debug_path}/step_{step_num}_01_prophet_input.csv", index=False)

        # Apply rounding and ensure non-negative for discrete counts
        predicted_tweets = max(0, int(round(predicted_tweets))) # Apply suggested rounding here

        all_predictions.append({"ds": prediction_date, "y_pred": predicted_tweets})

        # 3. Prepare for the *next* iteration: update features with the new prediction
        # Create a new row for the next day
        next_day_features = latest_features.copy()
        next_day_features.index = [prediction_date]
        
        # Update the 'n_tweets' with our new prediction. This is the recursive step.
        next_day_features['n_tweets'] = predicted_tweets # Now this is the rounded integer value
        
        # CRITICAL FIX: Update lagged features in next_day_features based on the *newly predicted n_tweets*
        # This prevents lag_1 from carrying over historical actuals into recursive predictions.
        next_day_features['lag_1'] = predicted_tweets
        # Other lagged/rolling features will be correctly re-calculated by feature_engineer.process_data

        # Append this new predicted row to our ongoing history
        extended_history = pd.concat([extended_history, next_day_features])
        
        # Recalculate features based on this extended history.
        # This will correctly update 'lag_1', 'roll_sum_7', etc. for the *next* day to be predicted.
        # Ensure enough history for rolling features, so pass the entire extended_history.
        recalculated_features = feature_engineer.process_data(extended_history)
        
        # --- DEBUG: Check columns after feature engineering ---
        if debug_mode:
            logger.debug(f"Step {i:03d}: Recalculated features columns: {recalculated_features.columns.tolist()}")
            logger.debug(f"Step {i:03d}: Recalculated features tail:\n{recalculated_features.tail()}")

        # The last row of the newly calculated features is our input for the next loop iteration
        latest_features = recalculated_features.iloc[[-1]].copy()

        # --- DEBUG: Check columns of latest_features ---
        if debug_mode:
            logger.debug(f"Step {i:03d}: Latest features columns (for next Prophet input): {latest_features.columns.tolist()}")

        # --- DEBUG SNAPSHOT: Engineered Features Output ---
        if debug_mode:
            step_num = f"{i:03d}"
            latest_features.to_csv(f"{debug_path}/step_{step_num}_02_engineered_output.csv")

    predictions_df = pd.DataFrame(all_predictions)
    
    # Calculate final metrics
    sum_predictions = predictions_df["y_pred"].sum()
    metrics = {
        "weekly_total_prediction": sum_predictions,
        "sum_of_predictions": sum_predictions,
        "sum_of_actuals": 0, # In a pure forecast, actuals are 0
        "remaining_days_fraction": days_forward
    }
    
    return predictions_df, metrics