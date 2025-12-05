import pickle
import os
import pandas as pd
from tabulate import tabulate

def get_feature_importance(model_path: str):
    """
    Loads a Prophet model from a .pkl file and extracts regressor coefficients
    to determine feature importance.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)

    m = model_package['model']
    regressors = model_package['regressors']
    
    if not regressors:
        print("No regressors were used in this model, so no feature importance to display.")
        return

    # Prophet stores regressor coefficients in m.params['beta']
    # The order corresponds to the order regressors were added
    # m.extra_features provides names in order of addition
    
    # Ensure coefficients are available
    if 'beta' not in m.params:
        print("Model does not contain 'beta' parameters for regressors.")
        return

    # Extract coefficients
    coefficients = m.params['beta'][0] # Beta has shape (1, num_regressors)

    # Create a DataFrame for display
    feature_importance_df = pd.DataFrame({
        'Feature': regressors,
        'Coefficient (Magnitude)': [abs(c) for c in coefficients],
        'Coefficient (Value)': coefficients
    })
    
    # Sort by magnitude for easier interpretation
    feature_importance_df = feature_importance_df.sort_values(
        by='Coefficient (Magnitude)', ascending=False
    ).reset_index(drop=True)

    print("\n--- Feature Importance for the Best Model ---")
    print(tabulate(feature_importance_df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    print("\nNote: Magnitude indicates importance. Positive value means feature increases tweet count, negative means it decreases.")

if __name__ == "__main__":
    model_file = "best_prophet_model_20251205.pkl"
    get_feature_importance(model_file)
