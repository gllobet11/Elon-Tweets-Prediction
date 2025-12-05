import pandas as pd
import numpy as np
from loguru import logger

def extract_prophet_coefficients(model, regressor_names: list) -> pd.DataFrame:
    """
    Safely extracts coefficients for specific regressors from a Prophet model.
    
    Args:
        model: The trained Prophet model object.
        regressor_names: List of strings representing the custom regressors to extract.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Coefficient', 'Abs_Coefficient']
    """
    if 'beta' not in model.params:
        logger.warning("Model does not contain 'beta' params. Was it trained with MCMC or MAP?")
        return pd.DataFrame()

    beta = model.params['beta']
    # If beta is (n_samples, n_features), take the mean across samples
    if beta.ndim > 1 and beta.shape[0] > 1:
        beta = beta.mean(axis=0)
    elif beta.ndim > 1:
        beta = beta[0]

    feature_data = []

    for name in regressor_names:
        # Check if the regressor exists in the model's component columns
        if name not in model.train_component_cols.columns:
            logger.warning(f"Regressor '{name}' not found in model components. Skipping.")
            continue

        # Find the column index in the beta matrix corresponding to this regressor
        # train_component_cols is a matrix where 1 indicates the feature is active
        col_index = np.where(model.train_component_cols[name] == 1)[0]

        if len(col_index) > 0:
            # Extract coefficient (summing if it spans multiple columns, though usually it's 1-to-1 for linear)
            coeff_value = np.sum(beta[col_index])
            feature_data.append({
                'Feature': name,
                'Coefficient': coeff_value,
                'Abs_Coefficient': abs(coeff_value)
            })

    if not feature_data:
        return pd.DataFrame(columns=['Feature', 'Coefficient', 'Abs_Coefficient'])

    return pd.DataFrame(feature_data).sort_values(by='Abs_Coefficient', ascending=False)
