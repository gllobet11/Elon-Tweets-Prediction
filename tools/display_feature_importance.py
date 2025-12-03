import os
import pickle
import sys
from datetime import datetime

import pandas as pd

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    print(f"Error configuring project root: {e}")
    sys.exit(1)


def display_prophet_regressor_coefficients():
    """
    Loads the latest saved Prophet model and displays the coefficients of its regressors.
    """
    model_files = [
        f
        for f in os.listdir(project_root)
        if f.startswith("best_prophet_model_") and f.endswith(".pkl")
    ]

    if not model_files:
        print(
            "No Prophet model files found. Please ensure 'tools/models_evals.py' has been run to train and save a model.",
        )
        return

    # Sort to get the latest model
    model_files.sort(
        key=lambda x: datetime.strptime(x, "best_prophet_model_%Y%m%d.pkl"),
        reverse=True,
    )
    latest_model_file = os.path.join(project_root, model_files[0])

    print(f"Loading latest model: {latest_model_file}")
    with open(latest_model_file, "rb") as f:
        model_package = pickle.load(f)

    m = model_package["model"]
    regressors = model_package.get("regressors", [])

    if not regressors:
        print("The loaded model was trained without any extra regressors.")
        return

    # --- Debugging starts here ---
    print(f"DEBUG: m.params['beta'] shape: {m.params['beta'].shape}")
    print(f"DEBUG: m.train_component_cols head:\n{m.train_component_cols.head()}")
    # --- Debugging ends here ---

    if regressors and "beta" in m.params:
        beta_coefficients = m.params["beta"]

        feature_coefficients = []

        for regressor_name in regressors:
            if regressor_name in m.train_component_cols.columns:
                component_indices = m.train_component_cols[
                    m.train_component_cols[regressor_name] == 1
                ].index.values
                print(
                    f"DEBUG: Regressor '{regressor_name}', component_indices: {component_indices}",
                )  # Debug individual indices

                if len(component_indices) > 0:
                    coeff_mean = beta_coefficients.mean(axis=0)[
                        component_indices
                    ].mean()
                    feature_coefficients.append(
                        {"Regressor": regressor_name, "Coefficient": coeff_mean},
                    )
                else:
                    print(
                        f"Warning: No coefficient found for regressor '{regressor_name}' in m.params['beta'].",
                    )
            else:
                print(
                    f"Warning: Regressor '{regressor_name}' not found in m.train_component_cols.",
                )

        if feature_coefficients:
            regressor_coefficients_df = pd.DataFrame(feature_coefficients)
            regressor_coefficients_df["Abs_Coefficient"] = regressor_coefficients_df[
                "Coefficient"
            ].abs()
            regressor_coefficients_df = regressor_coefficients_df.sort_values(
                by="Abs_Coefficient", ascending=False,
            )

            print(
                "\n--- Prophet Regressor Coefficients (Absolute Value as Importance Proxy) ---",
            )
            print(regressor_coefficients_df.to_string(index=False))
            print(
                "\nNote: A higher absolute coefficient generally indicates a stronger linear relationship with the target variable.",
            )
            print(
                "Positive coefficients indicate a positive relationship, negative indicate a negative relationship.",
            )
        else:
            print("No extra regressor coefficients could be extracted.")
    else:
        print(
            "No 'beta' parameter or regressors found in the model. This model might not have been trained correctly or lacks components.",
        )


if __name__ == "__main__":
    display_prophet_regressor_coefficients()
