"""
test_dashboard_logic.py

Tests for the DashboardLogicProcessor class to ensure its methods
are functioning correctly.
"""

import os
import sys
import pickle
import glob
import pandas as pd
import pytest

# --- Path Configuration ---
# This ensures that the script can be run from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dashboard.dashboard_logic_processor import DashboardLogicProcessor

@pytest.fixture(scope="module")
def model_data():
    """
    Pytest fixture to load the latest trained Prophet model.
    This runs once per module, providing the model data to the tests.
    """
    model_files = glob.glob(os.path.join(project_root, "best_prophet_model_*.pkl"))
    if not model_files:
        pytest.fail("No Prophet model file (.pkl) found. A trained model is required for this test.")
    
    latest_model_path = max(model_files, key=os.path.getmtime)
    with open(latest_model_path, "rb") as f:
        loaded_model_data = pickle.load(f)
    return loaded_model_data

@pytest.fixture(scope="module")
def logic_processor():
    """
    Pytest fixture to provide an instance of the DashboardLogicProcessor.
    """
    return DashboardLogicProcessor()

def test_get_feature_importance_runs_successfully(logic_processor, model_data):
    """
    Tests that get_feature_importance runs without errors and returns a DataFrame.
    This is the primary test for the fix of the 'ValueError: All arrays must be of the same length'.
    """
    # Act
    feature_importance_df = logic_processor.get_feature_importance(model_data)

    # Assert
    assert isinstance(feature_importance_df, pd.DataFrame), "The function should always return a DataFrame."

    # If the model has regressors, we expect a non-empty frame with specific columns
    if model_data.get("regressors"):
        assert not feature_importance_df.empty, "DataFrame should not be empty if regressors are present."
        
        expected_columns = ['Feature', 'Coefficient', 'Magnitude', 'Impact']
        for col in expected_columns:
            assert col in feature_importance_df.columns, f"Expected column '{col}' not found in the DataFrame."
        
        # Check that the number of features in the output matches the number of regressors
        # This implicitly confirms the fix for the length mismatch error
        assert len(feature_importance_df) == len(model_data.get("regressors", [])), "The number of rows should match the number of regressors."

    else:
        # If no regressors, an empty DataFrame is expected
        assert feature_importance_df.empty, "An empty DataFrame is expected if the model has no regressors."

def test_get_feature_importance_no_regressors(logic_processor, model_data):
    """
    Tests that get_feature_importance returns an empty DataFrame if the model
    data contains no 'regressors' key.
    """
    # Arrange
    model_data_no_regressors = model_data.copy()
    if 'regressors' in model_data_no_regressors:
        del model_data_no_regressors['regressors']

    # Act
    feature_importance_df = logic_processor.get_feature_importance(model_data_no_regressors)

    # Assert
    assert isinstance(feature_importance_df, pd.DataFrame)
    assert feature_importance_df.empty, "An empty DataFrame is expected when no regressors are provided."

