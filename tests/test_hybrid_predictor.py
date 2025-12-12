"""
test_hybrid_predictor.py

Tests for the refactored, recursive walk-forward prediction logic.
"""

import os
import sys
import pickle
import glob
import pandas as pd
import pytest

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategy.hybrid_predictor import get_hybrid_prediction
from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer


@pytest.fixture(scope="module")
def prophet_model():
    """Loads the latest trained Prophet model for testing."""
    model_files = glob.glob(os.path.join(project_root, "best_prophet_model_*.pkl"))
    if not model_files:
        pytest.fail("A trained Prophet model file is required for this test.")
    latest_model_path = max(model_files, key=os.path.getmtime)
    with open(latest_model_path, "rb") as f:
        model_package = pickle.load(f)
    return model_package["model"]


@pytest.fixture(scope="module")
def all_features_df():
    """
    Loads real data and generates features to provide a realistic
    historical context for the recursive prediction.
    """
    df_granular = load_unified_data()
    if df_granular.empty:
        pytest.fail("Could not load unified data for creating test features.")

    feature_engineer = FeatureEngineer()
    features = feature_engineer.process_data(df_granular)
    return features


def test_recursive_prediction_logic(prophet_model, all_features_df):
    """
    Tests the recursive walk-forward prediction logic.

    Verifies that:
    1. It runs without errors and produces a forecast.
    2. The forecast contains the correct number of days.
    3. The predicted values are reasonable (not NaN and sum > 0).
    4. The total prediction in the metrics dictionary matches the sum of the DataFrame.
    """
    # Arrange
    days_to_predict = 7

    # Act
    predictions_df, metrics = get_hybrid_prediction(
        prophet_model=prophet_model,
        all_features_df=all_features_df,
        days_forward=days_to_predict,
    )

    # Assert
    # 1. Check structure and types
    assert isinstance(predictions_df, pd.DataFrame), "Should return a DataFrame."
    assert isinstance(metrics, dict), "Should return a metrics dictionary."
    assert not predictions_df.empty, "Prediction DataFrame should not be empty."

    # 2. Check dimensions
    assert (
        len(predictions_df) == days_to_predict
    ), f"Should predict for {days_to_predict} days."

    # 3. Check data integrity and reasonableness
    assert (
        not predictions_df["y_pred"].isnull().any()
    ), "Predictions should not contain any NaNs."
    assert (
        predictions_df["y_pred"].sum() > 0
    ), "Sum of predictions should be a positive value."

    # 4. Check metrics consistency
    assert "weekly_total_prediction" in metrics
    assert (
        metrics["weekly_total_prediction"] == predictions_df["y_pred"].sum()
    ), "Metrics should match the DataFrame."
    assert metrics["sum_of_actuals"] == 0, "Actuals should be zero in a pure forecast."


def test_recursive_predictor_handles_empty_input(prophet_model):
    """
    Tests that the recursive predictor handles an empty DataFrame gracefully.
    """
    # Arrange
    empty_df = pd.DataFrame()

    # Act
    predictions_df, metrics = get_hybrid_prediction(
        prophet_model=prophet_model, all_features_df=empty_df, days_forward=7
    )

    # Assert
    assert predictions_df.empty
    assert isinstance(metrics, dict)
    assert "weekly_total_prediction" not in metrics  # Or check for zero values
