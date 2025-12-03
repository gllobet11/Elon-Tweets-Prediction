import os
import sys
import pandas as pd
import pytest
from datetime import datetime, timedelta, date

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.generate_historical_performance import generate_backtest_predictions
from src.ingestion.unified_feed import load_unified_data

# --- Test Fixture for Mock Data ---

@pytest.fixture
def mock_tweet_data(monkeypatch):
    """
    Creates a mock dataset of tweets with a consistent daily count for a full year
    and mocks the data loading function to use it.
    """
    print("Setting up mock tweet data for the full year 2025...")
    
    daily_tweets = 20
    start_date = datetime(2025, 1, 1).date()
    end_date = datetime(2025, 12, 31).date()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
    tweet_times = []
    for d in dates:
        for i in range(daily_tweets):
            mock_time = datetime.combine(d, datetime.min.time()) + timedelta(hours=i)
            tweet_times.append(mock_time)
            
    mock_df = pd.DataFrame(tweet_times, columns=['created_at'])
    mock_df['id'] = range(len(mock_df))
    mock_df['text'] = "This is a mock tweet."

    def mock_load():
        print(f"Mock `load_unified_data` called. Returning DataFrame with {len(mock_df)} rows.")
        return mock_df

    monkeypatch.setattr('tools.generate_historical_performance.load_unified_data', mock_load)
    
    expected_y_true = 7 * daily_tweets
    
    yield expected_y_true
    
    print("Tearing down mock tweet data.")


# --- Unit Test ---

def test_generate_backtest_predictions(mock_tweet_data):
    """
    Tests the `generate_backtest_predictions` function to ensure it correctly
    calculates `y_true` and that `y_pred` is within a reasonable tolerance.
    """
    expected_y_true = mock_tweet_data
    weeks_to_validate = 2
    # Use a fixed end_date to make the test reproducible.
    # The backtester will find the last complete Friday before this date.
    # Last complete Friday before Nov 20 is Nov 14.
    # Validation weeks will start on Nov 14 and Nov 7.
    # Both weeks are fully contained within our 2025 mock data.
    test_end_date = date(2025, 11, 20) 

    print(f"Running `generate_backtest_predictions` for {weeks_to_validate} weeks ending on {test_end_date}...")
    df_performance = generate_backtest_predictions(
        weeks_to_validate=weeks_to_validate,
        end_date=test_end_date
    )

    print("Running assertions on the output DataFrame...")
    
    assert not df_performance.empty, "The performance DataFrame should not be empty."
    expected_cols = {'week_start_date', 'y_true', 'y_pred'}
    assert set(df_performance.columns) == expected_cols, f"DataFrame columns are wrong. Expected {expected_cols}"
    
    assert len(df_performance) == weeks_to_validate, f"Expected exactly {weeks_to_validate} rows, but got {len(df_performance)}"
    
    assert all(df_performance['y_true'] == expected_y_true), f"y_true should be {expected_y_true} for all weeks. Got: {df_performance['y_true'].tolist()}"

    tolerance = 10
    prediction_errors = abs(df_performance['y_pred'] - expected_y_true)
    assert all(prediction_errors < tolerance), f"y_pred is not within the tolerance of {tolerance}. Errors: {prediction_errors.tolist()}"
    
    print("✅ All assertions passed.")
