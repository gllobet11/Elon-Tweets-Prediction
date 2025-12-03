"""
utils.py

This module contains common utility functions shared across different parts
of the strategy and analysis codebase.
"""

from datetime import datetime, timedelta
import pandas as pd

def get_last_complete_friday(last_data_date) -> datetime:
    """
    Finds the last Friday that can start a complete 7-day forecast window,
    ensuring that ground truth data is available for the full week.

    Args:
        last_data_date (datetime or pd.Timestamp): The date of the last available data point.

    Returns:
        datetime: A datetime object representing the last complete Friday.
    """
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    # A forecast window starting on a Friday needs 7 days of data, ending on the next Thursday.
    # So the latest possible end date for ground truth is `last_data_date`.
    # This means the latest possible start date is 6 days before that.
    last_possible_forecast_start = last_data_date - timedelta(days=6)
    
    # Now, find the Friday of that week or the one before it.
    # weekday() -> Monday is 0 and Sunday is 6. Friday is 4.
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    return last_possible_forecast_start - timedelta(days=days_since_friday)
