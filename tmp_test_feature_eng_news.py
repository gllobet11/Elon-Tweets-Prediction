import pandas as pd
import sys
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress loguru console output during this test for cleaner output
logger.remove()
logger.add(sys.stderr, level="INFO")  # Keep INFO level for specific messages

try:
    from src.processing.feature_eng import FeatureEngineer
except ImportError as e:
    logger.error(f"Failed to import FeatureEngineer: {e}")
    sys.exit(1)


def create_mock_daily_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Creates a mock DataFrame resembling daily processed tweet data (n_tweets)."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    # Simulate some daily tweet counts
    data = {"n_tweets": np.random.randint(0, 50, size=len(dates))}
    df = pd.DataFrame(data, index=dates)
    df.index.name = (
        "created_at"  # process_data expects DatetimeIndex or 'created_at' column
    )
    return df


if __name__ == "__main__":
    logger.info("--- Starting FeatureEngineer News Integration Test ---")

    test_start_date = "2025-11-01"
    test_end_date = "2025-12-10"

    mock_df_raw = create_mock_daily_data(test_start_date, test_end_date)

    fe = FeatureEngineer()
    processed_df = fe.process_data(mock_df_raw)

    logger.info("--- Processed DataFrame Head (News features) ---")
    print(
        processed_df[["n_tweets", "news_vol_log", "avg_sentiment"]].head(20).to_string()
    )

    logger.info("--- Processed DataFrame Tail (News features) ---")
    print(
        processed_df[["n_tweets", "news_vol_log", "avg_sentiment"]].tail(20).to_string()
    )

    # Basic assertion
    if (
        "news_vol_log" in processed_df.columns
        and "avg_sentiment" in processed_df.columns
    ):
        logger.success("News features 'news_vol_log' and 'avg_sentiment' are present!")
        # Further checks: check for non-zero values if news data is expected
        if (
            processed_df["news_vol_log"].sum() > 0
            or processed_df["avg_sentiment"].sum() != 0
        ):
            logger.success("Detected non-zero news values in the test period.")
        else:
            logger.warning(
                "News features are all zero. Check 'elon_news_cleaned.csv' or date range."
            )
    else:
        logger.error("News features are NOT present in the processed DataFrame.")

    logger.info("--- FeatureEngineer News Integration Test Completed ---")
