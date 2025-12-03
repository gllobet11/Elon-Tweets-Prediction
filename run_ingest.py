"""
run_ingest.py

This script orchestrates a hybrid data ingestion process. It combines a static,
long-term historical file (elonmusk.csv) with the latest tweets fetched directly
from the x-tracker API.

This approach ensures a deep historical record while keeping the data up-to-date
with the most recent activity.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

# --- Path Configuration ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ingestion.api_ingestor import ApiIngestor

    # Import the snowflake converter utility
    from src.ingestion.xtracker_feed import snowflake_to_dt_utc
except ImportError as e:
    logger.error(f"Could not import modules. Error: {e}")
    sys.exit(1)

# --- Path Configuration ---
raw_dir = os.path.join(project_root, "data", "raw")
processed_dir = os.path.join(project_root, "data", "processed")
historical_base_path = os.path.join(raw_dir, "elonmusk.csv")
merged_output_path = os.path.join(processed_dir, "merged_elon_tweets.csv")

os.makedirs(processed_dir, exist_ok=True)


def load_historical_base():
    """Loads and cleans the static, long-term historical CSV."""
    if not os.path.exists(historical_base_path):
        logger.error(f"Historical base file not found: {historical_base_path}")
        return pd.DataFrame()

    logger.info(
        f"Loading historical base from '{os.path.basename(historical_base_path)}'...",
    )
    try:
        df_hist = pd.read_csv(historical_base_path, sep=",", on_bad_lines="skip")

        # Clean column names
        df_hist.columns = [c.strip().lower() for c in df_hist.columns]

        # Ensure 'id' column exists
        if "id" not in df_hist.columns:
            logger.error("Historical base file is missing the 'id' column.")
            return pd.DataFrame()

        # Create 'created_at' from snowflake ID
        df_hist["created_at"] = df_hist["id"].apply(snowflake_to_dt_utc)

        # Standardize columns
        df_hist = df_hist[["id", "text", "created_at"]]
        df_hist.dropna(subset=["id", "created_at"], inplace=True)

        logger.success(f"Loaded {len(df_hist)} base historical tweets.")
        return df_hist
    except Exception as e:
        logger.error(f"Failed to load or process historical base file. Error: {e}")
        return pd.DataFrame()


def run_data_ingestion():
    """Main function to run the hybrid data ingestion process."""
    logger.info("--- Starting HYBRID data ingestion process ---")

    # 1. Load the long-term historical base
    df_hist = load_historical_base()
    if df_hist.empty:
        logger.error("Cannot proceed without the historical base file. Exiting.")
        return

    # 2. Fetch new tweets from the API since the last historical tweet
    last_hist_date = df_hist["created_at"].max()
    logger.info(
        f"Last tweet in historical base is from {last_hist_date}. Fetching new tweets since then.",
    )

    ingestor = ApiIngestor()
    df_new = ingestor.fetch(
        start_date=last_hist_date, end_date=datetime.now(timezone.utc),
    )

    if df_new.empty:
        logger.warning(
            "ApiIngestor returned no new tweets. The dataset may already be up-to-date.",
        )
    else:
        logger.info(f"Fetched {len(df_new)} new tweets from the API.")

    # 3. Unify, Deduplicate, and Save
    df_unified = pd.concat([df_hist, df_new], ignore_index=True)
    logger.info(
        f"Unification: {len(df_hist)} historical + {len(df_new) if not df_new.empty else 0} new = {len(df_unified)} total rows.",
    )

    # --- Data Cleaning on the unified frame ---
    # Ensure all dates are consistent timezone-aware objects
    df_unified["created_at"] = pd.to_datetime(
        df_unified["created_at"], utc=True, errors="coerce",
    )
    df_unified["id"] = pd.to_numeric(df_unified["id"], errors="coerce")
    df_unified.dropna(subset=["id", "created_at"], inplace=True)
    df_unified["id"] = df_unified["id"].astype("int64")

    # Deduplicate and sort
    initial_count = len(df_unified)
    df_unified.drop_duplicates(subset=["id"], keep="last", inplace=True)
    df_unified.sort_values("created_at", inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    final_count = len(df_unified)

    logger.info(f"Deduplication complete. Final total: {final_count} unique tweets.")
    if not df_unified.empty:
        logger.info(
            f"Final unified data range: {df_unified['created_at'].min().date()} to {df_unified['created_at'].max().date()}",
        )

    df_unified.to_csv(merged_output_path, index=False)
    logger.success(
        f"--- Unified data saved successfully to '{os.path.basename(merged_output_path)}' ---",
    )


if __name__ == "__main__":
    run_data_ingestion()
