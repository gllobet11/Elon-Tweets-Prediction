"""
verify_ingestion_quality.py

This script performs a quality check on the ingested tweet data to ensure its
completeness, correctness, and integrity. It reads the final processed data
and generates a summary report with key quality metrics.

Checks performed:
- Date range and total tweet count.
- Duplicate tweet IDs.
- Time gap analysis to find potential missing data periods.
- Daily tweet volume analysis to spot unusual activity drops.
"""

import os
import sys

import pandas as pd

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.unified_feed import load_unified_data
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error during import: {e}")
    print(
        "Please ensure you are running this script from the project's root directory.",
    )
    sys.exit(1)


def verify_data_quality():
    """
    Loads the unified tweet data and runs a series of quality checks.
    """
    print("--- Running Data Quality Verification ---")

    try:
        df = load_unified_data()
        if df.empty:
            print("❌ ERROR: The dataset is empty. Cannot perform quality checks.")
            return
    except FileNotFoundError:
        print("❌ ERROR: `data/processed/merged_elon_tweets.csv` not found.")
        print("   Please run `run_ingest.py` first.")
        return

    # --- 1. Basic Information ---
    print("\n1. DATASET OVERVIEW")
    print("-" * 20)
    print(f"   • Total Tweets: {len(df)}")
    if "created_at" in df.columns and not df.empty:
        # Ensure 'created_at' is datetime
        df["created_at"] = pd.to_datetime(df["created_at"])
        print(f"   • Start Date:   {df['created_at'].min()}")
        print(f"   • End Date:     {df['created_at'].max()}")
    print("-" * 20)

    # --- 2. Duplicate Check ---
    print("\n2. DUPLICATE CHECK")
    print("-" * 20)
    num_duplicates = df["id"].duplicated().sum()
    if num_duplicates == 0:
        print("   ✅ PASSED: No duplicate tweet IDs found.")
    else:
        print(f"   ❌ FAILED: Found {num_duplicates} duplicate tweet IDs.")
    print("-" * 20)

    # --- 3. Gap Analysis ---
    print("\n3. GAP ANALYSIS (Time between consecutive tweets)")
    print("-" * 20)
    if "created_at" in df.columns and len(df) > 1:
        df_sorted = df.sort_values("created_at")
        gaps = df_sorted["created_at"].diff()

        print("   Top 5 largest time gaps found:")
        largest_gaps = gaps.nlargest(5)
        for index, gap in largest_gaps.items():
            tweet_before_time = df_sorted.loc[index - 1, "created_at"]
            print(f"   - Gap of {gap} found after tweet at {tweet_before_time}")
    else:
        print("   Skipped: Not enough data for gap analysis.")
    print("-" * 20)

    # --- 4. Volume Analysis ---
    print("\n4. DAILY VOLUME ANALYSIS")
    print("-" * 20)
    if "created_at" in df.columns and not df.empty:
        daily_counts = df.set_index("created_at").resample("D").size()

        print("   Statistics for daily tweet counts:")
        print(f"   - Mean:   {daily_counts.mean():.2f} tweets/day")
        print(f"   - Median: {daily_counts.median():.0f} tweets/day")
        print(f"   - Max:    {daily_counts.max()} tweets/day")
        print(f"   - Min:    {daily_counts.min()} tweets/day")

        days_with_zero_tweets = daily_counts[daily_counts == 0]
        if not days_with_zero_tweets.empty:
            print(
                f"\n   ⚠️ Found {len(days_with_zero_tweets)} days with ZERO tweets (could be normal).",
            )
            print(
                f"   First 5 occurrences: {days_with_zero_tweets.head().index.date.tolist()}",
            )
        else:
            print("\n   ✅ No days with zero tweets found in the dataset.")

    else:
        print("   Skipped: Not enough data for volume analysis.")
    print("-" * 20)


if __name__ == "__main__":
    verify_data_quality()
