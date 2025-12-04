import argparse
import os
import sys
import traceback
import pandas as pd

# Path Configuration
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.ingestion.poly_feed import PolymarketFeed
except Exception as e:
    print("--- FATAL ERROR IN INITIAL CONFIGURATION ---")
    print(f"Error: {e}")
    sys.exit(1)


def check_max_data_date():
    """
    Loads unified data, processes it through FeatureEngineer, and prints the maximum date
    in the resulting feature DataFrame.
    """
    print("📡 Loading and processing data to check the maximum date...")
    try:
        df_tweets = load_unified_data()
        feat_eng = FeatureEngineer()
        all_features = feat_eng.process_data(df_tweets)

        if not all_features.empty:
            max_date = all_features.index.max()
            print(
                f"\n✅ The most recent date in the processed data is: {max_date.date()}",
            )
        else:
            print("❌ The processed features DataFrame is empty.")

    except Exception as e:
        print(f"\n❌ An error occurred while checking the maximum data date: {e}")
        print(traceback.format_exc())


def merge_and_process_for_verification():
    """
    Loads, unifies, processes features, and saves both results to
    CSV files for detailed debugging.
    """
    print("🚀 Starting merge and processing for verification...")

    try:
        # --- 1. Load and Unify ---
        df_unified = load_unified_data()

        output_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)

        # Save the unified result
        unified_path = os.path.join(output_dir, "merged_tweets.csv")
        df_unified.to_csv(unified_path, index=False, encoding="utf-8")
        print(f"📄 Unified tweets file saved at: {unified_path}")

        # --- 2. Process Features ---
        print("\n⚙️  Generating features from unified data...")
        feature_engineer = FeatureEngineer()
        all_features = feature_engineer.process_data(df_unified)

        # Save the feature engineering result
        features_path = os.path.join(output_dir, "verified_features.csv")
        all_features.to_csv(features_path, index=True, encoding="utf-8")
        print(f"📄 Verified features file saved at: {features_path}")

        print("\n✅ Verification completed successfully.")
        print(
            "🔍 You can now inspect 'merged_tweets.csv' and 'verified_features.csv'.",
        )

    except Exception as e:
        print(f"❌ Error during the verification process: {e}")


def verify_feature_engineering():
    """
    Verification script for the Feature Engineering pipeline and data split.
    It loads data, processes features, and simulates the split to validate
    the correction of the timezone comparison error.
    """
    print("--- Verifying Feature Engineering and Data Split ---")

    try:
        # 1. Load Unified Data
        print("📡 Loading unified data...")
        df_tweets = load_unified_data()
        if df_tweets.empty:
            raise ValueError("The tweets DataFrame is empty.")
        print("✅ Unified data loaded.")

        # 2. Feature Generation
        print("\n⚙️ Running FeatureEngineer.process_data...")
        feat_eng = FeatureEngineer()
        all_features = feat_eng.process_data(df_tweets)
        print("✅ All feature generation completed.")
        print(f"  -> Shape of the features DataFrame: {all_features.shape}")

        # 3. Simulate Data Split (the previous failure point)
        print("\n🔪 Simulating data split with market date...")

        # Create a realistic, yet fake, market start date (timezone-aware)
        market_start_date_aware = (
            pd.Timestamp("2025-11-25 17:00:00")
            .tz_localize("America/New_York")
            .tz_convert("UTC")
        )
        print(f"  -> Simulated market date (aware): {market_start_date_aware}")

        # The key fix: convert it to timezone-naive for comparison
        market_start_date_naive = market_start_date_aware.tz_localize(None)
        print(f"  -> Market date (naive) for comparison: {market_start_date_naive}")

        # Perform the split
        train_features = all_features[all_features.index < market_start_date_naive]
        predict_features = all_features.iloc[[-1]]

        print("\n✅ Data split performed successfully.")
        print(f"  -> Shape of train_features: {train_features.shape}")
        print(f"  -> Shape of predict_features: {predict_features.shape}")

        if train_features.empty or predict_features.empty:
            print("⚠️ Warning: One of the resulting DataFrames is empty.")
        else:
            print(
                "\n✅ Verification completed successfully. The split correction works.",
            )

    except Exception as e:
        print(f"\n❌ A fatal error occurred during verification: {e}")
        print(traceback.format_exc())


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


def verify_poly_feed():
    """
    Verification script for PolymarketFeed.
    Lists all active Elon Musk markets to find the correct one.
    """
    print("--- Verifying PolymarketFeed (Market Listing) ---")

    try:
        # 1. Instantiate the feed
        poly_feed = PolymarketFeed()
        if not poly_feed.valid:
            print("❌ Could not initialize ClobClient.")
            return

        # 2. Get all markets
        print("\n🔎 Fetching the list of active markets from Polymarket...")
        markets_resp = poly_feed._robust_api_call(
            poly_feed.client.get_markets, next_cursor="",
        )

        if not markets_resp or not markets_resp.get("data"):
            print("❌ Could not fetch the market list from the API.")
            return

        # 3. Filter and display relevant markets
        print("\n--- Found Elon Musk Markets ---")
        count = 0
        for market in markets_resp["data"]:
            question = market.get("question", "")
            if "elon" in question.lower() and "tweet" in question.lower():
                print("  ----------------------------------------")
                print(f"  📌 Question: {market.get('question')}")
                print(f"     Slug: {market.get('slug')}")
                print(f"     ID: {market.get('id')}")
                print(f"     Condition ID: {market.get('condition_id')}")
                count += 1

        if count == 0:
            print(
                "  [❌] No active market containing 'elon' and 'tweet' was found.",
            )

        print(f"\nSummary: Found {count} relevant markets.")

    except Exception as e:
        print(f"\n❌ A fatal error occurred during verification: {e}")


def verify_unified_data_ingestion():
    """
    Isolated verification script for the `unified_feed`.
    Calls `load_unified_data` and reports the structure of the resulting DataFrame.
    """
    print("--- Verifying Unified Data Feed ---")

    try:
        # 1. Execute the unification function
        df_unified = load_unified_data()

        # 2. Validate the result
        if df_unified is not None and not df_unified.empty:
            print("\n✅ Unification completed successfully.")
            print("\n--- Head of the Unified DataFrame ---")
            print(df_unified.head())
            print("\n--- Tail of the Unified DataFrame ---")
            print(df_unified.tail())
            print(f"\nMinimum Date: {df_unified['created_at'].min()}")
            print(f"Maximum Date: {df_unified['created_at'].max()}")
        else:
            print(
                "❌ The `load_unified_data` function returned an empty or None DataFrame.",
            )

    except Exception as e:
        print(f"\n❌ A fatal error occurred during verification: {e}")


def main():
    parser = argparse.ArgumentParser(description="Data verification tools.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "check_max_date",
            "merge_and_verify",
            "verify_feature_eng",
            "verify_ingestion_quality",
            "verify_poly_feed",
            "verify_unified_feed",
        ],
        help="The verification task to execute.",
    )
    args = parser.parse_args()

    if args.task == "check_max_date":
        check_max_data_date()
    elif args.task == "merge_and_verify":
        merge_and_process_for_verification()
    elif args.task == "verify_feature_eng":
        verify_feature_engineering()
    elif args.task == "verify_ingestion_quality":
        verify_data_quality()
    elif args.task == "verify_poly_feed":
        verify_poly_feed()
    elif args.task == "verify_unified_feed":
        verify_unified_data_ingestion()


if __name__ == "__main__":
    main()
