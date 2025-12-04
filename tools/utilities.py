import argparse
import os
import pickle
import glob
import sys # Added sys import
from pprint import pprint

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config.bins_definition import MARKET_BINS
    from src.ingestion.poly_feed import PolymarketFeed
except Exception as e:
    print(f"Error configuring project root or importing modules: {e}")
    sys.exit(1)

def inspect_pkl_file():
    """
    Loads a pickle file and inspects its content.
    """
    try:
        model_files = glob.glob("best_prophet_model_*.pkl")
        if not model_files:
            print("❌ No model .pkl file found.")
            return

        latest_model_path = max(model_files, key=os.path.getmtime)
        print(f"🔍 Inspecting file: {latest_model_path}")

        with open(latest_model_path, "rb") as f:
            loaded_object = pickle.load(f)

        print("\n✅ Object loaded successfully.")
        print(f"   -> Type of loaded object: {type(loaded_object)}")

        if isinstance(loaded_object, dict):
            print("   -> Object is a dictionary. Keys:")
            for key in loaded_object.keys():
                print(f"      - {key}")
        else:
            print("\n   -> Content of the object:")
            print(loaded_object)

    except Exception as e:
        print(f"❌ An error occurred: {e}")

def test_final_price_logic():
    """
    Final test script to verify price fetching logic.
    """
    print("--- Final Price Logic Test ---")

    MARKET_KEYWORDS = ["elon musk", "tweets", "november 25", "december 2"]

    try:
        poly_feed = PolymarketFeed()
        if not poly_feed.valid:
            print("❌ Could not initialize ClobClient.")
            return

        # 1. Map IDs
        print("\n🔎 Mapping 'Yes' and 'No' token IDs for each bin...")
        updated_bins = poly_feed.fetch_market_ids_automatically(
            keywords=MARKET_KEYWORDS, bins_dict=MARKET_BINS,
        )

        # 2. Get prices
        print("\n💰 Fetching prices with final valuation logic...")
        price_snapshot = poly_feed.get_all_bins_prices(updated_bins)

        # 3. Print results
        print("\n--- Final Price Snapshot Obtained ---")
        pprint(price_snapshot)

    except Exception as e:
        print(f"\n❌ A fatal error occurred during the test: {e}")

def main():
    parser = argparse.ArgumentParser(description="Utility tools.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["inspect_pkl_file", "test_market_prices"],
        help="The utility task to execute.",
    )
    args = parser.parse_args()

    if args.task == "inspect_pkl_file":
        inspect_pkl_file()
    elif args.task == "test_market_prices":
        test_final_price_logic()

if __name__ == "__main__":
    main()
