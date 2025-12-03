import sys
import os
import pandas as pd
import numpy as np
import hashlib

# Add project root to path so we can import tools
# This needs to be adjusted because financial_optimizer is in src/strategy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the specific function - adjust import path as needed
from src.strategy.financial_optimizer import simulate_trading_run
# Assuming generate_backtest_predictions is gone, and we load from CSV
# If generate_backtest_predictions is used, we need to adapt this.
# For this test, we will load from CSV directly.


def get_file_hash(filepath):
    """Returns SHA-256 hash of a file."""
    if not os.path.exists(filepath): return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f:
        bytes = f.read()
        return hashlib.sha256(bytes).hexdigest()

def run_test():
    print("--- DIAGNOSTIC: DETERMINISM CHECK ---")
    
    # 1. Verify Input Stability
    input_file = os.path.join(project_root, 'data', 'processed', 'historical_performance.csv')
    print(f"1. Input File Hash: {get_file_hash(input_file)}")
    
    # Load data (Simulate how financial_optimizer.py does it)
    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} not found. Please run tools/generate_historical_performance.py first.")
        return

    df = pd.read_csv(input_file)
    
    # Store a copy to check for mutation later
    df_initial_state = df.copy()
    
    # Parameters to test
    ALPHA = 0.05
    KELLY = 0.2
    
    print(f"\n2. Running Simulation A (Alpha={ALPHA}, Kelly={KELLY})...")
    equity_A, mdd_A = simulate_trading_run(df, ALPHA, KELLY)
    print(f"   -> Final Capital: {equity_A.iloc[-1]:.4f}")
    print(f"   -> Max DD: {mdd_A:.4f}")

    print(f"\n3. Running Simulation B (Alpha={ALPHA}, Kelly={KELLY})...")
    # CRITICAL: We pass the SAME df object to see if Run A broke it
    equity_B, mdd_B = simulate_trading_run(df, ALPHA, KELLY) 
    print(f"   -> Final Capital: {equity_B.iloc[-1]:.4f}")
    print(f"   -> Max DD: {mdd_B:.4f}")
    
    # 4. Compare Results
    if equity_A.iloc[-1] == equity_B.iloc[-1]:
        print("\n✅ SUCCESS: Results are DETERMINISTIC.")
    else:
        print("\n❌ FAILURE: Results DIVERGED.")
        
    # 5. Check for Data Mutation
    print("\n5. Checking for DataFrame Mutation...")
    try:
        pd.testing.assert_frame_equal(df, df_initial_state)
        print("✅ DataFrame remained static (Good).")
    except AssertionError:
        print("❌ CRITICAL: The dataframe was modified by the function! (Bad)")
        print("   This explains why the second run differs.")

if __name__ == "__main__":
    run_test()
