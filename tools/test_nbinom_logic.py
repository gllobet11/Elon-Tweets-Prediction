import sys
import os
import pandas as pd

# Ensure project root is in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategy.prob_math import DistributionConverter
from config.bins_definition import MARKET_BINS

def run_stability_test():
    """
    Tests the stability and behavior of the Negative Binomial (nbinom) distribution
    compared to Poisson, based on different `mu` (mean) and `alpha` (dispersion) values.
    """
    print("=" * 80)
    print("🔬 Running Distribution Stability Test 🔬")
    print("=" * 80)
    print("This test compares Poisson vs. Negative Binomial (NBinom) distributions.")
    print("Key insight: NBinom uses an 'alpha' parameter to model dispersion.")
    print("A higher alpha means 'fatter tails' (higher probability of extreme events).\n")

    # Bins configuration for the test
    bins_config = [(k, v['lower'], v['upper']) for k, v in MARKET_BINS.items()]

    # --- Test Scenarios ---
    scenarios = [
        {"name": "Low Mean, Low Dispersion", "mu": 220, "alpha": 0.01},
        {"name": "Low Mean, High Dispersion", "mu": 220, "alpha": 0.1},
        {"name": "High Mean, Low Dispersion", "mu": 280, "alpha": 0.01},
        {"name": "High Mean, High Dispersion", "mu": 280, "alpha": 0.1},
        {"name": "High Mean, VERY High Dispersion", "mu": 280, "alpha": 0.3},
    ]

    for scenario in scenarios:
        mu = scenario['mu']
        alpha = scenario['alpha']

        print("-" * 80)
        print(f"🎬 SCENARIO: {scenario['name']} (mu={mu}, alpha={alpha})")
        print("-" * 80)

        # --- Calculate Probabilities ---
        try:
            # Poisson probabilities (alpha is ignored)
            poisson_probs = DistributionConverter.get_bin_probabilities(
                mu=mu, model_type='poisson', bins_config=bins_config
            )

            # Negative Binomial probabilities
            nbinom_probs = DistributionConverter.get_bin_probabilities(
                mu=mu, model_type='nbinom', alpha=alpha, bins_config=bins_config
            )
            
            # --- Create Comparison DataFrame ---
            comparison_df = pd.DataFrame({
                'Bin': list(poisson_probs.keys()),
                'Poisson': list(poisson_probs.values()),
                'NBinom': list(nbinom_probs.values()),
            })
            comparison_df['Diff (NB - P)'] = comparison_df['NBinom'] - comparison_df['Poisson']
            
            # --- Identify Peak Probability ---
            peak_poisson = comparison_df.loc[comparison_df['Poisson'].idxmax()]
            peak_nbinom = comparison_df.loc[comparison_df['NBinom'].idxmax()]

            print(f"Peak Probability (Poisson): {peak_poisson['Poisson']:.2%} in bin '{peak_poisson['Bin']}'")
            print(f"Peak Probability (NBinom):  {peak_nbinom['NBinom']:.2%} in bin '{peak_nbinom['Bin']}'")
            print("\nFull Distribution Comparison:")
            
            # --- Display DataFrame ---
            with pd.option_context('display.max_rows', None, 'display.float_format', '{:.4f}'.format):
                 print(comparison_df.to_string(index=False))

            print(f"\nSum of Probabilities (Poisson): {comparison_df['Poisson'].sum():.4f}")
            print(f"Sum of Probabilities (NBinom):  {comparison_df['NBinom'].sum():.4f}\n")


        except ValueError as e:
            print(f"❌ ERROR in scenario '{scenario['name']}': {e}\n")


if __name__ == "__main__":
    run_stability_test()
