"""
test_prob_math.py

Unit tests for the probability calculation utilities.
"""

import pytest
import os
import sys

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategy.prob_math import DistributionConverter

# --- Test Fixtures ---

@pytest.fixture
def sample_bins_config():
    """A sample bin configuration for testing."""
    return [
        ("0-9", 0, 9),
        ("10-19", 10, 19),
        ("20-29", 20, 29),
        ("30+", 30, float('inf'))
    ]

# --- Unit Tests ---

def test_get_bin_probabilities_nbinom(sample_bins_config):
    """
    Tests the basic functionality of get_bin_probabilities with nbinom.
    """
    # --- Arrange ---
    mu_remainder = 15.0
    current_actuals = 5
    
    # --- Act ---
    probabilities = DistributionConverter.get_bin_probabilities(
        mu_remainder=mu_remainder,
        current_actuals=current_actuals,
        model_type='nbinom',
        alpha=0.2,
        bins_config=sample_bins_config
    )

    # --- Assert ---
    assert isinstance(probabilities, dict)
    assert set(probabilities.keys()) == {"0-9", "10-19", "20-29", "30+"}
    
    # Check that probabilities are valid
    total_prob = 0
    for bin_label, prob in probabilities.items():
        assert 0.0 <= prob <= 1.0
        total_prob += prob
        
    # Check that the sum is close to 1.0
    assert total_prob == pytest.approx(1.0, abs=1e-3)

def test_kelly_bet_calculation():
    """
    Tests the kelly bet sizing logic.
    """
    # --- Arrange ---
    my_prob = 0.60  # We believe there's a 60% chance of winning
    market_price = 0.40 # Market implies a 40% chance
    bankroll = 1000
    
    # --- Act ---
    bet_size = DistributionConverter.calculate_kelly_bet(
        my_prob=my_prob,
        market_price=market_price,
        bankroll=bankroll,
        kelly_fraction=0.5 # Use 50% of full Kelly for safety
    )
    
    # --- Assert ---
    # b (odds) = (1 / 0.40) - 1 = 1.5
    # f_star = (0.60 * (1.5 + 1) - 1) / 1.5 = (0.60 * 2.5 - 1) / 1.5 = (1.5 - 1) / 1.5 = 0.5 / 1.5 = 0.333
    # f_safe = 0.333 * 0.5 = 0.1666
    # bet_size = 1000 * 0.1666 = 166.66
    assert bet_size == pytest.approx(166.66, abs=0.1)

def test_kelly_bet_no_edge():
    """
    Tests that bet size is zero when there is no edge.
    """
    bet_size = DistributionConverter.calculate_kelly_bet(
        my_prob=0.40,
        market_price=0.50,
        bankroll=1000
    )
    assert bet_size == 0.0
