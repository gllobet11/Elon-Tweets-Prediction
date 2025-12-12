import numpy as np
import pytest

from src.strategy.prob_math import DistributionConverter

# Bins de ejemplo para las pruebas (Label, Low, High)
BINS_CONFIG_EXAMPLE = [
    ("0-199", 0, 199),
    ("200-219", 200, 219),
    ("220-239", 220, 239),
    ("240-259", 240, 259),
    ("260+", 260, float("inf")),
]


def test_basic_nbinom_calculation():
    """
    Test 1: Ensures basic probability calculation works and the sum is close to 1.0.
    - Scenario: Early in the week, no actual tweets yet.
    - Expected: The returned probabilities should be valid and sum to approx. 1.
    """
    mu_remainder = 230.0
    current_actuals = 0
    alpha = 0.05

    probs = DistributionConverter.get_bin_probabilities(
        mu_remainder,
        current_actuals,
        "nbinom",
        alpha,
        BINS_CONFIG_EXAMPLE,
    )

    assert isinstance(probs, dict)
    assert len(probs) == len(BINS_CONFIG_EXAMPLE)
    # The sum of probabilities for all bins should be very close to 1.0
    assert np.isclose(
        sum(probs.values()), 1.0, atol=1e-5
    ), "Probabilities should sum to 1"
    assert probs["220-239"] > 0, "Expected some probability in the central bin"


def test_impossible_bin_due_to_high_actuals():
    """
    Test 2: Checks if bins that are already passed have a probability of 0.
    - Scenario: Actual tweet count has already surpassed the lower bins.
    - Expected: Bins like "0-199" and "200-219" must have a probability of 0.
    """
    mu_remainder = 20.0  # Only a few tweets remaining
    current_actuals = 230  # We are already at 230 tweets
    alpha = 0.05

    probs = DistributionConverter.get_bin_probabilities(
        mu_remainder,
        current_actuals,
        "nbinom",
        alpha,
        BINS_CONFIG_EXAMPLE,
    )

    # Bins "0-199", "200-219" are impossible now. Their high limit is below current_actuals.
    assert probs["0-199"] == 0.0
    assert probs["200-219"] == 0.0
    # The bin we are currently in should have some probability
    assert probs["220-239"] > 0
    # The sum of remaining possibilities should still be close to 1
    assert np.isclose(
        sum(probs.values()), 1.0, atol=1e-5
    ), "Probabilities should still sum to 1"


def test_shift_logic_when_actuals_are_in_a_bin():
    """
    Test 3: Validates the boundary shifting logic.
    - Scenario: Actual count is inside a bin. The calculation for the remainder
      should correctly adjust the lower bound to 0.
    - Expected: The logic should handle the negative `low_rem` correctly.
    """
    mu_remainder = 30.0
    current_actuals = 210  # Current count is within the "200-219" bin
    alpha = 0.05

    probs = DistributionConverter.get_bin_probabilities(
        mu_remainder,
        current_actuals,
        "nbinom",
        alpha,
        BINS_CONFIG_EXAMPLE,
    )

    # "0-199" is impossible
    assert probs["0-199"] == 0.0
    # All other bins should have some probability, as they are still reachable
    assert probs["200-219"] > 0
    assert probs["220-239"] > 0
    assert probs["240-259"] > 0
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-5)


def test_zero_alpha_handling():
    """
    Test 4: Ensures that a very small or zero alpha does not cause division errors.
    - Scenario: Alpha is set to 0.
    - Expected: The function should handle it gracefully by treating it as a very small number.
    """
    mu_remainder = 230.0
    current_actuals = 0
    alpha = 0.0  # Test with zero alpha

    try:
        probs = DistributionConverter.get_bin_probabilities(
            mu_remainder,
            current_actuals,
            "nbinom",
            alpha,
            BINS_CONFIG_EXAMPLE,
        )
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-5)
    except ZeroDivisionError:
        pytest.fail("A zero or very small alpha should not cause a ZeroDivisionError.")


def test_poisson_calculation():
    """
    Test 5: Checks if the Poisson calculation runs and produces a valid distribution.
    - Scenario: Using 'poisson' model type.
    - Expected: Probabilities should be valid and sum to approx. 1.
    """
    mu_remainder = 230.0
    current_actuals = 0

    probs = DistributionConverter.get_bin_probabilities(
        mu_remainder,
        current_actuals,
        model_type="poisson",
        bins_config=BINS_CONFIG_EXAMPLE,
    )

    assert isinstance(probs, dict)
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-5)
    assert probs["220-239"] > 0
