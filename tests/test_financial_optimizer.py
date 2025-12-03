"""
test_financial_optimizer.py

Unit tests for the financial optimizer utilities.
"""

import os
import sys
from datetime import date

import pytest

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategy.financial_optimizer import get_last_complete_friday

# --- Unit Tests ---


# Parameterize the test to cover different scenarios
@pytest.mark.parametrize(
    "input_date, expected_friday",
    [
        # Case 1: Input date is a Saturday. Last complete week started on the previous Friday.
        (date(2025, 12, 6), date(2025, 11, 28)),
        # Case 2: Input date is a Friday. The window must END on this day, so the last COMPLETE week
        # started the Friday before.
        (date(2025, 12, 5), date(2025, 11, 28)),
        # Case 3: Input date is a Thursday. A full week can't be formed, so we go to the previous Friday.
        (date(2025, 12, 4), date(2025, 11, 21)),
        # Case 4: A mid-week case
        (date(2025, 11, 20), date(2025, 11, 7)),
    ],
)
def test_get_last_complete_friday(input_date, expected_friday):
    """
    Tests the get_last_complete_friday function to ensure it correctly identifies
    the start of the last full week for backtesting.
    """
    # --- Act ---
    result = get_last_complete_friday(input_date)

    # --- Assert ---
    # The function returns a datetime object, so we compare dates.
    assert result.date() == expected_friday
