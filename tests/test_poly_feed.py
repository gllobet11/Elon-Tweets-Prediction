import pytest
from unittest.mock import MagicMock, call
import pandas as pd
from datetime import datetime, timezone
import copy

# --- Add project root to path ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.poly_feed import PolymarketFeed
from config.bins_definition import MARKET_BINS

# --- Mocks for ClobClient objects ---
class MockOrder:
    def __init__(self, price, size):
        self.price = price
        self.size = size

class MockOrderBook:
    def __init__(self, bids, asks):
        self.bids = [MockOrder(p, s) for p, s in bids]
        self.asks = [MockOrder(p, s) for p, s in asks]

# --- Test Fixtures ---
@pytest.fixture
def mock_clob_client():
    client = MagicMock()
    client.get_order_book.return_value = MockOrderBook([], [])
    client.get_markets.return_value = {"data": [], "next_cursor": ""}
    return client

@pytest.fixture
def poly_feed(mock_clob_client):
    feed = PolymarketFeed()
    feed.client = mock_clob_client
    feed.valid = True
    return feed

# --- Tests for get_market_valuation ---
def test_valuation_with_yes_liquidity(poly_feed, mock_clob_client):
    yes_token_id, no_token_id = "0xYES", "0xNO"
    mock_clob_client.get_order_book.side_effect = lambda token_id: {
        yes_token_id: MockOrderBook(bids=[("0.60", "100")], asks=[("0.62", "100")])
    }.get(token_id, MockOrderBook([], []))
    result = poly_feed.get_market_valuation(yes_token_id, no_token_id)
    assert result['status'] == 'ACTIVE_YES'
    assert result['mid_price'] == pytest.approx(0.61)

def test_valuation_inferred_from_no_liquidity(poly_feed, mock_clob_client):
    yes_token_id, no_token_id = "0xYES", "0xNO"
    mock_clob_client.get_order_book.side_effect = lambda token_id: {
        no_token_id: MockOrderBook(bids=[("0.35", "100")], asks=[("0.38", "100")])
    }.get(token_id, MockOrderBook([], []))
    result = poly_feed.get_market_valuation(yes_token_id, no_token_id)
    assert result['status'] == 'ACTIVE_NO'
    assert result['mid_price'] == pytest.approx(0.635)

def test_valuation_with_no_liquidity(poly_feed, mock_clob_client):
    result = poly_feed.get_market_valuation("0xYES", "0xNO", entry_price_fallback=0.5)
    assert result['status'] == 'NO_LIQUIDITY'
    assert result['mid_price'] == 0.5

# --- Tests for get_market_dates ---
@pytest.mark.parametrize("description, expected_start, expected_end", [
    ("from December 2, 2025 12:00 PM ET to December 9, 2025 12:00 PM ET", "2025-12-02 17:00:00", "2025-12-09 17:00:00"),
    ("from December 30, 2024 12:00 PM ET to January 6, 2025 12:00 PM ET", "2024-12-30 17:00:00", "2025-01-06 17:00:00"),
    ("from December 23 12:00 PM ET to December 30, 2024 12:00 PM ET", "2024-12-23 17:00:00", "2024-12-30 17:00:00")
])
def test_get_market_dates(poly_feed, description, expected_start, expected_end):
    start_date, end_date = poly_feed.get_market_dates(description)
    assert start_date == pd.to_datetime(expected_start, utc=True)
    assert end_date == pd.to_datetime(expected_end, utc=True)

def test_get_market_dates_bad_string(poly_feed):
    start_date, end_date = poly_feed.get_market_dates("this has no dates")
    assert start_date is None
    assert end_date is None

# --- Test for fetch_market_ids_automatically ---
def test_fetch_market_ids_automatically(poly_feed, mock_clob_client):
    """
    Tests that the function can find a market by keywords and correctly
    map the token IDs for its corresponding bins.
    """
    # Arrange
    keywords = ["elon musk", "tweets", "december 2"]
    
    # Deep copy to avoid modifying the original dict during tests
    test_bins_dict = copy.deepcopy(MARKET_BINS)
    
    mock_markets_response = {
        "data": [
            {
                "question": "Will Trump win?", "active": True, "tokens": []
            },
            {
                "question": "How many tweets will Elon Musk post from Nov 25 to December 2?",
                "active": True,
                "tokens": [
                    {"token_id": "0xYES_300", "outcome": "Yes"},
                    {"token_id": "0xNO_300", "outcome": "No"},
                ],
                # The question contains the bin label that links it to the MARKET_BINS dict
                "condition_id": "0xCOND_ID_300_319" # This is usually how Polymarket links, but let's assume question parsing
            },
             {
                "question": "Bin 320-339: How many tweets will Elon Musk post from Nov 25 to December 2?",
                "active": True,
                "tokens": [
                    {"token_id": "0xYES_320", "outcome": "Yes"},
                    {"token_id": "0xNO_320", "outcome": "No"},
                ]
            }
        ],
        "next_cursor": ""
    }
    
    # To properly test, we need to extract the bin from the question string
    # Let's refine the mock to be more realistic. The question itself contains the bin.
    mock_markets_response['data'][1]['question'] = "300-319: How many tweets will Elon Musk post from Nov 25 to December 2?"
    
    mock_clob_client.get_markets.return_value = mock_markets_response

    # Act
    updated_bins = poly_feed.fetch_market_ids_automatically(keywords, test_bins_dict)

    # Assert
    # 1. Check that the correct number of get_markets calls were made (only 1, no pagination needed)
    mock_clob_client.get_markets.assert_called_once()
    
    # 2. Check that the returned dict is updated
    assert updated_bins["300-319"]["id_yes"] == "0xYES_300"
    assert updated_bins["300-319"]["id_no"] == "0xNO_300"
    assert updated_bins["320-339"]["id_yes"] == "0xYES_320"
    assert updated_bins["320-339"]["id_no"] == "0xNO_320"
    
    # 3. Check that a bin not in the mock response remains un-updated
    assert updated_bins["280-299"]["id_yes"] is None
