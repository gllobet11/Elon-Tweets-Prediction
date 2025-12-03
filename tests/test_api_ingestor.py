"""
test_api_ingestor.py

Unit tests for the ApiIngestor class.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.api_ingestor import ApiIngestor

# --- Test Fixtures ---


@pytest.fixture
def mock_api_response():
    """Provides a mock API response structure."""
    return {
        "posts": [
            {
                "id": "1731034382607892796",
                "text": "Comedy is legal on this platform.",
                "created_at": 1701548481,  # Unix timestamp
                "possibly_sensitive": False,
            },
            {
                "id": "1731032338523304249",
                "text": "@SpaceX Starship launch 3 maybe a month away",
                "created_at": 1701547996,
                "possibly_sensitive": False,
            },
        ],
        "next_cursor": "1731032338523304249",
    }


@pytest.fixture
def mock_empty_api_response():
    """Provides an empty mock API response."""
    return {"posts": [], "next_cursor": None}


# --- Unit Tests ---


def test_fetch_parses_correctly(httpx_mock, mock_api_response):
    """
    Tests that the fetch method correctly parses a standard API response.
    """
    # Arrange: Mock the API endpoint to return our sample response
    httpx_mock.add_response(
        method="GET",
        url=ApiIngestor.API_ENDPOINT,
        json=mock_api_response,
        status_code=200,
    )
    # Mock the second call (for pagination) to return an empty response
    httpx_mock.add_response(
        method="GET",
        url=ApiIngestor.API_ENDPOINT + "?cursor=1731032338523304249",
        json={"posts": [], "next_cursor": None},
        status_code=200,
    )

    # Act: Call the fetch method
    ingestor = ApiIngestor()
    df = ingestor.fetch()

    # Assert: Check the DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 2

    expected_cols = {"id", "text", "created_at"}
    assert set(df.columns) == expected_cols

    # Check data types
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df["created_at"].dt.tz is not None  # Ensure timezone-aware

    # Check a value
    assert df.iloc[0]["id"] == 1731034382607892796
    assert df.iloc[1]["text"] == "@SpaceX Starship launch 3 maybe a month away"
    expected_dt = datetime(2023, 12, 2, 20, 21, 21, tzinfo=timezone.utc)
    assert df.iloc[0]["created_at"] == expected_dt


def test_fetch_since_id(httpx_mock, mock_api_response):
    """
    Tests that the fetch method correctly uses the `since_id` parameter.
    """
    since_id = 1731000000000000000
    expected_url = f"{ApiIngestor.API_ENDPOINT}?since_id={since_id}"

    # Arrange - Mock the initial request
    httpx_mock.add_response(
        method="GET",
        url=f"{ApiIngestor.API_ENDPOINT}?since_id={since_id}",
        json=mock_api_response,
        status_code=200,
    )

    # Arrange - Mock the subsequent request for the next page (which should be empty)
    # The URL needs to match what the ApiIngestor would generate for the next page
    # It will include both 'cursor' and 'since_id'
    next_cursor = mock_api_response["next_cursor"]
    httpx_mock.add_response(
        method="GET",
        url=f"{ApiIngestor.API_ENDPOINT}?cursor={next_cursor}&since_id={since_id}",
        json={"posts": [], "next_cursor": None},
        status_code=200,
    )

    # Act
    ingestor = ApiIngestor()
    df = ingestor.fetch(since_id=since_id)

    # Assert
    assert not df.empty
    assert len(df) == 2


def test_fetch_empty_response(httpx_mock, mock_empty_api_response):
    """
    Tests that the fetch method returns an empty DataFrame on an empty API response.
    """
    # Arrange
    httpx_mock.add_response(
        method="GET",
        url=ApiIngestor.API_ENDPOINT,
        json=mock_empty_api_response,
        status_code=200,
    )

    # Act
    ingestor = ApiIngestor()
    df = ingestor.fetch()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.empty
