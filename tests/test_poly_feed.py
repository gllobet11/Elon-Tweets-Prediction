import pytest
from unittest.mock import patch, MagicMock

from src.ingestion.poly_feed import PolymarketFeed
from config.bins_definition import MARKET_BINS

# Mock de la respuesta de la API para get_market
MOCK_MARKET_RESP = {"price": "0.55", "closed": False}

# Mock de la respuesta de la API para get_order_book
MOCK_OB_RESP = {
    "bids": [{"price": "0.54", "size": "100"}],
    "asks": [{"price": "0.56", "size": "100"}]
}

# Mock de la respuesta de la API para get_markets
MOCK_MARKETS_LIST_RESP = {
    "data": [
        {
            "slug": "some-other-market",
            "question": "Some other question",
            "outcomes": []
        },
        {
            "slug": "test-elon-market",
            "question": "How many tweets will Elon post?",
            "outcomes": [
                {"title": "140-159", "condition_id": "0x1111"},
                {"title": "160-179", "condition_id": "0x2222"},
                # Faltante a propósito para probar el handling de errores
            ]
        }
    ]
}


@patch('src.ingestion.poly_feed.ClobClient')
def test_get_market_price_success(MockClobClient):
    """Verifica que el precio de mercado se calcula correctamente con un order book válido."""
    # Arrange
    mock_instance = MockClobClient.return_value
    mock_instance.get_market.return_value = MOCK_MARKET_RESP
    mock_instance.get_order_book.return_value = MOCK_OB_RESP
    
    feed = PolymarketFeed()
    
    # Act
    result = feed.get_market_price("0x123")
    
    # Assert
    assert result['status'] == 'ACTIVE'
    assert result['last_price'] == 0.55
    assert result['bid'] == 0.54
    assert result['ask'] == 0.56
    assert result['mid_price'] == pytest.approx(0.55) # (0.54 + 0.56) / 2

@patch('src.ingestion.poly_feed.ClobClient')
def test_get_market_price_no_ob(MockClobClient):
    """Verifica que se usa el last_price si el order book está vacío."""
    # Arrange
    mock_instance = MockClobClient.return_value
    mock_instance.get_market.return_value = MOCK_MARKET_RESP
    mock_instance.get_order_book.return_value = {"bids": [], "asks": []} # OB vacío
    
    feed = PolymarketFeed()
    
    # Act
    result = feed.get_market_price("0x123")
    
    # Assert
    assert result['mid_price'] == 0.55 # Fallback al last_price

@patch('src.ingestion.poly_feed.ClobClient')
def test_fetch_market_ids_automatically_success(MockClobClient):
    """Verifica que los IDs se mapean correctamente desde el slug del mercado."""
    # Arrange
    mock_instance = MockClobClient.return_value
    mock_instance.get_markets.return_value = MOCK_MARKETS_LIST_RESP
    
    feed = PolymarketFeed()
    
    # Copia de los bins para no afectar el import original en otros tests
    test_bins = {k: v.copy() for k, v in MARKET_BINS.items()}
    
    # Act
    updated_bins = feed.fetch_market_ids_automatically(
        market_slug="test-elon-market",
        bins_dict=test_bins
    )
    
    # Assert
    assert updated_bins is not None
    assert updated_bins["140-159"]["id"] == "0x1111"
    assert updated_bins["160-179"]["id"] == "0x2222"
    assert updated_bins["180-199"]["id"] is None # Verifica que los no encontrados son None

@patch('src.ingestion.poly_feed.ClobClient')
def test_fetch_market_ids_market_not_found(MockClobClient):
    """Verifica que devuelve None si no se encuentra el slug."""
    # Arrange
    mock_instance = MockClobClient.return_value
    mock_instance.get_markets.return_value = MOCK_MARKETS_LIST_RESP
    
    feed = PolymarketFeed()
    test_bins = {k: v.copy() for k, v in MARKET_BINS.items()}
    
    # Act
    result = feed.fetch_market_ids_automatically(
        market_slug="non-existent-market",
        bins_dict=test_bins
    )
    
    # Assert
    assert result is None

@patch('src.ingestion.poly_feed.PolymarketFeed.get_market_price')
def test_get_all_bins_prices(mock_get_price):
    """Verifica que se llama a get_market_price para cada bin."""
    # Arrange
    # Hacemos que el mock devuelva un diccionario simple para verificar la llamada
    mock_get_price.side_effect = lambda id: {"market_id": id, "price": 0.5}
    
    feed = PolymarketFeed()
    
    bins_ids_map = {
        "140-159": "0x1111",
        "160-179": "0x2222"
    }
    
    # Act
    snapshot = feed.get_all_bins_prices(bins_ids_map)
    
    # Assert
    assert mock_get_price.call_count == 2
    assert "140-159" in snapshot
    assert "160-179" in snapshot
    assert snapshot["140-159"]["price"] == 0.5
