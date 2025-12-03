from datetime import timezone

import pandas as pd
import pytest

from src.ingestion.xtracker_feed import XTrackerIngestor


@pytest.fixture
def create_dummy_csv(tmp_path):
    """Crea un archivo CSV de prueba para los tests."""
    csv_content = """Date,Text,Link
2025-10-25 10:00:00,First tweet,http://example.com/1
2025-10-26 12:00:00,Second tweet,http://example.com/2
2025-10-27 14:00:00,Third tweet,http://example.com/3
"""
    csv_path = tmp_path / "dummy_tweets.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


def test_load_and_clean_data_success(create_dummy_csv):
    """Verifica que los datos se cargan y limpian correctamente."""
    # Arrange
    ingestor = XTrackerIngestor(data_path=create_dummy_csv)

    # Act
    df = ingestor.load_and_clean_data()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert "created_at" in df.columns
    assert "text" in df.columns  # Verifica la limpieza de nombres
    assert df["created_at"].dt.tz is not None
    assert df["created_at"].dt.tz == timezone.utc
    assert len(df) == 3


def test_load_data_file_not_found():
    """Verifica que se lanza un error si el archivo no existe."""
    # Arrange
    ingestor = XTrackerIngestor(data_path="non_existent_file.csv")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        ingestor.load_and_clean_data()


def test_get_current_week_count(create_dummy_csv):
    """Verifica el conteo de tweets desde una fecha de inicio."""
    # Arrange
    ingestor = XTrackerIngestor(data_path=create_dummy_csv)

    # Act: Contar desde el segundo tweet
    result = ingestor.get_current_week_count(start_date_iso="2025-10-26 00:00:00")

    # Assert
    assert result["current_count"] == 2
    assert result["last_update"].strftime("%Y-%m-%d %H:%M:%S") == "2025-10-27 14:00:00"


def test_get_current_week_count_no_tweets(create_dummy_csv):
    """Verifica el caso donde no hay tweets nuevos desde la fecha de inicio."""
    # Arrange
    ingestor = XTrackerIngestor(data_path=create_dummy_csv)

    # Act: Contar desde una fecha futura
    result = ingestor.get_current_week_count(start_date_iso="2025-11-01 00:00:00")

    # Assert
    assert result["current_count"] == 0
    # El 'last_update' debería ser la fecha de inicio del mercado si no hay tweets
    assert result["last_update"].strftime("%Y-%m-%d %H:%M:%S") == "2025-11-01 00:00:00"
