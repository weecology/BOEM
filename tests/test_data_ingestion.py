import pytest
from data_ingestion import DataIngestion

def test_ingest_data():
    ingestion = DataIngestion()
    data = ingestion.ingest_data()
    assert data is not None
    # Add more specific assertions based on your expected data structure
