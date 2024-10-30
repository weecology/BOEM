import pytest
from src.data_ingestion import DataIngestion

@pytest.fixture
def data_ingestion():
    return DataIngestion()

def test_ingest_data(data_ingestion):
    # Example test for data ingestion
    data = data_ingestion.ingest_data()
    assert data is not None
    # Add more assertions based on expected data structure

if __name__ == '__main__':
    pytest.main()
