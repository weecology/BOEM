import pytest
from src.data_ingestion import DataIngestion
from src.monitoring import Monitoring

@pytest.fixture
def data_ingestion():
    return DataIngestion()

def test_ingest_data(data_ingestion):
    data = data_ingestion.ingest_data()
    assert data is not None
    # Add more specific assertions based on your expected data structure

if __name__ == '__main__':
    pytest.main()
