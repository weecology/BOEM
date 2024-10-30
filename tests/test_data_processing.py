import pytest
from src.data_processing import DataProcessing

@pytest.fixture
def data_processing():
    return DataProcessing()

def test_process_data(data_processing):
    # Example test for data processing
    raw_data = "raw data"
    processed_data = data_processing.process_data(raw_data)
    assert processed_data is not None
    # Add more assertions based on expected processed data
