import pytest
from monitoring import Monitoring

def test_start_monitoring():
    monitoring = Monitoring()
    deployed_model = "Sample deployed model"  # Replace with appropriate test model
    result = monitoring.start_monitoring(deployed_model)
    assert result is not None
    # Add more specific assertions based on your expected monitoring results
