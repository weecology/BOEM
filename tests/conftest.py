# Standard library imports
from pathlib import Path
from unittest.mock import MagicMock

# Third party imports
import pytest
from dotenv import load_dotenv

# Load .env file
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


@pytest.fixture(scope="session")
def comet_logger():
    """Avoid Comet API/network during unit tests."""
    return MagicMock()
