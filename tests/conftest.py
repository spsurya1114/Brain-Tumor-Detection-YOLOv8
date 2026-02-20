import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys
import os

# Add project root/src to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app import app, lifespan

@pytest.fixture
def client():
    # Mock the classifier loading to avoid loading the heavy model during tests
    with pytest.MonkeyPatch.context() as m:
        mock_classifier = MagicMock()
        mock_classifier.predict_image.return_value = {"predictions": []}
        
        # Mock the BrainTumorClassifier class
        m.setattr("app.BrainTumorClassifier", MagicMock(return_value=mock_classifier))

        with TestClient(app) as c:
            yield c
