# Configure Python environment for test
# ---- Basics ----
import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import glob
import random
from pathlib import Path
from starlette.testclient import TestClient

# ---- cell-instance-segmentation-_API modules ----
from api.main import app
from unet.model import get_model
from mocks import MockModel


# Call the mock model
def get_model_override():
    model = MockModel()
    return model

# Dependency injection
app.dependency_overrides[get_model] = get_model_override


@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def image_path():
    return Path(os.path.join(os.path.dirname(__file__), 'api', 'images')) 