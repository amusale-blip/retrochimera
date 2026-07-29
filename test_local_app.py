"""
Local Test Suite for RetroChimera FastAPI Microservice.
Verifies /health and /predict endpoints using FastAPI TestClient in mock mode.
"""

import os
os.environ["MOCK_TRANSLATOR"] = "1"

from fastapi.testclient import TestClient
import app as app_module

# Trigger predictor initialization in mock mode
app_module.initialize_predictor()

client = TestClient(app_module.app)


def test_health():
    """Test GET /health liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict():
    """Test POST /predict retrosynthetic precursor prediction."""
    payload = {
        "instances": [
            {
                "smiles": "CC(=O)OCC",
                "n_best": 3
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "predictions" in res_data
    assert len(res_data["predictions"]) == 1
    assert len(res_data["predictions"][0]["results"]) == 3


if __name__ == "__main__":
    test_health()
    test_predict()
    print("\nAll local FastAPI microservice tests passed successfully!")
