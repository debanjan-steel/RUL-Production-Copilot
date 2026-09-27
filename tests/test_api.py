"""Integration Tests for FastAPI Serving and OpsCopilot Safety Endpoints."""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Checks root endpoint metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Turbofan RUL OpsCopilot API"
    assert data["status"] == "online"


def test_health_probe():
    """Checks health monitoring probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True


def test_predict_single_cycle():
    """Verifies that single cycle telemetry is auto-padded and returns RUL."""
    payload = {
        "engine_id": 42,
        "cycle": 105,
        "sensor_readings": [
            642.5, 1588.0, 1405.0, 553.0, 2388.0, 9050.0,
            47.5, 521.0, 2388.0, 8130.0, 8.4, 392.0, 38.8, 23.3
        ],
        "mc_samples": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["engine_id"] == 42
    assert "predicted_rul" in res_data
    assert "uncertainty_std" in res_data
    assert res_data["status"] in ["CONFIDENT", "ABSTAIN_ESCALATE"]
    assert len(res_data["confidence_interval_95"]) == 2


def test_predict_sequence_window():
    """Verifies temporal sequence window prediction."""
    seq = [[640.0 + i * 0.1 + j * 0.05 for j in range(14)] for i in range(30)]
    payload = {
        "engine_id": 7,
        "cycle": 150,
        "sequence": seq,
        "mc_samples": 25,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 7
    assert data["predicted_rul"] >= 0.0
    assert len(data["top_degrading_sensors"]) > 0
