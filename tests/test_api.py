"""Tests for the Sign Language to Text API."""

import pickle
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Load the label set so we can verify predictions are valid.
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
VALID_LABELS = set(encoder.classes_)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid():
    """A 63-float array should return a known letter and a confidence score."""
    landmarks = [0.0] * 63
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 200
    data = response.json()
    assert data["letter"] in VALID_LABELS
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_wrong_length():
    """An array that isn't exactly 63 elements should be rejected with 422."""
    landmarks = [0.0] * 10
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 422


def test_predict_non_float():
    """Non-numeric values in the array should be rejected with 422."""
    landmarks = ["not", "a", "number"] + ["x"] * 60
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 422
