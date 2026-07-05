import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_get_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "success_rate" in data

def test_get_history():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_endpoint():
    payload = {
        "product_name": "Test Amazon Wireless Mouse",
        "category": "Computers&Accessories|Accessories&Peripherals",
        "discounted_price": 299.0,
        "actual_price": 599.0,
        "discount_percentage": 50.1,
        "rating_count": 1500
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_prob" in data
    assert "predicted_status" in data
    assert "recommendation" in data
    assert 0.0 <= data["prediction_prob"] <= 1.0
