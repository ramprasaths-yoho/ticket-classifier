import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakePredictor:
    def predict(self, texts):
        return [{"category": "Billing", "priority": "High", "confidence": 0.93} for _ in texts]


@pytest.fixture
def client(monkeypatch):
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1
        return FakePredictor()

    monkeypatch.setattr("app.main.load_predictor", fake_loader)
    with TestClient(app) as c:  # context manager runs the startup (lifespan)
        c.loader_calls = calls
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_category_priority_confidence(client):
    r = client.post("/predict", json={"text": "I was charged twice"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"category": "Billing", "priority": "High", "confidence": 0.93}


def test_empty_text_returns_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_whitespace_only_text_returns_422(client):
    assert client.post("/predict", json={"text": "    "}).status_code == 422


def test_missing_text_field_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_model_loaded_once_not_per_request(client):
    for _ in range(3):
        client.post("/predict", json={"text": "hello"})
    assert client.loader_calls["n"] == 1


def test_predict_batch(client):
    r = client.post("/predict-batch", json=[{"text": "a"}, {"text": "b"}])
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_predict_batch_empty_list_422(client):
    assert client.post("/predict-batch", json=[]).status_code == 422
