from fastapi.testclient import TestClient
from src.api.app import app

def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_demo():
    c = TestClient(app)
    payload = {"rows":[{"load":100.0,"temp":15.0},{"load":110.0,"temp":16.0}]}
    r = c.post("/predict", json=payload)
    assert r.status_code == 200
    assert "predictions" in r.json()
    assert len(r.json()["predictions"]) == 2
