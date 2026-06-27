from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_type2_fast_path_shape(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("MOCK_MODE", "true")
    client = TestClient(app)
    resp = client.post(
        "/predict",
        json={
            "query_id": "T2_0001",
            "type": "type2",
            "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
            "premises": [],
            "options": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["query_id"] == "T2_0001"
    assert data[0]["answer"] == "5"
    assert data[0]["unit"] == "A"
    assert data[0]["premises_used"] == []


def test_type1_mock_shape(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("MOCK_MODE", "true")
    client = TestClient(app)
    resp = client.post(
        "/predict",
        json={
            "query_id": "T1_0001",
            "type": "type1",
            "query": "Is Student A eligible for graduation?",
            "premises": ["A student with >= 120 credits is eligible.", "Student A has 118 credits."],
            "options": ["Yes", "No", "Uncertain"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["query_id"] == "T1_0001"
    assert data[0]["unit"] == ""
    assert data[0]["answer"] in ["Yes", "No", "Uncertain"]
