from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_model_info_reflects_settings(monkeypatch):
    monkeypatch.setattr(settings, "decision_threshold", 0.37)
    monkeypatch.setattr(settings, "aggregation", "topk")
    monkeypatch.setattr(settings, "detect_faces", True)
    body = client.get("/api/v1/model").json()
    assert body["decision_threshold"] == 0.37
    assert body["aggregation"] == "topk"
    assert body["detect_faces"] is True
    assert body["predictor"]
    assert set(body) >= {"num_frames", "frame_size", "normalize_mode", "topk_fraction"}
