from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from engine.config import Settings
from engine.orchestration.daily_runner import DailyRunner
from service.server import main as server_main


def test_daily_pipeline_api_runs_in_degraded_mode(tmp_path) -> None:
    settings = Settings.load(Path(__file__).resolve().parents[1])
    settings.storage_root = tmp_path
    settings.market_storage = tmp_path / "market"
    settings.text_storage = tmp_path / "text"
    settings.disclosure_storage = settings.text_storage / "disclosures"
    settings.disclosure_pdf_storage = settings.disclosure_storage / "pdf"
    settings.disclosure_text_storage = settings.disclosure_storage / "text"
    settings.artifact_storage = tmp_path / "artifacts"
    settings.db_path = tmp_path / "state.db"
    for path in (
        settings.storage_root,
        settings.market_storage,
        settings.text_storage,
        settings.disclosure_storage,
        settings.disclosure_pdf_storage,
        settings.disclosure_text_storage,
        settings.artifact_storage,
    ):
        path.mkdir(parents=True, exist_ok=True)
    settings.market_provider = "sample"
    settings.text_provider = "derived"
    settings.primary_provider.api_key = ""
    settings.fallback_provider.api_key = ""
    server_main.runner = DailyRunner(settings)
    client = TestClient(server_main.app)

    response = client.post("/api/sim/run-daily", json={})
    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["status"] == "degraded"

    dashboard = client.get("/api/dashboard/summary")
    assert dashboard.status_code == 200
    assert "portfolio" in dashboard.json()
