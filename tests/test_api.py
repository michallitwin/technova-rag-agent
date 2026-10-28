"""API tests: the agent is mocked, so no LLM or Ollama is needed."""

from dataclasses import replace

import pytest
import requests
from fastapi.testclient import TestClient

from src.api import main

FAKE_RESULT = {
    "answer": "Yes, you can return it until 2026-11-09.",
    "steps": [
        {
            "call": {"tool": "get_order", "args": {"order_id": 1042}},
            "result": {"status": "delivered"},
        }
    ],
    "forced": False,
}


@pytest.fixture
def client() -> TestClient:
    """HTTP client that talks to the app in memory (no real server)."""
    return TestClient(main.app)


@pytest.fixture
def fake_agent(monkeypatch):
    """Replace the real agent with a fake that returns FAKE_RESULT instantly."""
    monkeypatch.setattr(main, "run_agent", lambda question: FAKE_RESULT)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer(client, fake_agent):
    response = client.post(
        "/api/ask", json={"question": "Can I still return order 1042?"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == FAKE_RESULT["answer"]


def test_ask_rejects_blank_question(client):
    response = client.post("/api/ask", json={"question": "     "})
    assert response.status_code == 422


def test_ask_returns_503_when_llm_is_down(client, monkeypatch):
    def broken_agent(question):
        raise requests.ConnectionError("Ollama is not running")

    monkeypatch.setattr(main, "run_agent", broken_agent)
    response = client.post(
        "/api/ask", json={"question": "Can I still return order 1042?"}
    )
    assert response.status_code == 503


def test_eval_latest_404_without_reports(client, monkeypatch, tmp_path):
    empty_settings = replace(main.settings, eval_results_dir=tmp_path)  # empty folder
    monkeypatch.setattr(main, "settings", empty_settings)
    response = client.get("/api/eval/latest")
    assert response.status_code == 404
