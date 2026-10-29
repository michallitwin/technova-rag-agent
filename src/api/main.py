"""HTTP API for the TechNova support agent (FastAPI)."""

import json

import requests
from fastapi import FastAPI, HTTPException
from src.agent.react import run_agent
from src.api.schemas import AskRequest, AskResponse, EvalSummaryResponse, HealthResponse
from src.config import settings

app = FastAPI(title="TechNova Support Agent", version="0.1.0")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check for Docker and monitoring."""
    return HealthResponse()


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Ask the ReAct agent one question."""
    try:
        result = run_agent(request.question)
    except requests.RequestException as err:
        raise HTTPException(status_code=503, detail="The language model is not available.") from err
    return AskResponse(**result)


@app.get("/api/eval/latest", response_model=EvalSummaryResponse)
def latest_eval() -> EvalSummaryResponse:
    """Summary of the most recent evaluation report."""
    reports = sorted(settings.eval_results_dir.glob("*.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="No evaluation report found.")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    react = report["agents"]["react"]["summary"]
    return EvalSummaryResponse(
        report=reports[-1].name,
        model=report["model"],
        hit_at_k=report["hit_at_k"],
        accuracy=react["accuracy"],
        avg_steps=react["avg_steps"],
    )