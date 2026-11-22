"""HTTP API for the TechNova support agent (FastAPI)."""

import json
import logging

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.agent.react import run_agent
from src.api.schemas import AgentSummary, AskRequest, AskResponse, EvalSummaryResponse, HealthResponse
from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="TechNova Support Agent", version="0.1.0")


@app.exception_handler(Exception)
def unexpected_error(request: Request, err: Exception) -> JSONResponse:
    """Last line of defence: details go to the server log, the client gets a generic message."""
    logger.error("Unexpected error on %s", request.url.path, exc_info=err)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check for Docker and monitoring."""
    return HealthResponse()


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest, x_customer: str | None = Header(default=None)) -> AskResponse:
    """Ask the ReAct agent one question. The logged-in customer comes from the X-Customer header."""
    try:
        result = run_agent(request.question, customer=x_customer)
    except requests.RequestException as err:
        raise HTTPException(status_code=503, detail="The language model is not available.") from err
    return AskResponse(**result)


@app.get("/api/eval/latest", response_model=EvalSummaryResponse)
def latest_eval() -> EvalSummaryResponse:
    """Summary of the most recent evaluation report."""
    # Golden-set reports only: adversarial_*.json has a different format and sorts after digits
    reports = sorted(settings.eval_results_dir.glob("[0-9]*.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="No evaluation report found.")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    agents = {}
    for name, data in report["agents"].items():
        summary = data["summary"]
        agents[name] = AgentSummary(accuracy=summary["accuracy"], avg_steps=summary["avg_steps"])

    return EvalSummaryResponse(
        report=reports[-1].name, model=report["model"], hit_at_k=report["hit_at_k"], agents=agents
    )
