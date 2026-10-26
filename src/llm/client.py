import json

import requests

from src.config import settings


def _generate(prompt: str, system: str | None, options: dict, json_mode: bool = False) -> str:
    """Send one request to Ollama and return the generated text."""
    payload = {"model": settings.llm_model, "prompt": prompt, "stream": False, "options": options}
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    response = requests.post(settings.ollama_url, json=payload, timeout=settings.llm_timeout_s)
    response.raise_for_status()
    return response.json()["response"]


def ask_ollama(prompt: str, system: str | None = None, temperature: float = 0) -> str:
    """Send a prompt (and optional system message) to Ollama; return the text answer."""
    return _generate(prompt, system, options={"temperature": temperature})


def ask_ollama_json(prompt: str, system: str | None = None) -> dict:
    """Ask Ollama for a JSON answer (format=json, temperature 0); return a Python dict."""
    return json.loads(_generate(prompt, system, options={"temperature": 0}, json_mode=True))