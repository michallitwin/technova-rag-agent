import json

import requests

from src.config import settings


def ask_ollama(prompt, system=None, model_name="llama3.2", temperature=0):
    """Send a prompt (and optional system message) to Ollama; return the text answer."""
    payload = {"model": model_name, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    if system is not None:
        payload["system"] = system
    response = requests.post(settings.ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def ask_ollama_json(prompt, system=None, model_name="llama3.2"):
    """Ask Ollama for a JSON answer (format=json, temperature 0); return a Python dict."""
    payload = {"model": model_name, "prompt": prompt, "stream": False,
               "format": "json", "options": {"temperature": 0}}
    if system is not None:
        payload["system"] = system
    response = requests.post(settings.ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    return json.loads(response.json()["response"])