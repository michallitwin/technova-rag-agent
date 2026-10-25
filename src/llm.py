import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ollama(prompt, system=None, model_name="llama3.2", temperature=0):
    """Send a prompt (and optional system message) to Ollama; return the text answer."""
    payload = {"model": model_name, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    if system is not None:
        payload["system"] = system
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]