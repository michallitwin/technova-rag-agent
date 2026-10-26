import json

from src.config import load_prompt
from src.llm.client import ask_ollama


def answer_from_results(question: str, steps: list[dict]) -> str:
    """Ask the LLM to answer the question using only the collected tool results."""
    prompt = (f"Tool results:\n{json.dumps(steps, indent=2)}\n\n"
              f"Question: {question}")
    return ask_ollama(prompt, system=load_prompt("answer_system"))