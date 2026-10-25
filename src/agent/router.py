from string import Template

from src.config import load_prompt
from src.llm.client import ask_ollama_json


def choose_tool(question: str) -> dict:
    """Ask the LLM which tool to use for the question; return {"tool": ..., "args": {...}}."""
    template = Template(load_prompt("router_system"))
    system = template.substitute(tools=load_prompt("tools"))
    return ask_ollama_json(question, system=system)