"""LLM judge: a second model reads the question, the reference facts and the answer, and grades the meaning.

The keyword grader in run_eval.py only looks for words, so it misses negations ("not free" contains "free"),
other wordings ("still be valid" is not "still valid") and invented claims next to a correct fact.
The judge uses a different, larger model than the agent (JUDGE_MODEL), so it does not grade its own answers.
It also gets the store data (policy and every order with the statuses computed in code), so it can tell a
correct extra detail ("you can return it until November 19") from an invented one.
"""

import json

import requests

from src.config import current_date, load_prompt, settings
from src.llm.client import ask_ollama_json


def format_facts(expected_facts: list[list[str]]) -> str:
    """One line per fact, alternatives joined with " / ", e.g. "- yes / still valid"."""
    return "\n".join("- " + " / ".join(variants) for variants in expected_facts)


def build_prompt(question: str, expected_facts: list[list[str]], answer: str, store_data: str = "") -> str:
    """The user message for the judge: store data, question, reference facts and the answer to grade."""
    return (
        f"Store data:\n{store_data}\n\n"
        f"Customer question:\n{question}\n\n"
        f"Reference facts:\n{format_facts(expected_facts)}\n\n"
        f"Assistant answer:\n{answer}"
    )


def judge_answer(question: str, expected_facts: list[list[str]], answer: str, store_data: str = "") -> dict:
    """Return {"pass": True/False/None, "reason": str}. None means the judge gave no usable verdict."""
    try:
        result = ask_ollama_json(
            build_prompt(question, expected_facts, answer, store_data),
            system=load_prompt("judge_system"),
            model=settings.judge_model,
        )
    except (json.JSONDecodeError, requests.RequestException) as err:
        return {"pass": None, "reason": f"judge error: {err.__class__.__name__}"}

    verdict = result.get("pass")
    if not isinstance(verdict, bool):  # e.g. "yes", "PASS" or missing: do not guess
        return {"pass": None, "reason": f"invalid judge output: {result}"}
    return {"pass": verdict, "reason": str(result.get("reason", ""))}


def build_store_data(policy_text: str, orders: list[dict]) -> str:
    """The truth the judge checks claims against: today's date, the policy and one line per order."""
    lines = [f"Today: {current_date().isoformat()}", "", "Store policy:", policy_text.strip(), "", "Orders:"]
    for order in orders:
        lines.append(
            f"- order {order['order_id']}: {order['customer']}, {order['product']}, ${order['total_usd']:.2f}, "
            f"ordered {order['order_date']}, delivered {order['delivery_date'] or 'not yet'}, "
            f"status {order['status']}. Return: {order['return_status']} Delivery: {order['delivery_status']}"
        )
    return "\n".join(lines)
