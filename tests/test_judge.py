"""LLM judge tests: the model is mocked, so they check prompt building, parsing and agreement only."""

import json

from src.eval import judge
from src.eval.validate_judge import agreement

FACTS = [["yes", "still valid"], ["24 months"]]


def test_prompt_contains_question_facts_and_answer():
    prompt = judge.build_prompt("Is it under warranty?", FACTS, "Yes, 24 months.")
    assert "Is it under warranty?" in prompt
    assert "- yes / still valid" in prompt
    assert "- 24 months" in prompt
    assert "Yes, 24 months." in prompt


def test_valid_verdict_is_returned(monkeypatch):
    monkeypatch.setattr(judge, "ask_ollama_json", lambda *a, **k: {"reason": "correct", "pass": True})
    assert judge.judge_answer("q", FACTS, "a") == {"pass": True, "reason": "correct"}


def test_judge_uses_the_judge_model(monkeypatch):
    seen = {}

    def fake(prompt, system=None, model=None):
        seen["model"] = model
        return {"reason": "ok", "pass": False}

    monkeypatch.setattr(judge, "ask_ollama_json", fake)
    judge.judge_answer("q", FACTS, "a")
    assert seen["model"] == judge.settings.judge_model


def test_non_boolean_verdict_gives_no_verdict(monkeypatch):
    monkeypatch.setattr(judge, "ask_ollama_json", lambda *a, **k: {"reason": "ok", "pass": "yes"})
    assert judge.judge_answer("q", FACTS, "a")["pass"] is None


def test_invalid_json_gives_no_verdict(monkeypatch):
    def broken(*a, **k):
        raise json.JSONDecodeError("bad", "doc", 0)

    monkeypatch.setattr(judge, "ask_ollama_json", broken)
    assert judge.judge_answer("q", FACTS, "a")["pass"] is None


def test_agreement_counts_both_kinds_of_mistakes():
    labels = [
        {"human": True, "judge": True},
        {"human": False, "judge": True},   # false pass
        {"human": True, "judge": False},   # false fail
        {"human": False, "judge": None},   # no verdict: left out
    ]
    s = agreement(labels, "judge")
    assert (s["rows"], s["agree"], s["false_pass"], s["false_fail"]) == (3, 1, 1, 1)


def test_store_data_lists_policy_and_computed_statuses(monkeypatch):
    monkeypatch.setattr(judge, "current_date", lambda: __import__("datetime").date(2026, 10, 26))
    order = {"order_id": 1044, "customer": "Piotr Nowak", "product": "Bluetooth speaker", "total_usd": 85.5,
             "order_date": "2026-10-17", "delivery_date": "2026-10-20", "status": "delivered",
             "return_status": "Return possible until 2026-11-19.", "delivery_status": "Free delivery."}
    data = judge.build_store_data("Warranty. 24 months.", [order])
    assert "Today: 2026-10-26" in data
    assert "Warranty. 24 months." in data
    assert "order 1044" in data and "Return possible until 2026-11-19." in data


def test_store_data_goes_into_the_prompt():
    prompt = judge.build_prompt("q", FACTS, "a", store_data="Orders: - order 1044")
    assert prompt.startswith("Store data:\nOrders: - order 1044")
