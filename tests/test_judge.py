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
