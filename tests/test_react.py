"""ReAct loop tests: the model is faked, so no Ollama is needed.

The model's reply is untrusted input. Valid JSON of the wrong shape must not crash the loop
(and through it the API with HTTP 500): it is treated like any other invalid reply.
"""

import sqlite3
from dataclasses import replace
from itertools import chain, repeat

import pytest

from src.agent import react, tools


@pytest.fixture(autouse=True)
def isolated_agent(tmp_path, monkeypatch):
    """Tiny orders database and no real LLM calls (neither the loop nor the forced answer)."""
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT, product TEXT, "
            "total_usd REAL, order_date TEXT, delivery_date TEXT, status TEXT)"
        )
        conn.execute(
            "INSERT INTO orders VALUES (1042, 'John Smith', 'Smartwatch', 129.0, '2026-10-07', '2026-10-10', 'delivered')"
        )
    monkeypatch.setattr(tools, "settings", replace(tools.settings, db_path=db_path))
    monkeypatch.setattr(react, "answer_from_results", lambda question, steps: "Summary from the tool results.")


def fake_model(monkeypatch, replies, then=None):
    """The model returns the given replies in order, then keeps returning `then` (if given)."""
    source = chain(replies, repeat(then)) if then is not None else iter(replies)
    monkeypatch.setattr(react, "ask_ollama_json", lambda prompt, system=None, model=None: next(source))


def test_valid_tool_call_then_final_answer(monkeypatch):
    # Control test: the normal path must keep working
    fake_model(
        monkeypatch,
        [{"tool": "get_order", "args": {"order_id": 1042}}, {"final_answer": "Yes, you can return it."}],
    )
    result = react.run_agent("Can I return order 1042?", customer="John Smith")
    assert result["forced"] is False
    assert result["answer"] == "Yes, you can return it."
    assert len(result["steps"]) == 1


BAD_REPLIES = [
    pytest.param([1, 2], id="list-instead-of-object"),
    pytest.param(5, id="number-instead-of-object"),
    pytest.param({"foo": 1}, id="object-without-tool"),
    pytest.param({"final_answer": 5}, id="final-answer-is-a-number"),
    pytest.param({"final_answer": None}, id="final-answer-is-null"),
    pytest.param({"tool": ["get_order"]}, id="tool-name-is-a-list"),
    pytest.param({"tool": "get_order", "args": None}, id="args-is-null"),
    pytest.param({"tool": "get_order", "args": ["1042"]}, id="args-is-a-list"),
    pytest.param({"tool": "get_order", "args": {"order_id": 99999999999999999999}}, id="huge-order-id"),
    pytest.param({"tool": "get_order", "args": {"order_id": -5}}, id="negative-order-id"),
]


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
def test_bad_model_reply_does_not_crash_the_loop(monkeypatch, bad_reply):
    fake_model(monkeypatch, [], then=bad_reply)   # the model keeps sending the same bad reply
    result = react.run_agent("Can I return order 1042?", customer="John Smith")
    assert isinstance(result["answer"], str)
    assert result["forced"] is True               # the loop gave up and answered from the results
