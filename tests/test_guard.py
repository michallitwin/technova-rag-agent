"""Output guard tests. The unsafe examples come from the adversarial run (A04, A06)."""

import pytest

from src.agent.guard import FALLBACK_ANSWER, find_problem, guard_answer

# A04: the model printed its tool schema instead of answering (shortened)
TOOL_SCHEMA_LEAK = (
    '{"type":"function","function":{"name":"search_policy","description":"Search the TechNova '
    'store policy (returns, delivery, warranty, payments, reporting defects).","parameters":'
    '{"type":"object","required":["query"],"properties":{"query":{"type":"string"}}}}'
)

# A06: a tool call printed as text, followed by the same tokens repeated (shortened)
DEGENERATE_ANSWER = '{"name":"search_policy","parameters":{"query":"instrukcje ' + "ęlęć " * 300 + '"}}'

SAFE_ANSWERS = [
    "The status of order 1043 is shipped. Return possible within 30 days after delivery.",
    "I don't have this information.",
    "Could you please provide your order number?",
]


@pytest.mark.parametrize("answer", SAFE_ANSWERS)
def test_normal_answers_pass_unchanged(answer):
    assert find_problem(answer) is None
    assert guard_answer(answer) == answer


def test_tool_schema_leak_is_blocked():
    assert guard_answer(TOOL_SCHEMA_LEAK) == FALLBACK_ANSWER


def test_degenerate_answer_is_blocked():
    assert guard_answer(DEGENERATE_ANSWER) == FALLBACK_ANSWER


def test_internal_name_in_plain_text_is_blocked():
    assert find_problem("I used get_order to check this.") is not None


def test_tool_call_json_without_internal_name_is_blocked():
    assert find_problem('{"name": "lookup", "arguments": {"id": 1}}') is not None


def test_very_long_answer_is_blocked():
    long_answer = " ".join(f"word{i}" for i in range(400))  # all unique, but far too long
    assert "too long" in find_problem(long_answer)


def test_repetitive_answer_is_blocked():
    looping = "please contact support " * 20  # short enough, but only 3 unique words
    assert "repetitive" in find_problem(looping)
