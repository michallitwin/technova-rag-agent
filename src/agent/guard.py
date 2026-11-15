"""Output guard: check the agent's answer before it reaches the customer.

The adversarial test set showed two failures that a small local model can produce:
- A04: the model printed its own tool schema instead of answering (internal details leak),
- A06: the model printed a tool call as text and then repeated the same tokens until the
  length limit (degenerate output).

The checks below are simple and deterministic. They do not judge whether the answer is
correct; they only block answers that must never be shown to a customer.
"""

import logging
import re

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "Sorry, I can't answer this. Please contact our support team."

# Internal names the customer should never see
INTERNAL_NAMES = ("get_order", "search_policy", "return_status", "delivery_status")

# Keys that appear when the model prints a tool call or a tool schema as text
TOOL_JSON_KEYS = ('"name"', '"parameters"', '"function"', '"arguments"')

# Golden-set answers are under 400 characters, so 1000 leaves plenty of room
MAX_ANSWER_CHARS = 1000

# Repetition: share of unique words. Normal answers score above 0.5,
# a model stuck in a loop scores far lower.
MIN_WORDS_FOR_REPETITION_CHECK = 40
MIN_UNIQUE_WORD_RATIO = 0.3


def find_problem(answer: str) -> str | None:
    """Return the reason an answer is unsafe to show, or None if it looks fine."""
    text = answer.strip()
    lowered = text.lower()

    for name in INTERNAL_NAMES:
        if name in lowered:
            return f"internal name in answer: {name}"

    if text.startswith("{") and any(key in text for key in TOOL_JSON_KEYS):
        return "tool call or tool schema printed as text"

    if len(text) > MAX_ANSWER_CHARS:
        return f"answer too long: {len(text)} characters"

    words = re.findall(r"\w+", lowered)
    if len(words) >= MIN_WORDS_FOR_REPETITION_CHECK:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            return f"repetitive answer: {unique_ratio:.0%} unique words"

    return None


def guard_answer(answer: str) -> str:
    """Return the answer if it is safe to show, otherwise a fixed fallback message."""
    problem = find_problem(answer)
    if problem is None:
        return answer
    logger.warning("Answer blocked by output guard (%s)", problem)
    return FALLBACK_ANSWER
