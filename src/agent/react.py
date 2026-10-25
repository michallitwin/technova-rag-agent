"""ReAct agent loop: the LLM picks tools step by step until it can answer."""

import argparse
import json
import logging
from datetime import date
from string import Template

from src.agent.tools import run_tool
from src.config import load_prompt, settings
from src.llm.client import ask_ollama_json

logger = logging.getLogger(__name__)

# Messages fed back to the model as tool results, so it can correct itself
REPEATED_CALL_ERROR = {"error": "You already made this call. Use the results above."}
INVALID_JSON_ERROR = {"error": "Your reply was not valid JSON. Reply with one JSON object."}
STEP_LIMIT_ANSWER = "Sorry, I couldn't answer within the step limit."


def build_system_prompt() -> str:
    """Fill the system prompt template with today's date and the tool descriptions."""
    template = Template(load_prompt("react_system"))
    return template.substitute(today=date.today().isoformat(), tools=load_prompt("tools"))


def build_prompt(question: str, steps: list[dict]) -> str:
    """Combine the question and the history of tool calls into one prompt."""
    lines = [f"Question: {question}"]
    for i, step in enumerate(steps, start=1):
        lines.append(f"Step {i} call: {json.dumps(step['call'])}")
        lines.append(f"Step {i} result: {json.dumps(step['result'])}")
    lines.append("What is your next step? Reply with one JSON object.")
    return "\n".join(lines)


def run_agent(question: str, max_steps: int = settings.agent_max_steps) -> dict:
    """Run the ReAct loop and return the final answer with the list of steps taken."""
    system = build_system_prompt()
    steps: list[dict] = []

    for step_no in range(1, max_steps + 1):
        try:
            decision = ask_ollama_json(build_prompt(question, steps), system=system)
        except json.JSONDecodeError:
            steps.append({"call": None, "result": INVALID_JSON_ERROR})
            continue

        if "final_answer" in decision:
            logger.info("Step %d: final answer", step_no)
            return {"answer": decision["final_answer"], "steps": steps}

        is_repeat = any(step["call"] == decision for step in steps)
        result = REPEATED_CALL_ERROR if is_repeat else run_tool(decision)
        logger.info("Step %d: %s -> %s", step_no, decision, result)
        steps.append({"call": decision, "result": result})

    logger.warning("Step limit (%d) reached for: %s", max_steps, question)
    return {"answer": STEP_LIMIT_ANSWER, "steps": steps}


def main() -> None:
    """CLI entry point: ask the agent one question and print its answer."""
    parser = argparse.ArgumentParser(description="Ask the TechNova support agent a question.")
    parser.add_argument("question", help="e.g. 'Can I still return order 1042?'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run_agent(args.question)["answer"])


if __name__ == "__main__":
    main()