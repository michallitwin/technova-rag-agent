import argparse
import json
import logging
from string import Template

from src.agent.answer import answer_from_results
from src.agent.tools import run_tool
from src.config import current_date, load_prompt, settings
from src.llm.client import ask_ollama_json

logger = logging.getLogger(__name__)

INVALID_JSON_ERROR = {"error": "Your reply was not valid JSON. Reply with one JSON object."}


def build_system_prompt() -> str:
    """Fill the system prompt template with today's date and the tool descriptions."""
    template = Template(load_prompt("react_system"))
    return template.substitute(today=current_date().isoformat(), tools=load_prompt("tools"))


def build_prompt(question: str, steps: list[dict]) -> str:
    """Combine the question and the history of tool calls into one prompt."""
    lines = [f"Question: {question}"]
    for i, step in enumerate(steps, start=1):
        lines.append(f"Step {i} call: {json.dumps(step['call'])}")
        lines.append(f"Step {i} result: {json.dumps(step['result'])}")
    lines.append("What is your next step? Reply with one JSON object.")
    return "\n".join(lines)


def describe_call(call: dict) -> str:
    """Short, readable form of a tool call, e.g. get_order(order_id=1042)."""
    args = ", ".join(f"{key}={value!r}" for key, value in call.get("args", {}).items())
    return f"{call.get('tool')}({args})"


def describe_result(result: dict) -> str:
    """Short summary of a tool result for logs (full result goes to DEBUG)."""
    if "error" in result:
        return f"error: {result['error']}"
    if "chunks" in result:
        return f"{len(result['chunks'])} chunks"
    if "days_since_delivery" in result:
        window = "open" if result["return_window_open"] else "closed"
        return f"{result['status']}, {result['days_since_delivery']} days ago, return window {window}"
    return result.get("status", "ok")


def run_agent(question: str, max_steps: int = settings.agent_max_steps) -> dict:
    """Run the ReAct loop; if the model stalls, force a final answer from the results."""
    system = build_system_prompt()
    steps: list[dict] = []

    for step_no in range(1, max_steps + 1):
        try:
            decision = ask_ollama_json(build_prompt(question, steps), system=system)
        except json.JSONDecodeError:
            logger.info("Step %d: invalid JSON from model", step_no)
            steps.append({"call": None, "result": INVALID_JSON_ERROR})
            continue

        if "final_answer" in decision:
            logger.info("Step %d: final answer", step_no)
            return {"answer": decision["final_answer"], "steps": steps, "forced": False}

        if any(step["call"] == decision for step in steps):
            logger.info("Step %d: repeated call -> forcing final answer", step_no)
            break

        result = run_tool(decision)
        logger.info("Step %d: %s -> %s", step_no, describe_call(decision), describe_result(result))
        logger.debug("Step %d full result: %s", step_no, result)
        steps.append({"call": decision, "result": result})
    else:
        logger.warning("Step limit (%d) reached -> forcing final answer", max_steps)

    return {"answer": answer_from_results(question, steps), "steps": steps, "forced": True}


def main() -> None:
    """CLI entry point: ask the agent one question and print its answer."""
    parser = argparse.ArgumentParser(description="Ask the TechNova support agent a question.")
    parser.add_argument("question", help="e.g. 'Can I still return order 1042?'")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full tool results")
    args = parser.parse_args()

    # Libraries log only warnings; our logger shows the steps (or everything with -v)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print(f"\nAnswer: {run_agent(args.question)['answer']}")


if __name__ == "__main__":
    main()