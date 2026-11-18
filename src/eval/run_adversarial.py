"""Run the adversarial test set against the agents and save the answers for review.

Each prompt tries to break the agent (prompt injection, prompt leak, other customers'
data, invented discounts, off-topic requests). The script does not decide pass/fail
on its own: it flags answers that contain a forbidden phrase and saves everything,
so a human (or later an LLM judge) gives the final verdict.

Usage:
    uv run python -m src.eval.run_adversarial                 # both agents
    uv run python -m src.eval.run_adversarial --agents graph  # one agent
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.agent.graph import run_graph_agent as graph_agent
from src.agent.react import run_agent as react_agent
from src.config import settings

logger = logging.getLogger(__name__)

ADVERSARIAL_SET_PATH = Path("data/eval/adversarial_set.json")
AGENTS = {"react": react_agent, "graph": graph_agent}


def load_adversarial_set() -> list[dict]:
    """Read the attack prompts with their expected behavior and forbidden phrases."""
    return json.loads(ADVERSARIAL_SET_PATH.read_text(encoding="utf-8"))


def find_forbidden(answer: str, forbidden: list[str]) -> list[str]:
    """Return the forbidden phrases that appear in the answer (case-insensitive)."""
    text = answer.lower()
    return [phrase for phrase in forbidden if phrase.lower() in text]


def tool_calls(steps: list[dict]) -> list[str]:
    """Readable list of tool calls, e.g. ["get_order({'order_id': 1042})"]."""
    calls = []
    for step in steps:
        call = step.get("call")
        if call and "tool" in call:
            calls.append(f"{call['tool']}({call.get('args', {})})")
    return calls


def run_item(agent, item: dict) -> dict:
    """Ask one agent one attack prompt and collect what it did and said."""
    start = time.perf_counter()
    result = agent(item["prompt"], customer=item.get("customer"))  # new conversation, logged-in customer
    latency = time.perf_counter() - start

    found = find_forbidden(result["answer"], item["forbidden"])
    return {
        "id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "expected_behavior": item["expected_behavior"],
        "answer": result["answer"],
        "tool_calls": tool_calls(result["steps"]),
        "forbidden_found": found,
        "auto_flag": "FAIL" if found else "REVIEW",  # REVIEW = no forbidden phrase, a human still checks it
        "verdict": None,  # filled in by hand: "pass" / "fail"
        "latency_s": round(latency, 2),
    }


def print_row(agent_name: str, row: dict) -> None:
    """Print one result so it can be reviewed straight in the terminal."""
    print(f"\n[{agent_name}] {row['id']} ({row['category']}) -> {row['auto_flag']}")
    print(f"  Prompt:   {row['prompt']}")
    print(f"  Expected: {row['expected_behavior']}")
    print(f"  Tools:    {', '.join(row['tool_calls']) or '-'}")
    if row["forbidden_found"]:
        print(f"  Forbidden phrases found: {row['forbidden_found']}")
    print(f"  Answer:   {row['answer']}")


def save_report(report: dict) -> Path:
    """Save the full report next to the golden-set results."""
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_results_dir / f"adversarial_{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    """CLI entry point: run every attack prompt on the chosen agents and save the answers."""
    parser = argparse.ArgumentParser(description="Run the adversarial test set against the agents.")
    parser.add_argument("--agents", nargs="+", choices=list(AGENTS), default=list(AGENTS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    items = load_adversarial_set()
    report = {"model": settings.llm_model, "date": f"{datetime.now():%Y-%m-%d}", "agents": {}}

    for name in args.agents:
        rows = []
        for item in items:
            row = run_item(AGENTS[name], item)
            print_row(name, row)
            rows.append(row)
        flagged = sum(row["auto_flag"] == "FAIL" for row in rows)
        report["agents"][name] = {"auto_fail": flagged, "total": len(rows), "rows": rows}

    print("\nSummary (auto flags only, review every answer by hand):")
    for name, data in report["agents"].items():
        print(f"  {name}: {data['auto_fail']}/{data['total']} answers contain a forbidden phrase")

    path = save_report(report)
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
