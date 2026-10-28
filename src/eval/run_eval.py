"""Evaluate both agents on the golden set: correctness, tool use, retrieval hit@k, steps, latency."""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.agent.agent import answer as baseline_agent
from src.agent.react import run_agent as react_agent
from src.config import settings
from src.rag.pipeline import search_db

logger = logging.getLogger(__name__)

AGENTS = {"baseline": baseline_agent, "react": react_agent}
METRICS = ["accuracy", "tools_ok", "avg_steps", "forced_rate", "avg_latency_s"]
LABEL_WIDTH = 22  # width of the first column (metric name)
COLUMN_WIDTH = 12  # width of each agent's column


def load_golden_set() -> list[dict]:
    """Read the golden set (questions with expected facts, tools and section) from JSON."""
    return json.loads(settings.eval_set_path.read_text(encoding="utf-8"))


def facts_found(answer: str, expected_facts: list[list[str]]) -> bool:
    """True if the answer contains at least one variant of every expected fact (case-insensitive)."""
    text = answer.lower()

    for variants in expected_facts:  # go through each FACT
        fact_found = False
        for variant in variants:  # go through each WORDING of this fact
            # match whole words only: "no" must not match "not" or "know"
            pattern = rf"(?<!\w){re.escape(variant.lower())}(?!\w)"
            if re.search(pattern, text):
                fact_found = True  # one wording is enough
                break
        if not fact_found:
            return False  # one fact missing -> answer fails

    return True


def tools_used(steps: list[dict]) -> list[str]:
    """Names of the tools the agent called (skipping steps without a valid call)."""
    names = []
    for step in steps:
        call = step.get("call")  # None if the model's reply was not valid JSON
        if call and "tool" in call:  # skip empty calls and calls without a tool name
            names.append(call["tool"])
    return names


def retrieval_hit(question: str, expected_section: str | None) -> bool | None:
    """hit@k: is the expected policy section among the top-k chunks? None if not applicable."""
    if (
        expected_section is None
    ):  # e.g. "What is the status of order 1043?" - no policy needed
        return None

    for section, source, similarity, text in search_db(question):  # top-k chunks
        if section == expected_section:
            return True  # found the right paragraph
    return False  # right paragraph not among top-k


def evaluate_item(agent, item: dict) -> dict:
    """Run one agent on one golden-set question and score the result."""
    start = time.perf_counter()
    result = agent(item["question"])
    latency = time.perf_counter() - start
    used = tools_used(result["steps"])
    return {
        "id": item["id"],
        "category": item["category"],
        "answer": result["answer"],
        "correct": facts_found(result["answer"], item["expected_facts"]),
        "tools_ok": set(item["expected_tools"]) <= set(used),
        "tools_used": used,
        "steps": len(result["steps"]),
        "forced": result.get("forced", False),
        "latency_s": round(latency, 2),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-question rows into overall and per-category metrics."""
    n = len(rows)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["correct"])
    return {
        "accuracy": round(sum(r["correct"] for r in rows) / n, 2),
        "tools_ok": round(sum(r["tools_ok"] for r in rows) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "forced_rate": round(sum(r["forced"] for r in rows) / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "accuracy_by_category": {
            cat: round(sum(v) / len(v), 2) for cat, v in by_category.items()
        },
    }


def run_evaluation(golden_set: list[dict], agent_names: list[str]) -> dict:
    """Measure retrieval hit@k once, then evaluate each selected agent on every question."""
    hits = []
    for item in golden_set:
        hit = retrieval_hit(item["question"], item["expected_section"])
        if hit is not None:
            hits.append(hit)

    report = {
        "date": settings.today_override,
        "model": settings.llm_model,
        "top_k": settings.top_k,
        "hit_at_k": round(sum(hits) / len(hits), 2),
        "agents": {},
    }
    for name in agent_names:
        rows = []
        for item in golden_set:
            logger.info("%s: %s", name, item["id"])
            rows.append(evaluate_item(AGENTS[name], item))
        report["agents"][name] = {"summary": summarize(rows), "rows": rows}
    return report


def table_row(label: str, values: list) -> str:
    """One line of the summary table: a label and one column per agent."""
    columns = "".join(f"{value:>{COLUMN_WIDTH}}" for value in values)
    return f"{label:<{LABEL_WIDTH}}{columns}"


def format_summary(report: dict) -> str:
    """Readable comparison table of the agents' summaries."""
    names = list(report["agents"])
    summaries = [report["agents"][name]["summary"] for name in names]

    lines = [
        f"Model: {report['model']} | date: {report['date']} | hit@{report['top_k']}: {report['hit_at_k']:.0%}",
        "",
        table_row("metric", names),
    ]
    for metric in METRICS:
        lines.append(table_row(metric, [s[metric] for s in summaries]))
    for category in summaries[0]["accuracy_by_category"]:
        values = [s["accuracy_by_category"][category] for s in summaries]
        lines.append(table_row(f"accuracy {category}", values))
    return "\n".join(lines)


def save_report(report: dict) -> Path:
    """Save the full report as JSON in the results folder and return the file path."""
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_results_dir / f"{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    """CLI entry point: run the evaluation, save the report and print a summary."""
    parser = argparse.ArgumentParser(
        description="Evaluate the agents on the golden set."
    )
    parser.add_argument(
        "--agents", nargs="+", choices=list(AGENTS), default=list(AGENTS)
    )
    args = parser.parse_args()
    if not settings.today_override:
        parser.error("set TODAY=YYYY-MM-DD so date-based answers are reproducible")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    report = run_evaluation(load_golden_set(), args.agents)
    path = save_report(report)
    print(format_summary(report))
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
