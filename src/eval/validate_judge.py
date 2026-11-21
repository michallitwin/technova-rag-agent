"""Validate the LLM judge against hand labels before trusting it.

Reads data/eval/judge_labels.json (86 agent answers, each with the keyword grader's verdict and a human
label), asks the judge to grade every answer and reports how often the grader and the judge agree with
the human. The judge is only worth using if it agrees with the human more often than the grader.

Usage:
    uv run --env-file .env python -m src.eval.validate_judge
    uv run --env-file .env python -m src.eval.validate_judge --limit 5   # quick test on 5 answers
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.eval.judge import judge_answer

LABELS_PATH = Path("data/eval/judge_labels.json")
SET_PATHS = {"golden": Path("data/eval/golden_set.json"), "heldout": Path("data/eval/heldout_set.json")}
RESULTS_DIR = Path("data/eval/results/judge")  # separate folder: /api/eval/latest must not read these


def load_expected_facts() -> dict[tuple[str, str], list[list[str]]]:
    """All accepted wordings of every fact, keyed by (set, question id)."""
    facts = {}
    for set_name, path in SET_PATHS.items():
        for item in json.loads(path.read_text(encoding="utf-8")):
            facts[(set_name, item["id"])] = item["expected_facts"]
    return facts


def agreement(labels: list[dict], field: str) -> dict:
    """How often `field` ("grader" or "judge") matches the human label, with the two kinds of mistakes."""
    rows = [row for row in labels if row.get(field) is not None]
    agree = sum(row[field] == row["human"] for row in rows)
    return {
        "rows": len(rows),
        "agree": agree,
        "agreement": round(agree / len(rows), 3) if rows else 0.0,
        "false_pass": sum(row[field] and not row["human"] for row in rows),  # wrong answer graded as correct
        "false_fail": sum(row["human"] and not row[field] for row in rows),  # correct answer graded as wrong
    }


def main() -> None:
    """CLI entry point: grade every labelled answer with the judge and compare with the human labels."""
    parser = argparse.ArgumentParser(description="Validate the LLM judge against hand labels.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N answers (quick test)")
    args = parser.parse_args()

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))[: args.limit]
    facts = load_expected_facts()

    start = time.perf_counter()
    for i, row in enumerate(labels, start=1):
        verdict = judge_answer(row["question"], facts[(row["set"], row["id"])], row["answer"])
        row["judge"], row["judge_reason"] = verdict["pass"], verdict["reason"]
        print(f"{i:>3}/{len(labels)} {row['set']:<7} {row['agent']:<5} {row['id']:<4} "
              f"human={row['human']!s:<5} judge={row['judge']!s:<5}")
    elapsed = time.perf_counter() - start

    summary = {"grader": agreement(labels, "grader"), "judge": agreement(labels, "judge")}
    unusable = sum(row["judge"] is None for row in labels)

    print(f"\nJudge: {settings.judge_model} | answers: {len(labels)} | no verdict: {unusable} | {elapsed:.0f} s")
    print(f"{'':<8}{'agreement':>11}{'false pass':>12}{'false fail':>12}")
    for name, s in summary.items():
        print(f"{name:<8}{s['agreement']:>11.1%}{s['false_pass']:>12}{s['false_fail']:>12}")

    print("\nJudge disagrees with the human:")
    for row in labels:
        if row["judge"] is not None and row["judge"] != row["human"]:
            print(f"  {row['set']} {row['agent']} {row['id']}: human={row['human']} judge={row['judge']}"
                  f" | {row['judge_reason']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"validation_{datetime.now():%Y%m%d_%H%M}.json"
    report = {"judge_model": settings.judge_model, "summary": summary, "no_verdict": unusable, "rows": labels}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
