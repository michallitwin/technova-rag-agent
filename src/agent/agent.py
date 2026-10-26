import argparse

from src.agent.answer import answer_from_results
from src.agent.router import choose_tool
from src.agent.tools import run_tool


def answer(question: str) -> dict:
    """Choose one tool, run it, and answer from its result."""
    choice = choose_tool(question)
    steps = [{"call": choice, "result": run_tool(choice)}]
    return {"answer": answer_from_results(question, steps), "steps": steps}


def main() -> None:
    """CLI entry point: ask the one-step agent a question and print its answer."""
    parser = argparse.ArgumentParser(description="Ask the one-step (baseline) agent a question.")
    parser.add_argument("question", help="e.g. 'What is the status of order 1043?'")
    args = parser.parse_args()
    print(f"\nAnswer: {answer(args.question)['answer']}")


if __name__ == "__main__":
    main()