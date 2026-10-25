import json

from src.agent.router import choose_tool
from src.agent.tools import run_tool
from src.llm.client import ask_ollama

ANSWER_SYSTEM_PROMPT = (
    "Answer the customer's question in one or two sentences using only the tool result. "
    "If the tool result does not contain the answer, say you don't know."
)


def answer(question: str) -> dict:
    """One-step agent: choose a tool, run it, and let the LLM answer from the tool result."""
    choice = choose_tool(question)                                    # 1. decide
    result = run_tool(choice)                                         # 2. act
    prompt = (f"Tool used: {choice.get('tool')}\n"
              f"Tool result:\n{json.dumps(result, indent=2)}\n\n"
              f"Question: {question}")
    return {"answer": ask_ollama(prompt, system=ANSWER_SYSTEM_PROMPT),  # 3. answer
            "tool": choice, "result": result}


if __name__ == "__main__":
    for q in ["How much does delivery cost?",
              "What is the status of order 1043?",
              "What is the status of order 9999?",
              "Can I still return order 1042?"]:
        out = answer(q)
        print(f"\nQ: {q}\nTool: {out['tool']}\nA: {out['answer']}")