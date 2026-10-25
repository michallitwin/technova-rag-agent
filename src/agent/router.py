from src.llm.client import ask_ollama_json

TOOLS_DESCRIPTION = """You can use these tools:
1. search_policy(query: str) - search the store policy documents (returns, delivery, warranty, payments).
   Use it for general questions about store rules. Do NOT use it for questions about a specific order.
   Example: {"tool": "search_policy", "args": {"query": "return period"}}
2. get_order(order_id: int) - get one order from the database (product, dates, status).
   Use it when the user mentions a specific order number. Do NOT use it for general rules.
   Example: {"tool": "get_order", "args": {"order_id": 1042}}
"""


def choose_tool(question: str) -> dict:
    """Ask the LLM which tool to use for the question; return {"tool": ..., "args": {...}}."""
    system = TOOLS_DESCRIPTION + '\nAnswer only with JSON: {"tool": "<tool name>", "args": {<arguments>}}'
    return ask_ollama_json(question, system=system)


if __name__ == "__main__":
    for q in ["How much does delivery cost?",
              "What is the status of order 1043?",
              "Can I still return order 1042?"]:
        print(q, "->", choose_tool(q))