"""TechNova support agent built with LangGraph (same tools as the hand-written ReAct loop)."""

from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent import tools as store_tools
from src.config import settings

PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "graph_system.txt"


# --- Tools: thin wrappers around src/agent/tools.py (business logic stays there) ---


@tool
def get_order(order_id: int) -> dict:
    """
    Look up one customer order by its order number.

    Use this when the customer gives an order number or asks about a specific order
    (status, delivery, return eligibility, delivery fee). Returns ready-made decisions
    in `return_status` and `delivery_status`; repeat them to the customer.
    """
    return store_tools.get_order(order_id)


@tool
def search_policy(query: str) -> dict:
    """
    Search the TechNova store policy (returns, delivery, warranty, payments, reporting defects).

    Use this for general questions about store rules that do not need a specific order.
    Returns matching policy fragments with their source, or a note that the policy has
    no information on the topic; in that case tell the customer you don't have this information.
    """
    return store_tools.search_policy(query)


TOOLS = [get_order, search_policy]


def ollama_base_url() -> str:
    """Return the Ollama server address (scheme + host + port) from the configured endpoint URL."""
    parts = urlsplit(settings.ollama_url)
    return f"{parts.scheme}://{parts.netloc}"


def load_system_prompt() -> SystemMessage:
    """Load the system prompt for the LangGraph agent from config/prompts."""
    return SystemMessage(content=PROMPT_PATH.read_text(encoding="utf-8"))


def build_model():
    """Create the chat model with TechNova tools bound to it."""
    llm = ChatOllama(model=settings.llm_model, base_url=ollama_base_url(), temperature=0)
    return llm.bind_tools(TOOLS)


def make_agent_node(model, system_prompt: SystemMessage):
    """Return the agent node: the LLM reads the conversation and either answers or calls a tool."""

    def agent(state: MessagesState) -> dict:
        response = model.invoke([system_prompt, *state["messages"]])
        return {"messages": [response]}

    return agent


def build_graph():
    """Build the ReAct agent graph: agent -> (tool call?) -> tools -> agent ... -> END."""
    model = build_model()
    system_prompt = load_system_prompt()

    builder = StateGraph(MessagesState)
    builder.add_node("agent", make_agent_node(model, system_prompt))
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "Can I still return order 1042?"},
            ]
        }
    )
    for message in result["messages"]:
        message.pretty_print()
