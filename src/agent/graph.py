"""TechNova support agent built with LangGraph (same tools as the hand-written ReAct loop)."""

from functools import cache
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
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
    Write the query as a short policy topic such as "returns", "warranty" or "delivery",
    without product names or order numbers.
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

    return builder.compile(checkpointer=InMemorySaver())


def run_agent(graph, question: str, thread_id: str) -> str:
    """Ask the agent one question in a conversation thread and return its answer."""
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {
                "recursion_limit": 8,
                "configurable": {"thread_id": thread_id},
            },
        )
        return result["messages"][-1].content
    except GraphRecursionError:
        return "Sorry, I can't answer this. Please contact our support team."


@cache
def get_graph():
    """Build the graph once and reuse it for every question."""
    return build_graph()


def run_graph_agent(question: str) -> dict:
    """Run the LangGraph agent and return the same format as the ReAct agent (used by the evaluation)."""
    graph = get_graph()
    config = {"recursion_limit": 8, "configurable": {"thread_id": str(uuid4())}}
    try:
        result = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)
    except GraphRecursionError:
        return {"answer": "Sorry, I can't answer this. Please contact our support team.", "steps": [], "forced": True}

    steps = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            steps.append({"call": {"tool": call["name"], "args": call["args"]}})
    return {"answer": result["messages"][-1].content, "steps": steps, "forced": False}


if __name__ == "__main__":
    graph = build_graph()
    print(run_agent(graph, "Can I still return order 1043?", "customer-a"))
    print(run_agent(graph, "What is the status of order 1043?", "customer-b"))
    print(run_agent(graph, "Can I return it?", "customer-b"))
    state = graph.get_state({"configurable": {"thread_id": "customer-b"}})
    for message in state.values["messages"]:
        message.pretty_print()
