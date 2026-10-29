"""Streamlit demo: a simple UI client of the TechNova API (no agent code imported)."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_S = int(os.getenv("UI_TIMEOUT_S", "120"))
EXAMPLE_QUESTIONS = [
    "Can I still return order 1042?",
    "Did order 1047 get free delivery?",
    "How long is the warranty?",
    "Do you ship to Japan?",
    "What did Emma Brown order?",
]


def ask_agent(question: str) -> dict:
    """Send the question to POST /api/ask and return the JSON response."""
    response = requests.post(f"{API_URL}/api/ask", json={"question": question}, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def fetch_eval_summary() -> dict | None:
    """Latest evaluation summary from GET /api/eval/latest, or None if unavailable."""
    try:
        response = requests.get(f"{API_URL}/api/eval/latest", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def render_sidebar() -> None:
    """Show the agent's evaluation results next to the chat."""
    st.sidebar.header("Agent quality")
    summary = fetch_eval_summary()
    if summary is None:
        st.sidebar.warning("API not reachable or no evaluation report.")
        return
    st.sidebar.metric("Accuracy (golden set)", f"{summary['accuracy']:.0%}")
    st.sidebar.metric("Retrieval hit@k", f"{summary['hit_at_k']:.0%}")
    st.sidebar.metric("Avg. tool calls", summary["avg_steps"])
    st.sidebar.caption(f"Model: {summary['model']} · report: {summary['report']}")


def render_steps(steps: list[dict]) -> None:
    """Show what the agent did, step by step."""
    with st.expander(f"Agent steps ({len(steps)})"):
        for i, step in enumerate(steps, start=1):
            call = step["call"] or {}
            st.markdown(f"**Step {i}:** `{call.get('tool', 'invalid call')}` {call.get('args', {})}")
            st.json(step["result"], expanded=False)


def main() -> None:
    """Page layout: example questions, input, answer and agent steps."""
    st.set_page_config(page_title="TechNova Support Agent", page_icon="🛒")
    st.title("🛒 TechNova Support Agent")
    st.caption("RAG over store policy + SQL orders, ReAct loop. Demo client of the FastAPI service.")
    render_sidebar()

    example = st.selectbox("Try an example", [""] + EXAMPLE_QUESTIONS)
    question = st.text_input("Your question", value=example)
    if not (st.button("Ask", type="primary") and question.strip()):
        return

    with st.spinner("The agent is thinking..."):
        try:
            result = ask_agent(question)
        except requests.HTTPError as err:
            st.error(f"API error {err.response.status_code}: {err.response.json().get('detail')}")
            return
        except requests.RequestException:
            st.error(f"Cannot reach the API at {API_URL}. Is it running?")
            return

    st.success(result["answer"])
    if result["forced"]:
        st.caption("ℹ️ The code forced the final answer (the model did not finish on its own).")
    render_steps(result["steps"])


if __name__ == "__main__":
    main()