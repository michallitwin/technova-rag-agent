import sqlite3
from src.rag.pipeline import search_db
from datetime import date, timedelta
from src.config import settings


def get_order(order_id: int) -> dict:
    """Return one order as a dict, or an error dict if it does not exist. Read-only."""
    with sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True) as conn:   # read-only
        conn.row_factory = sqlite3.Row                                            # rows like dicts
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)               # parameter, no f-string
        ).fetchone()
    if row is None:
        return {"error": f"Order {order_id} not found"}

    order = dict(row)
    # Dates and the return decision are computed in code, not by the LLM
    if order["delivery_date"]:
        today = date.today()
        delivered = date.fromisoformat(order["delivery_date"])
        deadline = delivered + timedelta(days=settings.return_window_days)
        order["days_since_delivery"] = (today - delivered).days
        order["return_deadline"] = deadline.isoformat()
        order["return_window_open"] = today <= deadline
        order["return_status"] = (
            f"Return possible until {deadline.isoformat()}."
            if order["return_window_open"]
            else f"Return NOT possible: the return window closed on {deadline.isoformat()}."
        )
    return order


def search_policy(query: str) -> dict:
    """Search the store policy documents; return the best chunks with their sources."""
    results = search_db(query)
    return {"chunks": [{"source": f"{src} / {sec}", "text": doc} for sec, src, sim, doc in results]}


TOOLS = {"search_policy": search_policy, "get_order": get_order}


def run_tool(choice: dict) -> dict:
    """Validate the router's choice, then run the tool. Errors are returned as data, not raised."""
    name = choice.get("tool")
    args = choice.get("args", {})
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    try:
        if name == "get_order":
            return get_order(int(args["order_id"]))        # must be an integer
        return search_policy(str(args["query"]))          # must be text
    except (KeyError, ValueError, TypeError) as err:
        return {"error": f"Invalid arguments for {name}: {err}"}

