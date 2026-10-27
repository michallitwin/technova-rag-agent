import sqlite3
from datetime import date, timedelta

from src.config import current_date, settings
from src.rag.pipeline import search_db

NO_POLICY_INFO = {
    "chunks": [],
    "note": "No relevant information in the store policy. Tell the customer you don't have this information.",
}


def get_order(order_id: int) -> dict:
    """Return one order as a dict, or an error dict if it does not exist. Read-only."""
    with sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True) as conn:   # read-only
        conn.row_factory = sqlite3.Row                                            # rows like dicts
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)               # parameter, no f-string
        ).fetchone()
    if row is None:
        return {"error": f"Order {order_id} not found. Ask the customer to check the order number."}

    order = dict(row)
    # Dates and the return decision are computed in code, not by the LLM
    if order["delivery_date"]:
        today = current_date()
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
    # Delivery fee is decided in code, not by the LLM
    total = order["total_usd"]
    threshold = settings.free_delivery_threshold
    fee = settings.delivery_fee
    if total > threshold:
        order["delivery_status"] = f"Free delivery: order total ${total:.2f} is above ${threshold:.0f}."
    else:
        order["delivery_status"] = f"NOT free delivery: the customer paid ${fee:.0f} for delivery (free only above ${threshold:.0f}, order total ${total:.2f})."
    return order


def search_policy(query: str) -> dict:
    """Search the store policy; keep only chunks similar enough to the query."""
    chunks = []
    for section, source, similarity, text in search_db(query):
        if similarity >= settings.min_similarity:          # drop weakly related paragraphs
            chunks.append({"source": f"{source} / {section}", "text": text})
    if not chunks:
        return NO_POLICY_INFO                               # nothing relevant -> tell the model what to do
    return {"chunks": chunks}

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
    except (KeyError, ValueError, TypeError):
        if name == "get_order":
            return {"error": "No valid order number given. Ask the customer for their order number."}
        return {"error": f"Invalid arguments for {name}."}

