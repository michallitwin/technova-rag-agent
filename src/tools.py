import sqlite3
from pathlib import Path

DB_PATH = Path("data/technova.db")


def get_order(order_id: int) -> dict:
    """Return one order as a dict, or an error dict if it does not exist. Read-only."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:   # open in read-only mode
        conn.row_factory = sqlite3.Row                                    # rows behave like dicts
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)       # parameter, not f-string!
        ).fetchone()
    if row is None:
        return {"error": f"Order {order_id} not found"}
    return dict(row)


if __name__ == "__main__":
    print(get_order(1042))
    print(get_order(9999))