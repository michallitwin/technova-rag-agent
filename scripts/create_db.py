import sqlite3
from pathlib import Path

DB_PATH = Path("data/technova.db")

# Sample orders; today in this project is 2026-10-25
ORDERS = [
    # (order_id, customer, product, total_usd, order_date, delivery_date, status)
    (1041, "Anna Kowalski", "Wireless headphones", 59.99, "2026-09-15", "2026-09-18", "delivered"),
    (1042, "John Smith", "Smartwatch", 129.00, "2026-10-07", "2026-10-10", "delivered"),
    (1043, "Maria Garcia", "USB-C charger", 19.99, "2026-10-23", None, "shipped"),
    (1044, "Piotr Nowak", "Bluetooth speaker", 85.50, "2026-10-17", "2026-10-20", "delivered"),
    (1045, "Emma Brown", "Laptop stand", 34.00, "2026-10-12", None, "cancelled"),
    (1046, "Lukas Meyer", "Mechanical keyboard", 99.90, "2026-10-24", None, "processing"),
]


def create_db():
    """Create the orders table and fill it with sample data (rebuilt from scratch each run)."""
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS orders")            # start clean every time
        conn.execute("""
            CREATE TABLE orders (
                order_id      INTEGER PRIMARY KEY,
                customer      TEXT NOT NULL,
                product       TEXT NOT NULL,
                total_usd     REAL NOT NULL,
                order_date    TEXT NOT NULL,      -- ISO date: YYYY-MM-DD
                delivery_date TEXT,               -- NULL if not delivered yet
                status        TEXT NOT NULL
            )
        """)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", ORDERS)
    print(f"Created {DB_PATH} with {len(ORDERS)} orders")


if __name__ == "__main__":
    create_db()