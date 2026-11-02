"""Tool tests: business decisions are made in code, so they must be deterministic."""

import sqlite3
from dataclasses import replace
from datetime import date

import pytest

from src.agent import tools

ORDERS = [
    (1041, "Anna Kowalski", "Wireless headphones", 59.99, "2026-09-15", "2026-09-18", "delivered"),
    (1042, "John Smith", "Smartwatch", 129.00, "2026-10-07", "2026-10-10", "delivered"),
    (1043, "Maria Garcia", "USB-C charger", 19.99, "2026-10-23", None, "shipped"),
    (1044, "Piotr Nowak", "Bluetooth speaker", 85.50, "2026-10-18", "2026-10-20", "delivered"),
    (1047, "Sophie Martin", "Webcam", 70.00, "2026-10-21", "2026-10-23", "delivered"),
]


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    """Small orders database in a temp folder and a frozen 'today' for every test."""
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT, product TEXT, "
            "total_usd REAL, order_date TEXT, delivery_date TEXT, status TEXT)"
        )
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", ORDERS)
    monkeypatch.setattr(tools, "settings", replace(tools.settings, db_path=db_path))
    monkeypatch.setattr(tools, "current_date", lambda: date(2026, 10, 26))


def test_order_of_exactly_70_has_paid_delivery():
    order = tools.get_order(1047)
    assert order["delivery_status"].startswith("NOT free delivery")


def test_order_over_70_has_free_delivery():
    order = tools.get_order(1044)
    assert order["delivery_status"].startswith("Free delivery")


def test_return_not_possible_after_30_days():
    order = tools.get_order(1041)
    assert order["return_window_open"] is False
    assert "2026-10-18" in order["return_status"]


def test_return_possible_within_30_days():
    order = tools.get_order(1042)
    assert order["return_window_open"] is True
    assert "2026-11-09" in order["return_status"]


def test_unknown_order_returns_error():
    order = tools.get_order(9999)
    assert "error" in order


def test_return_status_for_order_not_delivered_yet():
    order = tools.get_order(1043)
    assert order["return_status"].startswith("Not delivered yet")


def test_search_policy_drops_weak_chunks(monkeypatch):
    weak_result = [("Delivery", "store_policy.txt", 0.20, "Delivery. Orders placed before...")]
    monkeypatch.setattr(tools, "search_db", lambda query: weak_result)
    assert tools.search_policy("Do you ship to Japan?") == tools.NO_POLICY_INFO


def test_run_tool_without_order_number_asks_for_it():
    result = tools.run_tool({"tool": "get_order", "args": {}})
    assert "order number" in result["error"]