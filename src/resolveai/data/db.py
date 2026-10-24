"""SQLite-backed mock commerce store + ticket/queue persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resolveai.models import Customer, Order, RefundRecord, Resolution, Ticket

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    created_at TEXT NOT NULL, tier TEXT NOT NULL DEFAULT 'standard'
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
    ordered_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id, ordered_at);
CREATE TABLE IF NOT EXISTS refunds (
    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, order_id TEXT NOT NULL,
    amount REAL NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed'
);
CREATE INDEX IF NOT EXISTS idx_refunds_customer ON refunds(customer_id);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, channel TEXT NOT NULL, customer_email TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resolutions (
    ticket_id TEXT PRIMARY KEY, outcome TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, packet TEXT NOT NULL,
    resolved_at TEXT, resolved_by TEXT, human_action TEXT, human_amount REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY, order_id TEXT NOT NULL, amount REAL NOT NULL,
    idempotency_key TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, customer_email TEXT NOT NULL,
    template TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommerceDB:
    """Thin repository over SQLite. Thread-safe for the FastAPI worker via a lock."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _one(self, sql: str, args: tuple[Any, ...]) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def _all(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ commerce
    def add_customer(self, c: Customer) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?)",
                (c.id, c.email.lower(), c.name, c.created_at.isoformat(), c.tier),
            )

    def add_order(self, o: Order) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders VALUES (?,?,?,?)",
                (o.id, o.customer_id, o.ordered_at.isoformat(), o.model_dump_json()),
            )

    def add_refund(self, r: RefundRecord) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO refunds VALUES (?,?,?,?,?,?,?)",
                (
                    r.id,
                    r.customer_id,
                    r.order_id,
                    r.amount,
                    r.reason,
                    r.created_at.isoformat(),
                    r.status,
                ),
            )

    @staticmethod
    def _customer(row: sqlite3.Row) -> Customer:
        return Customer(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            tier=row["tier"],
        )

    def get_customer_by_email(self, email: str) -> Customer | None:
        row = self._one("SELECT * FROM customers WHERE email = ?", (email.lower(),))
        return self._customer(row) if row else None

    def get_customer(self, customer_id: str) -> Customer | None:
        row = self._one("SELECT * FROM customers WHERE id = ?", (customer_id,))
        return self._customer(row) if row else None

    def get_order(self, order_id: str) -> Order | None:
        row = self._one("SELECT data FROM orders WHERE id = ?", (order_id,))
        return Order.model_validate_json(row["data"]) if row else None

    def orders_for_customer(self, customer_id: str) -> list[Order]:
        rows = self._all(
            "SELECT data FROM orders WHERE customer_id = ? ORDER BY ordered_at DESC", (customer_id,)
        )
        return [Order.model_validate_json(r["data"]) for r in rows]

    def refunds_for_customer(self, customer_id: str) -> list[RefundRecord]:
        rows = self._all(
            "SELECT * FROM refunds WHERE customer_id = ? ORDER BY created_at DESC", (customer_id,)
        )
        return [
            RefundRecord(
                id=r["id"],
                customer_id=r["customer_id"],
                order_id=r["order_id"],
                amount=r["amount"],
                reason=r["reason"],
                created_at=datetime.fromisoformat(r["created_at"]),
                status=r["status"],
            )
            for r in rows
        ]

    def refunded_amount_for_order(self, order_id: str) -> float:
        row = self._one(
            "SELECT COALESCE(SUM(amount), 0) AS s FROM refunds "
            "WHERE order_id = ? AND status = 'completed'",
            (order_id,),
        )
        return round(float(row["s"]) if row else 0.0, 2)

    def counts(self) -> dict[str, int]:
        tables = ["customers", "orders", "refunds", "tickets", "escalations", "payments"]
        return {t: self._all(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in tables}

    # ------------------------------------------------------------------ tickets & resolutions
    def save_ticket(self, t: Ticket, status: str = "open") -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?)",
                (
                    t.id,
                    t.channel.value,
                    t.customer_email,
                    status,
                    t.created_at.isoformat(),
                    t.model_dump_json(),
                ),
            )

    def set_ticket_status(self, ticket_id: str, status: str) -> None:
        with self.tx() as conn:
            conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))

    def get_ticket(self, ticket_id: str) -> tuple[Ticket, str] | None:
        row = self._one("SELECT data, status FROM tickets WHERE id = ?", (ticket_id,))
        return (Ticket.model_validate_json(row["data"]), row["status"]) if row else None

    def list_tickets(self, status: str | None = None) -> list[tuple[Ticket, str]]:
        sql, args = "SELECT data, status FROM tickets", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        rows = self._all(sql + " ORDER BY created_at DESC", args)
        return [(Ticket.model_validate_json(r["data"]), r["status"]) for r in rows]

    def save_resolution(self, r: Resolution) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO resolutions VALUES (?,?,?,?)",
                (r.ticket_id, r.outcome.value, _now(), r.model_dump_json()),
            )

    def get_resolution(self, ticket_id: str) -> Resolution | None:
        row = self._one("SELECT data FROM resolutions WHERE ticket_id = ?", (ticket_id,))
        return Resolution.model_validate_json(row["data"]) if row else None

    # ------------------------------------------------------------------ escalations (human queue)
    def add_escalation(
        self, esc_id: str, ticket_id: str, priority: str, packet: dict[str, Any]
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO escalations (id, ticket_id, priority, status, created_at, packet) "
                "VALUES (?,?,?,?,?,?)",
                (esc_id, ticket_id, priority, "open", _now(), json.dumps(packet, default=str)),
            )

    def list_escalations(self, status: str | None = "open") -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM escalations", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        order = (
            " ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, created_at"
        )
        return [self._esc(r) for r in self._all(sql + order, args)]

    def get_escalation(self, esc_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM escalations WHERE id = ?", (esc_id,))
        return self._esc(row) if row else None

    def close_escalation(
        self, esc_id: str, *, by: str, action: str, amount: float, notes: str
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE escalations SET status='resolved', resolved_at=?, resolved_by=?, "
                "human_action=?, human_amount=?, notes=? WHERE id=?",
                (_now(), by, action, amount, notes, esc_id),
            )

    @staticmethod
    def _esc(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["packet"] = json.loads(d["packet"])
        return d

    # ------------------------------------------------------------------ payments & notifications
    def payment_by_key(self, key: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM payments WHERE idempotency_key = ?", (key,))
        return dict(row) if row else None

    def add_payment(self, pay_id: str, order_id: str, amount: float, key: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO payments VALUES (?,?,?,?,?)", (pay_id, order_id, amount, key, _now())
            )

    def add_notification(
        self, notif_id: str, ticket_id: str, email: str, template: str, message: str
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO notifications VALUES (?,?,?,?,?,?)",
                (notif_id, ticket_id, email, template, message, _now()),
            )

    def notifications_for(self, ticket_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._all("SELECT * FROM notifications WHERE ticket_id = ?", (ticket_id,))
        ]
