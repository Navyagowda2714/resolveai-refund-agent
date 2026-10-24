"""Side-effecting actions: mock payments, templated notifications, human escalation queue."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from string import Template
from typing import Any

from resolveai.data.db import CommerceDB
from resolveai.models import RefundRecord, Ticket


class PaymentError(RuntimeError):
    pass


class MockPaymentsService:
    """Idempotent refund API backed by SQLite. Mirrors a real PSP contract (idempotency keys,
    balance checks) so the agent code is the same against a real provider."""

    def __init__(self, db: CommerceDB) -> None:
        self.db = db

    def refund(
        self, *, order_id: str, customer_id: str, amount: float, reason: str, idempotency_key: str
    ) -> str:
        existing = self.db.payment_by_key(idempotency_key)
        if existing:
            return str(existing["id"])
        order = self.db.get_order(order_id)
        if order is None:
            raise PaymentError(f"unknown order {order_id}")
        if amount <= 0:
            raise PaymentError("refund amount must be positive")
        remaining = order.total - self.db.refunded_amount_for_order(order_id)
        if amount > remaining + 0.01:
            raise PaymentError(f"amount {amount:.2f} exceeds refundable balance {remaining:.2f}")
        pay_id = f"PAY-{uuid.uuid4().hex[:10].upper()}"
        self.db.add_payment(pay_id, order_id, round(amount, 2), idempotency_key)
        self.db.add_refund(
            RefundRecord(
                id=pay_id.replace("PAY", "RFD"),
                customer_id=customer_id,
                order_id=order_id,
                amount=round(amount, 2),
                reason=reason,
                created_at=datetime.now(timezone.utc),
            )
        )
        return pay_id


TEMPLATES: dict[str, Template] = {
    "approve_refund": Template(
        "Hi $name,\n\nThanks for reaching out about order $order_id. We've issued a full refund of "
        "$amount $currency to your original payment method; it should appear within 3-5 business "
        "days.\n\n$reason_line\nReference: $ticket_id"
    ),
    "partial_refund": Template(
        "Hi $name,\n\nThanks for reaching out about order $order_id. We've issued a refund of "
        "$amount $currency to your original payment method.\n\n$reason_line\nReference: $ticket_id"
    ),
    "deny": Template(
        "Hi $name,\n\nThank you for contacting us about order $order_id. "
        "We've reviewed your request carefully but unfortunately it isn't eligible "
        "for a refund.\n\n$reason_line\nIf you think we've got this wrong, just reply "
        "and a member of our team will take another look.\n\nReference: $ticket_id"
    ),
    "escalate": Template(
        "Hi $name,\n\nThanks for reaching out about $order_ref. "
        "Your request needs a quick review by a member of our team, "
        "who will get back to you within 24 hours.\n\nReference: $ticket_id"
    ),
}


class NotificationService:
    def __init__(self, db: CommerceDB) -> None:
        self.db = db

    def render(self, template: str, **fields: Any) -> str:
        return TEMPLATES[template].safe_substitute(**fields)

    def notify(self, ticket: Ticket, template: str, **fields: Any) -> str:
        message = self.render(template, ticket_id=ticket.id, **fields)
        notif_id = f"NTF-{hashlib.sha1((ticket.id + template).encode()).hexdigest()[:8].upper()}"
        self.db.add_notification(notif_id, ticket.id, ticket.customer_email, template, message)
        return message


class HumanQueue:
    """Escalation queue persisted in SQLite. Each item carries a full handoff packet."""

    def __init__(self, db: CommerceDB) -> None:
        self.db = db

    def escalate(self, ticket_id: str, packet: dict[str, Any], priority: str = "normal") -> str:
        esc_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        self.db.add_escalation(esc_id, ticket_id, priority, packet)
        return esc_id

    def list_open(self) -> list[dict[str, Any]]:
        return self.db.list_escalations("open")

    def get(self, esc_id: str) -> dict[str, Any] | None:
        return self.db.get_escalation(esc_id)
