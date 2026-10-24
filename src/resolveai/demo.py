"""Five showcase tickets (one per channel) with their own fixture data."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from resolveai.data.db import CommerceDB
from resolveai.data.seed import DataFactory
from resolveai.evaluation.payloads import spoken

SHOWCASE_TITLES = [
    "Email: cracked dinner set, photo attached",
    "Live chat: one item missing from a 3-item order",
    "Phone: 1,149 EUR laptop won't turn on",
    "In-app: 'never arrived' on a brand-new account",
    "Portal: change of mind, 40 days after delivery",
]


def build_showcase(db: CommerceDB, now: datetime | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Insert showcase customers/orders into ``db`` and return (channel, payload) pairs."""
    now = now or datetime.now(timezone.utc)
    f = DataFactory(random.Random(42), now, prefix="S", order_start=90000)
    ts = now.isoformat()

    def add(c: Any, *orders: Any) -> None:
        db.add_customer(c)
        for o in orders:
            db.add_order(o)

    # 1. Email - damaged dinner set (64 EUR), with photo -> auto approve.
    maya = f.customer(name="Maya Rossi", email="maya.rossi@example.com", account_age_days=820)
    o1 = f.order(maya, items=[f.item("HM-300")], days_ago=6)
    add(maya, o1, *[f.order(maya, days_ago=d) for d in (90, 150, 300)])

    # 2. Chat - charging hub missing from a 3-item order -> auto partial refund.
    luca = f.customer(name="Luca Bianchi", email="luca.bianchi@example.com", account_age_days=400)
    o2 = f.order(luca, items=[f.item("AP-220"), f.item("EL-130"), f.item("BT-400")], days_ago=5)
    add(luca, o2, *[f.order(luca, days_ago=d) for d in (70, 200)])

    # 3. Phone - 1,149 EUR ultrabook defective -> escalate (above automation limit).
    aisha = f.customer(name="Aisha Khan", email="aisha.khan@example.com", account_age_days=1300)
    o3 = f.order(aisha, items=[f.item("EL-140")], days_ago=9)
    add(aisha, o3, *[f.order(aisha, days_ago=d) for d in (60, 120, 240, 365)])

    # 4. In-app - 'not received' on a 5-day-old account, carrier confirms delivery, billing in
    #    another country -> escalate with fraud signals.
    ivan = f.customer(name="Ivan Novak", email="ivan.novak@example.com", account_age_days=5)
    o4 = f.order(ivan, items=[f.item("EL-120")], days_ago=4, delivery_days=2, billing_country="DE")
    add(ivan, o4)

    # 5. Portal - hair dryer (beauty, 14-day window) returned after 40 days -> auto deny.
    tom = f.customer(name="Tom Martin", email="tom.martin@example.com", account_age_days=700)
    o5 = f.order(tom, items=[f.item("BT-410")], days_ago=43)
    add(tom, o5, f.order(tom, days_ago=200))

    return [
        (
            "email",
            {
                "message_id": "<demo-1@mail.example.com>",
                "from": "Maya Rossi <maya.rossi@example.com>",
                "subject": f"Broken plates in order {o1.id}",
                "body": "Hi,\n\nMy dinner set arrived today and two plates are cracked, "
                "the box was crushed. Photo attached. Could I get a refund please?"
                "\n\nThanks,\nMaya\n"
                "--\nSent from my phone\n> On Mon, ResolveShop wrote:\n> Your order shipped",
                "attachments": [
                    {
                        "filename": "plates.jpg",
                        "content_type": "image/jpeg",
                        "description": "two white ceramic plates with large cracks",
                    }
                ],
                "received_at": ts,
            },
        ),
        (
            "chat",
            {
                "session_id": "demo-chat-2",
                "customer": {"email": "luca.bianchi@example.com"},
                "order_id": o2.id,
                "messages": [
                    {"sender": "bot", "text": "Hi Luca! How can I help?", "ts": ts},
                    {"sender": "customer", "text": "My parcel came yesterday", "ts": ts},
                    {
                        "sender": "customer",
                        "text": "the shoes and the serum are there but the "
                        "charging hub wasn't in the box",
                        "ts": ts,
                    },
                ],
            },
        ),
        (
            "phone",
            {
                "call_id": "demo-call-3",
                "account_email": "aisha.khan@example.com",
                "started_at": ts,
                "duration_s": 312,
                "transcript": "AGENT: Thanks for calling, how can I help?\n"
                f"CUSTOMER: My order number is ORD {spoken(o3.id)}.\n"
                "AGENT: Thanks. What seems to be the problem?\n"
                "CUSTOMER: The new laptop won't turn on. I've tried charging it all "
                "night, nothing. I'd like a refund.\n"
                "AGENT: I'm sorry to hear that, let me check.",
            },
        ),
        (
            "in_app",
            {
                "request_id": "demo-app-4",
                "user": {"email": "ivan.novak@example.com"},
                "order_id": o4.id,
                "reason_code": "not_received",
                "comment": "Never arrived. I want my money back.",
                "created_at": ts,
            },
        ),
        (
            "portal",
            {
                "submission_id": "demo-portal-5",
                "submitted_at": ts,
                "form": {
                    "email": "tom.martin@example.com",
                    "order_number": o5.id,
                    "category": "Returns & refunds",
                    "description": "I don't really need the hair dryer anymore, changed my mind. "
                    "Can I send it back for a refund?",
                },
            },
        ),
    ]
