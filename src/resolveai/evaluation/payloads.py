"""Build raw channel payloads (as each channel would deliver them) from a message."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from resolveai.models import Channel

_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def spoken(order_id: str) -> str:
    return " ".join(_NUM_WORDS[int(d)] for d in order_id.split("-")[1])


def build_payload(
    channel: Channel,
    *,
    uid: str,
    email: str,
    name: str,
    order_id: str | None,
    text: str,
    created_at: datetime,
    reason_code: str = "other",
    photo_caption: str | None = None,
) -> dict[str, Any]:
    ts = created_at.isoformat()
    photos = (
        [{"filename": "photo_1.jpg", "content_type": "image/jpeg", "description": photo_caption}]
        if photo_caption
        else []
    )
    ref = f" (order {order_id})" if order_id else ""
    if channel == Channel.EMAIL:
        return {
            "message_id": f"<{uid}@mail.example.com>",
            "from": f"{name} <{email}>",
            "subject": f"Problem with my order{ref}",
            "body": f"Hello,\n\n{text}\n\nThanks,\n{name.split()[0]}\n--\nSent from my phone",
            "attachments": photos,
            "received_at": ts,
        }
    if channel == Channel.CHAT:
        return {
            "session_id": uid,
            "customer": {"email": email},
            "order_id": order_id,
            "messages": [
                {"sender": "bot", "text": "Hi! How can I help today?", "ts": ts},
                {"sender": "customer", "text": text, "ts": ts},
            ],
            "attachments": photos,
        }
    if channel == Channel.PHONE:
        num = f"My order number is ORD {spoken(order_id)}." if order_id else ""
        return {
            "call_id": uid,
            "account_email": email,
            "started_at": ts,
            "duration_s": 240,
            "transcript": (
                "AGENT: Thanks for calling, how can I help?\n"
                f"CUSTOMER: Hi, I'm calling about an order. {num}\n"
                "AGENT: Sure, what happened?\n"
                f"CUSTOMER: {text}\n"
                "AGENT: Thanks, let me look into that."
            ),
        }
    if channel == Channel.IN_APP:
        return {
            "request_id": uid,
            "user": {"email": email},
            "order_id": order_id,
            "reason_code": reason_code,
            "comment": text,
            "photos": photos,
            "created_at": ts,
            "app_version": "5.12.0",
        }
    return {
        "submission_id": uid,
        "submitted_at": ts,
        "form": {
            "email": email,
            "order_number": order_id or "",
            "category": "Returns & refunds",
            "description": text,
        },
        "files": photos,
    }
