"""Channel adapters: normalise raw payloads from each support channel into a ``Ticket``.

Every channel has its own quirks (quoted email replies, multi-turn chat, speaker-labelled phone
transcripts, structured in-app reason codes, portal web forms). Adapters isolate those quirks so
the agent only ever sees one ``Ticket`` model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from resolveai.models import Attachment, Channel, Ticket

ORDER_ID_RE = re.compile(r"\bORD(?:ER)?[\s#:-]*(\d{4,6})\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class IntakeError(ValueError):
    """Raised when a payload cannot be normalised into a ticket."""


def _ticket_id(channel: Channel, source_id: str) -> str:
    digest = hashlib.sha1(f"{channel.value}:{source_id}".encode()).hexdigest()[:10]
    return f"TKT-{digest.upper()}"


def _parse_ts(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def extract_order_id(*texts: str | None) -> str | None:
    for text in texts:
        if text and (m := ORDER_ID_RE.search(text)):
            return f"ORD-{m.group(1)}"
    return None


def _normalise_order_id(raw: str | None) -> str | None:
    return extract_order_id(raw) if raw else None


def _attachments(raw: list[dict[str, Any]] | None) -> list[Attachment]:
    return [Attachment.model_validate(a) for a in raw or []]


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise IntakeError(f"payload missing required fields: {missing}")


# --------------------------------------------------------------------------------------- email
def _clean_email_body(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        is_signature = stripped == "--" or lowered.startswith("sent from my")
        is_reply_header = lowered.startswith("on ") and lowered.endswith("wrote:")
        if is_signature or is_reply_header:
            break
        if stripped.startswith(">"):
            continue  # quoted reply
        lines.append(line)
    return "\n".join(lines).strip()


def from_email(payload: dict[str, Any]) -> Ticket:
    _require(payload, "from", "body")
    sender = EMAIL_RE.search(payload["from"])
    if not sender:
        raise IntakeError("could not find a sender address")
    subject = payload.get("subject", "")
    body = _clean_email_body(payload["body"])
    return Ticket(
        id=_ticket_id(Channel.EMAIL, payload.get("message_id", subject + body)),
        channel=Channel.EMAIL,
        customer_email=sender.group(0).lower(),
        order_id=extract_order_id(subject, body),
        subject=subject,
        body=body,
        attachments=_attachments(payload.get("attachments")),
        created_at=_parse_ts(payload.get("received_at")),
        metadata={"message_id": payload.get("message_id")},
    )


# ---------------------------------------------------------------------------------------- chat
def from_chat(payload: dict[str, Any]) -> Ticket:
    _require(payload, "customer", "messages")
    customer_lines = [m["text"] for m in payload["messages"] if m.get("sender") == "customer"]
    if not customer_lines:
        raise IntakeError("chat contains no customer messages")
    body = "\n".join(customer_lines)
    first_ts = payload["messages"][0].get("ts")
    return Ticket(
        id=_ticket_id(Channel.CHAT, payload.get("session_id", body)),
        channel=Channel.CHAT,
        customer_email=payload["customer"]["email"].lower(),
        order_id=_normalise_order_id(payload.get("order_id")) or extract_order_id(body),
        subject="Live chat",
        body=body,
        attachments=_attachments(payload.get("attachments")),
        created_at=_parse_ts(first_ts),
        metadata={"session_id": payload.get("session_id"), "turns": len(payload["messages"])},
    )


# --------------------------------------------------------------------------------------- phone
_SPEAKER_RE = re.compile(r"^\s*(AGENT|CUSTOMER|CALLER)\s*:\s*(.*)$", re.IGNORECASE)
_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def _spoken_to_digits(text: str) -> str:
    """'order one zero two three four' -> 'order 10234' (speech-to-text often spells digits)."""
    words = text.split()
    out: list[str] = []
    buf = ""
    for w in words:
        key = w.lower().strip(".,")
        if key in _SPOKEN_DIGITS:
            buf += _SPOKEN_DIGITS[key]
            continue
        if buf:
            out.append(buf)
            buf = ""
        out.append(w)
    if buf:
        out.append(buf)
    return " ".join(out)


def from_phone(payload: dict[str, Any]) -> Ticket:
    _require(payload, "account_email", "transcript")
    customer_lines: list[str] = []
    for line in payload["transcript"].splitlines():
        m = _SPEAKER_RE.match(line)
        if m and m.group(1).upper() in {"CUSTOMER", "CALLER"}:
            customer_lines.append(_spoken_to_digits(m.group(2)))
    if not customer_lines:
        raise IntakeError("transcript has no customer turns")
    body = "\n".join(customer_lines)
    return Ticket(
        id=_ticket_id(Channel.PHONE, payload.get("call_id", body)),
        channel=Channel.PHONE,
        customer_email=payload["account_email"].lower(),
        order_id=_normalise_order_id(payload.get("order_id")) or extract_order_id(body),
        subject="Phone call",
        body=body,
        created_at=_parse_ts(payload.get("started_at")),
        metadata={"call_id": payload.get("call_id"), "duration_s": payload.get("duration_s")},
    )


# -------------------------------------------------------------------------------------- in-app
_REASON_CODES = {
    "damaged": "Item arrived damaged",
    "defective": "Item is defective",
    "not_received": "Order not received",
    "wrong_item": "Received the wrong item",
    "missing_item": "Item missing from order",
    "changed_mind": "No longer needed",
    "other": "Other issue",
}


def from_in_app(payload: dict[str, Any]) -> Ticket:
    _require(payload, "user", "reason_code")
    reason = payload["reason_code"]
    comment = payload.get("comment", "")
    photos = [
        {"filename": p.get("filename", "photo.jpg"), "content_type": "image/jpeg", **p}
        for p in payload.get("photos", [])
    ]
    return Ticket(
        id=_ticket_id(Channel.IN_APP, payload.get("request_id", reason + comment)),
        channel=Channel.IN_APP,
        customer_email=payload["user"]["email"].lower(),
        order_id=_normalise_order_id(payload.get("order_id")),
        subject=_REASON_CODES.get(reason, reason),
        body=comment or _REASON_CODES.get(reason, reason),
        attachments=_attachments(photos),
        created_at=_parse_ts(payload.get("created_at")),
        metadata={"reason_code": reason, "app_version": payload.get("app_version")},
    )


# -------------------------------------------------------------------------------------- portal
def from_portal(payload: dict[str, Any]) -> Ticket:
    _require(payload, "form")
    form = payload["form"]
    if "email" not in form or "description" not in form:
        raise IntakeError("portal form needs 'email' and 'description'")
    return Ticket(
        id=_ticket_id(Channel.PORTAL, payload.get("submission_id", form["description"])),
        channel=Channel.PORTAL,
        customer_email=form["email"].lower(),
        order_id=_normalise_order_id(form.get("order_number"))
        or extract_order_id(form["description"]),
        subject=form.get("category", "Support request"),
        body=form["description"],
        attachments=_attachments(payload.get("files")),
        created_at=_parse_ts(payload.get("submitted_at")),
        metadata={"submission_id": payload.get("submission_id")},
    )


ADAPTERS: dict[Channel, Callable[[dict[str, Any]], Ticket]] = {
    Channel.EMAIL: from_email,
    Channel.CHAT: from_chat,
    Channel.PHONE: from_phone,
    Channel.IN_APP: from_in_app,
    Channel.PORTAL: from_portal,
}


def normalise(channel: Channel | str, payload: dict[str, Any]) -> Ticket:
    """Normalise a raw channel payload into a ``Ticket``."""
    try:
        ch = Channel(channel)
    except ValueError as exc:
        raise IntakeError(f"unknown channel: {channel}") from exc
    return ADAPTERS[ch](payload)
