import pytest

from resolveai.intake import IntakeError, extract_order_id, normalise
from resolveai.models import Channel


def test_email_strips_quoted_reply_and_signature():
    t = normalise(
        "email",
        {
            "from": "Maya Rossi <Maya.Rossi@Example.com>",
            "subject": "Order ORD-12345 broken",
            "body": "My lamp is broken.\n--\nSent from my phone\n> On Monday you wrote:\n> shipped",
            "received_at": "2026-10-20T10:00:00Z",
        },
    )
    assert t.channel == Channel.EMAIL
    assert t.customer_email == "maya.rossi@example.com"
    assert t.order_id == "ORD-12345"
    assert "shipped" not in t.body and "Sent from" not in t.body


def test_chat_keeps_only_customer_turns():
    t = normalise(
        "chat",
        {
            "customer": {"email": "a@example.com"},
            "messages": [
                {"sender": "bot", "text": "Hello! order ORD-99999?", "ts": "2026-10-20T10:00:00Z"},
                {"sender": "customer", "text": "order 54321 never arrived"},
            ],
        },
    )
    assert t.body == "order 54321 never arrived"
    assert t.order_id == "ORD-54321"


def test_phone_transcript_spoken_digits():
    t = normalise(
        "phone",
        {
            "account_email": "b@example.com",
            "transcript": "AGENT: hi\nCUSTOMER: my order is ORD five zero zero one two\n"
            "AGENT: ok\nCALLER: it never arrived",
        },
    )
    assert t.order_id == "ORD-50012"
    assert "AGENT" not in t.body and "never arrived" in t.body


def test_in_app_reason_code_and_photos():
    t = normalise(
        "in_app",
        {
            "user": {"email": "c@example.com"},
            "order_id": "ORD-11111",
            "reason_code": "damaged",
            "comment": "",
            "photos": [{"filename": "x.jpg", "description": "cracked lamp"}],
        },
    )
    assert t.subject == "Item arrived damaged"
    assert t.attachments[0].content_type == "image/jpeg"


def test_portal_form():
    t = normalise(
        "portal",
        {
            "form": {
                "email": "d@example.com",
                "order_number": "ord 22222",
                "description": "please refund",
            }
        },
    )
    assert t.order_id == "ORD-22222"


def test_invalid_payloads_raise():
    with pytest.raises(IntakeError):
        normalise("fax", {})
    with pytest.raises(IntakeError):
        normalise("email", {"body": "no sender"})
    with pytest.raises(IntakeError):
        normalise(
            "chat",
            {"customer": {"email": "x@example.com"}, "messages": [{"sender": "bot", "text": "hi"}]},
        )


def test_ticket_ids_are_deterministic():
    p = {"from": "x@example.com", "body": "hi", "message_id": "m1"}
    assert normalise("email", p).id == normalise("email", p).id
    assert extract_order_id(None, "nothing here") is None
