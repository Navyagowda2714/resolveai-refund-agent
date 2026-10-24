import json

import pytest

from conftest import chat_payload
from resolveai.agent.offline_policy import OFFLINE_HANDLERS
from resolveai.llm import ScriptedLLM
from resolveai.models import DecisionAction, Outcome
from resolveai.service import ResolveService


def _setup(db, factory, sku="HM-330", days_ago=5, **cust_kw):
    c = factory.customer(account_age_days=cust_kw.pop("age", 600), **cust_kw)
    db.add_customer(c)
    for d in (80, 160):
        db.add_order(factory.order(c, days_ago=d))
    o = factory.order(c, items=[factory.item(sku)], days_ago=days_ago)
    db.add_order(o)
    return c, o


def test_low_value_damage_auto_refunds_and_notifies(service, db, factory):
    c, o = _setup(db, factory)
    t = service.ingest("chat", chat_payload(c.email, "The lamp arrived cracked.", o.id))
    res = service.resolve(t.id)
    assert res.outcome == Outcome.AUTO_APPROVED and res.amount_refunded == 54.0
    assert db.refunded_amount_for_order(o.id) == 54.0
    assert db.notifications_for(t.id)[0]["template"] == "approve_refund"
    assert {c.name for c in res.tool_calls} >= {
        "fetch_order_details",
        "run_fraud_checks",
        "check_refund_policy",
    }


def test_high_value_escalates_with_handoff_packet(service, db, factory):
    c, o = _setup(db, factory, sku="EL-140")
    t = service.ingest("chat", chat_payload(c.email, "The laptop won't turn on.", o.id))
    res = service.resolve(t.id)
    assert res.outcome == Outcome.ESCALATED and "amount_above_auto_limit" in res.guardrail_flags
    esc = db.get_escalation(res.escalation_id)
    packet = esc["packet"]
    assert packet["priority"] == "high"
    assert packet["ai_proposal"]["action"] == "approve_refund"
    assert "automation limit" in " ".join(packet["suggested_next_steps"])
    assert db.refunded_amount_for_order(o.id) == 0.0


def test_resolve_is_idempotent(service, db, factory):
    c, o = _setup(db, factory)
    t = service.ingest("chat", chat_payload(c.email, "The lamp arrived cracked.", o.id))
    first = service.resolve(t.id)
    second = service.resolve(t.id)
    assert first.refund_id == second.refund_id
    assert db.refunded_amount_for_order(o.id) == 54.0


def _overconfident_llm():
    """A misbehaving model: always approves a 500 EUR refund with full confidence."""
    handlers = dict(OFFLINE_HANDLERS)
    handlers["decide"] = lambda ctx: {
        "action": "approve_refund",
        "amount": 500.0,
        "confidence": 1.0,
        "reasons": ["trust me"],
        "policy_citations": [],
    }
    return ScriptedLLM(handlers)


def test_safety_gate_blocks_overconfident_model(db, factory):
    svc = ResolveService(db, llm=_overconfident_llm())
    c, o = _setup(db, factory, sku="BT-400", days_ago=60)  # outside 14-day window
    t = svc.ingest("chat", chat_payload(c.email, "I changed my mind.", o.id))
    res = svc.resolve(t.id)
    assert res.outcome == Outcome.ESCALATED
    assert {"policy_conflict:refund_not_eligible", "amount_above_auto_limit"} <= set(
        res.guardrail_flags
    )
    assert db.refunded_amount_for_order(o.id) == 0.0


def test_gate_disabled_ablation_lets_unsafe_refund_through_but_psp_caps(db, factory):
    svc = ResolveService(db, llm=_overconfident_llm(), enforce_guardrails=False)
    c, o = _setup(db, factory, sku="BT-400", days_ago=60)
    t = svc.ingest("chat", chat_payload(c.email, "I changed my mind.", o.id))
    res = svc.resolve(t.id)
    # Payments service still refuses to refund more than the order total.
    assert any(f.startswith("payment_failed") for f in res.guardrail_flags)
    assert db.refunded_amount_for_order(o.id) == 0.0


def test_invalid_model_output_escalates(db, factory):
    handlers = dict(OFFLINE_HANDLERS)
    handlers["decide"] = lambda ctx: {"action": "give_everything", "confidence": 3}
    svc = ResolveService(db, llm=ScriptedLLM(handlers))
    c, o = _setup(db, factory)
    t = svc.ingest("chat", chat_payload(c.email, "The lamp arrived cracked.", o.id))
    res = svc.resolve(t.id)
    assert res.outcome == Outcome.ESCALATED
    assert "invalid output" in res.proposed.reasons[0]


def test_completeness_guard_forces_required_tools(db, factory):
    handlers = dict(OFFLINE_HANDLERS)
    handlers["next_action"] = lambda ctx: {"thought": "lazy", "finish": True}
    svc = ResolveService(db, llm=ScriptedLLM(handlers))
    c, o = _setup(db, factory)
    t = svc.ingest("chat", chat_payload(c.email, "The lamp arrived cracked.", o.id))
    res = svc.resolve(t.id)
    names = [c.name for c in res.tool_calls]
    assert names == ["fetch_order_details", "analyze_issue"]
    # No fraud check / policy lookup -> gate refuses to automate a refund.
    assert res.outcome == Outcome.ESCALATED
    assert "fraud_check_missing" in res.proposed.reasons or res.guardrail_flags


def test_fraud_signals_escalate(service, db, factory):
    c, _ = _setup(db, factory, age=4)
    o2 = factory.order(
        c, items=[factory.item("HM-330")], days_ago=2, delivery_days=1, billing_country="ES"
    )
    db.add_order(o2)
    t = service.ingest("chat", chat_payload(c.email, "Lamp is broken, refund now.", o2.id))
    res = service.resolve(t.id)
    assert res.outcome == Outcome.ESCALATED
    assert any(f.startswith("fraud_risk") for f in res.guardrail_flags)


def test_human_decision_executes_refund_and_closes(service, db, factory):
    c, o = _setup(db, factory, sku="EL-140")
    t = service.ingest("chat", chat_payload(c.email, "The laptop won't turn on.", o.id))
    res = service.resolve(t.id)
    out = service.human_decide(
        res.escalation_id,
        action=DecisionAction.APPROVE_REFUND,
        amount=1149.0,
        agent="agent-7",
        notes="Verified with photos",
    )
    assert out["refund_id"]
    assert db.get_escalation(res.escalation_id)["status"] == "resolved"
    assert db.get_ticket(t.id)[1] == "resolved"
    with pytest.raises(ValueError):
        service.human_decide(res.escalation_id, action=DecisionAction.DENY, amount=0, agent="x")


def test_trace_is_json_serialisable(service, db, factory):
    c, o = _setup(db, factory)
    t = service.ingest("chat", chat_payload(c.email, "The lamp arrived cracked.", o.id))
    res = service.resolve(t.id)
    data = json.loads(res.model_dump_json())
    kinds = [e["kind"] for e in data["trace"]]
    assert kinds[0] == "intake" and kinds[-1] == "action" and "guardrail" in kinds
    assert res.simulated_latency_ms > 0 and res.llm_calls >= 2


def test_llm_factory_falls_back_to_offline_without_key(monkeypatch):
    from resolveai.llm import get_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), ScriptedLLM)
    with pytest.raises(ValueError):
        get_llm("not-a-provider")
