import pytest

from conftest import ANCHOR
from resolveai.config import DEFAULT_POLICY_DOC_PATH, DEFAULT_POLICY_PATH
from resolveai.models import Attachment, DecisionAction
from resolveai.policy import BM25Index, Policy, RulesEngine, load_sections
from resolveai.tools.analyzer import analyze
from resolveai.tools.fraud import run_fraud_checks
from resolveai.tools.schemas import CustomerHistory, IssueAnalysis, OrderDetails

POLICY = Policy.load(DEFAULT_POLICY_PATH)
RULES = RulesEngine(POLICY)


def details(order, refunded=0.0, owner=True):
    return OrderDetails(
        found=True,
        order=order,
        belongs_to_customer=owner,
        days_since_delivery=(ANCHOR - order.delivered_at).days if order.delivered_at else None,
        days_since_order=(ANCHOR - order.ordered_at).days,
        previously_refunded=refunded,
        remaining_refundable=round(order.total - refunded, 2),
    )


def issue(kind, skus=(), triggers=()):
    return IssueAnalysis(
        issue_type=kind, confidence=0.9, affected_skus=list(skus), triggers=list(triggers)
    )


@pytest.fixture
def cust(factory):
    return factory.customer(account_age_days=500)


def test_damaged_in_window_full_refund(factory, cust):
    o = factory.order(cust, items=[factory.item("HM-330")], days_ago=5)
    ev = RULES.evaluate(analysis=issue("damaged"), order=details(o))
    assert ev.recommended_action == DecisionAction.APPROVE_REFUND
    assert ev.refund_amount == 54.0 and "POL-3.1" in ev.citations


def test_outside_window_denied(factory, cust):
    o = factory.order(cust, items=[factory.item("BT-400")], days_ago=40)
    ev = RULES.evaluate(analysis=issue("changed_mind"), order=details(o))
    assert ev.recommended_action == DecisionAction.DENY and "POL-1.1" in ev.citations


def test_final_sale_change_of_mind_denied_but_damage_allowed(factory, cust):
    o = factory.order(cust, items=[factory.item("AP-230")], days_ago=5)
    assert RULES.evaluate(analysis=issue("changed_mind"), order=details(o)).deny_reasons
    assert RULES.evaluate(analysis=issue("damaged"), order=details(o)).eligible


def test_change_of_mind_restocking_fee(factory, cust):
    o = factory.order(cust, items=[factory.item("AP-200")], days_ago=10)
    ev = RULES.evaluate(analysis=issue("changed_mind"), order=details(o))
    assert ev.recommended_action == DecisionAction.PARTIAL_REFUND
    assert ev.refund_amount == pytest.approx(79.0 * 0.85)


def test_not_received_confirmed_is_carrier_dispute(factory, cust):
    o = factory.order(cust, items=[factory.item("EL-100")], days_ago=8)
    ev = RULES.evaluate(analysis=issue("not_received"), order=details(o))
    assert ev.recommended_action == DecisionAction.ESCALATE


def test_in_transit_denied_and_lost_refunded(factory, cust):
    fresh = factory.order(
        cust, items=[factory.item("BK-700")], days_ago=3, status="shipped", tracking_confirmed=False
    )
    assert RULES.evaluate(analysis=issue("not_received"), order=details(fresh)).deny_reasons
    lost = factory.order(
        cust,
        items=[factory.item("BK-700")],
        days_ago=20,
        status="shipped",
        tracking_confirmed=False,
    )
    ev = RULES.evaluate(analysis=issue("not_received"), order=details(lost))
    assert ev.refund_amount == lost.total  # includes shipping


def test_partial_missing_item_and_identity_and_duplicate(factory, cust):
    o = factory.order(cust, items=[factory.item("AP-220"), factory.item("EL-130")], days_ago=5)
    ev = RULES.evaluate(analysis=issue("missing_item", ["EL-130"]), order=details(o))
    assert (ev.recommended_action, ev.refund_amount) == (DecisionAction.PARTIAL_REFUND, 39.0)
    assert RULES.evaluate(analysis=issue("missing_item"), order=details(o)).escalate_reasons
    assert RULES.evaluate(analysis=issue("damaged"), order=details(o, owner=False)).escalate_reasons
    assert RULES.evaluate(
        analysis=issue("damaged"), order=details(o, refunded=o.total)
    ).deny_reasons


def test_triggers_override_everything(factory, cust):
    o = factory.order(cust, items=[factory.item("HM-330")], days_ago=5)
    ev = RULES.evaluate(analysis=issue("damaged", triggers=["legal_threat"]), order=details(o))
    assert ev.recommended_action == DecisionAction.ESCALATE and "POL-7.1" in ev.citations


def test_bm25_policy_retrieval():
    idx = BM25Index(load_sections(DEFAULT_POLICY_DOC_PATH))
    assert idx.search("carrier confirms delivery not received")[0].id == "POL-4.1"
    assert idx.search("final sale clearance")[0].id == "POL-2.2"


def test_analyzer_negation_triggers_and_clause_attribution():
    a = analyze("It's not broken, I just don't like the colour")
    assert a.issue_type == "changed_mind"
    items = [{"sku": "A", "name": "Running Shoes"}, {"sku": "B", "name": "USB-C Charging Hub"}]
    a = analyze("The shoes arrived fine but the charging hub wasn't in the box", order_items=items)
    assert a.issue_type == "missing_item" and a.affected_skus == ["B"]
    assert "chargeback_threat" in analyze("I will open a dispute with my bank").triggers
    assert analyze("hello there").issue_type == "other"


def test_image_description_hook_adds_evidence():
    att = Attachment(filename="p.jpg", content_type="image/jpeg", description="shattered glass")
    with_img = analyze("see photo", attachments=[att])
    assert with_img.issue_type == "damaged" and with_img.image_findings


def test_fraud_scoring_signals(factory, cust):
    o = factory.order(cust, items=[factory.item("EL-100")], days_ago=3, billing_country="ES")
    hist = CustomerHistory(found=True, account_age_days=5, total_orders=1, refunds_last_90d=3)
    rep = run_fraud_checks(
        hist, o, claimed_amount=89.0, issue_type="not_received", levels=POLICY.fraud["levels"]
    )
    names = {s.name for s in rep.signals}
    assert {"refund_frequency", "new_account", "delivery_contradiction"} <= names
    assert rep.score == min(100, sum(s.weight for s in rep.signals)) == 100
    assert rep.level == "high"
    clean = run_fraud_checks(
        CustomerHistory(found=True, account_age_days=900, total_orders=5, avg_order_value=80),
        None,
        claimed_amount=60,
        issue_type="damaged",
    )
    assert clean.score == 0 and clean.level == "low"


def test_policy_windows():
    assert POLICY.window_for("apparel") == 60 and POLICY.window_for("unknown") == 30
    assert POLICY.auto_resolution.max_auto_refund_amount == 150
