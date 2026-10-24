"""Business rules engine: deterministic policy evaluation from YAML.

The rules engine answers "what does the policy say?" (eligibility, refundable amount, mandatory
escalations). It deliberately does NOT decide autonomy; that is the safety gate's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from resolveai.models import DecisionAction, Order
from resolveai.tools.schemas import IssueAnalysis, OrderDetails


class AutoResolution(BaseModel):
    max_auto_refund_amount: float
    min_confidence: float
    min_deny_confidence: float
    max_fraud_score: int


class Windows(BaseModel):
    default: int
    categories: dict[str, int]
    warranty_days: int


class Policy(BaseModel):
    version: str
    currency: str
    auto_resolution: AutoResolution
    windows: Windows
    non_refundable_categories: list[str]
    final_sale_refundable_issues: list[str]
    issues: dict[str, dict[str, Any]]
    fraud: dict[str, dict[str, int]]
    escalation_triggers: list[str]

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def window_for(self, category: str) -> int:
        return self.windows.categories.get(category, self.windows.default)

    def issue_rule(self, issue_type: str) -> dict[str, Any]:
        return self.issues.get(issue_type, self.issues["other"])


class RuleEvaluation(BaseModel):
    eligible: bool
    recommended_action: DecisionAction
    refund_amount: float = 0.0
    deny_reasons: list[str] = Field(default_factory=list)
    escalate_reasons: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _affected_amount(order: Order, analysis: IssueAnalysis) -> tuple[float, bool]:
    """Amount for the affected items and whether the item attribution was ambiguous."""
    if analysis.affected_skus:
        items = [i for i in order.items if i.sku in analysis.affected_skus]
        return round(sum(i.line_total for i in items), 2), False
    distinct = {i.sku for i in order.items}
    if len(distinct) == 1:
        return order.items_total, False
    return order.items_total, True


class RulesEngine:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def evaluate(self, *, analysis: IssueAnalysis, order: OrderDetails) -> RuleEvaluation:
        p = self.policy
        issue = analysis.issue_type
        rule = p.issue_rule(issue)
        ev = RuleEvaluation(eligible=False, recommended_action=DecisionAction.ESCALATE)
        cite = ev.citations.append

        def escalate(reason: str, citation: str) -> RuleEvaluation:
            ev.escalate_reasons.append(reason)
            cite(citation)
            ev.recommended_action = DecisionAction.ESCALATE
            ev.refund_amount = 0.0
            return ev

        def deny(reason: str, citation: str) -> RuleEvaluation:
            ev.deny_reasons.append(reason)
            cite(citation)
            ev.recommended_action = DecisionAction.DENY
            ev.refund_amount = 0.0
            return ev

        # Mandatory human review triggers come first: they override everything.
        triggers = [t for t in analysis.triggers if t in p.escalation_triggers]
        if triggers:
            return escalate(f"mandatory review trigger(s): {', '.join(triggers)}", "POL-7.1")
        if rule.get("escalate"):
            return escalate(f"issue type '{issue}' always requires a human", rule["citation"])
        if not order.found or order.order is None:
            return escalate("no matching order found for this customer", "POL-1.2")
        if not order.belongs_to_customer:
            return escalate("order belongs to a different account (identity mismatch)", "POL-1.2")

        o = order.order
        categories = {i.category for i in o.items}
        if order.remaining_refundable <= 0.01:
            return deny("order has already been refunded in full", "POL-6.1")
        if categories and categories <= set(p.non_refundable_categories):
            return deny(f"category {sorted(categories)} is non-refundable", "POL-2.3")
        if any(i.final_sale for i in o.items) and issue not in p.final_sale_refundable_issues:
            return deny("final-sale item is not returnable for this reason", "POL-2.2")

        # Delivery-related issues.
        if issue == "not_received":
            if o.status == "delivered" and o.tracking_confirmed:
                return escalate("carrier confirms delivery: carrier dispute", "POL-4.1")
            grace = int(rule.get("in_transit_grace_days", 10))
            if o.status == "shipped" and (order.days_since_order or 0) <= grace:
                return deny(f"order is still in transit (within {grace}-day grace)", "POL-4.2")
            amount = o.total
        else:
            if o.delivered_at is None:
                return escalate(f"'{issue}' reported but order is not delivered yet", "POL-8.1")
            window = max(p.window_for(c) for c in categories)
            age = order.days_since_delivery or 0
            if age > window:
                if issue == "defective" and age <= p.windows.warranty_days:
                    return escalate("defect outside return window: warranty claim", "POL-3.2")
                return deny(
                    f"request is {age} days after delivery (window {window} days)", "POL-1.1"
                )
            cite("POL-1.1")

            refund_kind = rule.get("refund")
            if refund_kind == "affected_items":
                amount, ambiguous = _affected_amount(o, analysis)
                if ambiguous:
                    if issue == "missing_item":
                        return escalate("could not identify which item is missing", "POL-3.4")
                    ev.notes.append("affected item not identified; assuming all items")
            elif refund_kind == "restocking_fee":
                pct = float(rule.get("restocking_fee_pct", 15)) / 100
                base, _ = _affected_amount(o, analysis)
                amount = round(base * (1 - pct), 2)
                ev.notes.append(f"{int(pct * 100)}% restocking fee applied")
            elif refund_kind == "shipping_or_goodwill":
                if o.shipping_cost > 0:
                    amount = o.shipping_cost
                    ev.notes.append("shipping fee refunded for late delivery")
                else:
                    amount = round(o.items_total * float(rule.get("goodwill_pct", 10)) / 100, 2)
                    ev.notes.append("goodwill credit (free shipping order)")
            else:  # pragma: no cover - policy misconfiguration
                return escalate(f"no refund rule for issue '{issue}'", "POL-8.1")

        if amount > order.remaining_refundable:
            ev.notes.append(f"capped at remaining refundable {order.remaining_refundable:.2f}")
            amount = order.remaining_refundable
        cite(rule.get("citation", "POL-8.1"))
        ev.eligible = True
        ev.refund_amount = round(amount, 2)
        full = o.total if issue == "not_received" else o.items_total
        ev.recommended_action = (
            DecisionAction.APPROVE_REFUND
            if ev.refund_amount >= full - 0.01
            else DecisionAction.PARTIAL_REFUND
        )
        return ev
