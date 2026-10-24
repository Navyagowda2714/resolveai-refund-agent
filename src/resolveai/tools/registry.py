"""Investigation tools exposed to the agent via JSON-schema function definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ValidationError

from resolveai.data.db import CommerceDB
from resolveai.models import ToolCall
from resolveai.policy.engine import Policy
from resolveai.policy.retrieval import BM25Index
from resolveai.tools import analyzer, fraud
from resolveai.tools.schemas import (
    AnalyzeInput,
    CustomerHistory,
    FetchOrderInput,
    FraudInput,
    HistoryInput,
    OrderDetails,
    PolicyInput,
    PolicyLookup,
)


@dataclass
class ToolContext:
    db: CommerceDB
    policy: Policy
    index: BM25Index
    describer: analyzer.ImageDescriber = field(default_factory=analyzer.AltTextDescriber)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], BaseModel]
    simulated_latency_ms: float

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


# --------------------------------------------------------------------------- implementations
def _history(ctx: ToolContext, email: str) -> CustomerHistory:
    cust = ctx.db.get_customer_by_email(email)
    if cust is None:
        return CustomerHistory(found=False)
    orders = ctx.db.orders_for_customer(cust.id)
    refunds = ctx.db.refunds_for_customer(cust.id)
    spend = round(sum(o.total for o in orders), 2)
    recent = [r for r in refunds if r.created_at >= ctx.as_of - timedelta(days=90)]
    return CustomerHistory(
        found=True,
        customer_id=cust.id,
        tier=cust.tier,
        account_age_days=max(0, (ctx.as_of - cust.created_at).days),
        total_orders=len(orders),
        total_spend=spend,
        avg_order_value=round(spend / len(orders), 2) if orders else 0.0,
        refunds_total=len(refunds),
        refunds_last_90d=len(recent),
        refund_rate=round(len(refunds) / len(orders), 3) if orders else 0.0,
    )


def fetch_order_details(ctx: ToolContext, inp: FetchOrderInput) -> OrderDetails:
    cust = ctx.db.get_customer_by_email(inp.customer_email)
    order = ctx.db.get_order(inp.order_id) if inp.order_id else None
    inferred = False
    if order is None and inp.order_id is None and cust is not None:
        recent = ctx.db.orders_for_customer(cust.id)
        order, inferred = (recent[0], True) if recent else (None, False)
    if order is None:
        return OrderDetails(found=False)
    refunded = ctx.db.refunded_amount_for_order(order.id)
    return OrderDetails(
        found=True,
        order=order,
        inferred_from_history=inferred,
        belongs_to_customer=cust is not None and order.customer_id == cust.id,
        days_since_delivery=(ctx.as_of - order.delivered_at).days if order.delivered_at else None,
        days_since_order=(ctx.as_of - order.ordered_at).days,
        previously_refunded=refunded,
        remaining_refundable=round(max(0.0, order.total - refunded), 2),
    )


def check_refund_policy(ctx: ToolContext, inp: PolicyInput) -> PolicyLookup:
    query = inp.query or f"{inp.category} {inp.issue_type.replace('_', ' ')} refund"
    sections = [s.model_dump() for s in ctx.index.search(query, k=3)]
    return PolicyLookup(
        category=inp.category,
        issue_type=inp.issue_type,
        window_days=ctx.policy.window_for(inp.category),
        category_refundable=inp.category not in ctx.policy.non_refundable_categories,
        issue_rule=ctx.policy.issue_rule(inp.issue_type),
        sections=sections,
    )


def review_customer_history(ctx: ToolContext, inp: HistoryInput) -> CustomerHistory:
    return _history(ctx, inp.customer_email)


def run_fraud_checks(ctx: ToolContext, inp: FraudInput) -> BaseModel:
    order = ctx.db.get_order(inp.order_id) if inp.order_id else None
    return fraud.run_fraud_checks(
        _history(ctx, inp.customer_email),
        order,
        claimed_amount=inp.claimed_amount,
        issue_type=inp.issue_type,
        levels=ctx.policy.fraud.get("levels"),
    )


def analyze_issue(ctx: ToolContext, inp: AnalyzeInput) -> BaseModel:
    return analyzer.analyze(inp.text, inp.attachments, inp.order_items, ctx.describer)


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "fetch_order_details",
            "Look up an order (or the customer's latest order) with "
            "delivery status, timing and refundable balance.",
            FetchOrderInput,
            fetch_order_details,
            120.0,
        ),
        Tool(
            "analyze_issue",
            "Classify the customer's issue from text and image descriptions; "
            "detect affected items and escalation triggers.",
            AnalyzeInput,
            analyze_issue,
            90.0,
        ),
        Tool(
            "review_customer_history",
            "Summarise account age, order and refund history.",
            HistoryInput,
            review_customer_history,
            150.0,
        ),
        Tool(
            "run_fraud_checks",
            "Score refund fraud risk from frequency, account age, amount vs "
            "history, address mismatch and delivery contradictions.",
            FraudInput,
            run_fraud_checks,
            220.0,
        ),
        Tool(
            "check_refund_policy",
            "Retrieve the applicable refund policy rules and sections.",
            PolicyInput,
            check_refund_policy,
            60.0,
        ),
    ]
}


def execute_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> ToolCall:
    """Validate arguments, run a tool, and capture result/error (never raises)."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolCall(name=name, arguments=arguments, error=f"unknown tool '{name}'")
    try:
        inp = tool.input_model.model_validate(arguments)
    except ValidationError as exc:
        return ToolCall(name=name, arguments=arguments, error=f"invalid arguments: {exc.errors()}")
    try:
        result = tool.fn(ctx, inp)
    except Exception as exc:  # tool failures are observations, not crashes
        return ToolCall(name=name, arguments=arguments, error=f"{type(exc).__name__}: {exc}")
    return ToolCall(
        name=name,
        arguments=arguments,
        result=result.model_dump(mode="json"),
        simulated_latency_ms=tool.simulated_latency_ms,
    )
