"""Typed input/output schemas for the investigation tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from resolveai.models import Attachment, Order


# ----------------------------------------------------------------------------- inputs
class FetchOrderInput(BaseModel):
    order_id: str | None = Field(None, description="Order id like ORD-10234, if known.")
    customer_email: str = Field(description="Email of the requesting customer.")


class PolicyInput(BaseModel):
    category: str = Field(description="Product category of the affected item(s).")
    issue_type: str = Field(description="Issue type, e.g. damaged, not_received, changed_mind.")
    query: str = Field("", description="Free-text query used to retrieve policy sections.")


class HistoryInput(BaseModel):
    customer_email: str


class FraudInput(BaseModel):
    customer_email: str
    order_id: str | None = None
    claimed_amount: float = Field(0.0, ge=0)
    issue_type: str = "other"


class AnalyzeInput(BaseModel):
    text: str
    attachments: list[Attachment] = Field(default_factory=list)
    order_items: list[dict[str, Any]] = Field(
        default_factory=list, description="[{sku, name}] of the order, to match item mentions."
    )


# ----------------------------------------------------------------------------- outputs
class OrderDetails(BaseModel):
    found: bool
    order: Order | None = None
    inferred_from_history: bool = False
    belongs_to_customer: bool = False
    days_since_delivery: int | None = None
    days_since_order: int | None = None
    previously_refunded: float = 0.0
    remaining_refundable: float = 0.0


class PolicyLookup(BaseModel):
    category: str
    issue_type: str
    window_days: int
    category_refundable: bool
    issue_rule: dict[str, Any]
    sections: list[dict[str, Any]]


class CustomerHistory(BaseModel):
    found: bool
    customer_id: str | None = None
    tier: str = "standard"
    account_age_days: int = 0
    total_orders: int = 0
    total_spend: float = 0.0
    avg_order_value: float = 0.0
    refunds_total: int = 0
    refunds_last_90d: int = 0
    refund_rate: float = 0.0


class FraudSignal(BaseModel):
    name: str
    weight: int
    detail: str


class FraudReport(BaseModel):
    score: int = Field(ge=0, le=100)
    level: str  # low | medium | high
    signals: list[FraudSignal] = Field(default_factory=list)


class IssueAnalysis(BaseModel):
    issue_type: str
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)
    secondary_issues: list[str] = Field(default_factory=list)
    affected_skus: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    requested_resolution: str = "unspecified"
    image_findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
