"""Core domain models (pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Channel(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"
    IN_APP = "in_app"
    PORTAL = "portal"


class Attachment(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    description: str | None = Field(
        default=None, description="Alt text / vision-model caption for images, if available."
    )


class Ticket(BaseModel):
    """Channel-agnostic customer ticket produced by an intake adapter."""

    id: str
    channel: Channel
    customer_email: str
    order_id: str | None = None
    subject: str = ""
    body: str
    attachments: list[Attachment] = Field(default_factory=list)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()


class OrderItem(BaseModel):
    sku: str
    name: str
    category: str
    unit_price: float
    quantity: int = 1
    final_sale: bool = False

    @property
    def line_total(self) -> float:
        return round(self.unit_price * self.quantity, 2)


class Address(BaseModel):
    line1: str
    city: str
    postcode: str
    country: str


class Order(BaseModel):
    id: str
    customer_id: str
    items: list[OrderItem]
    shipping_cost: float = 0.0
    currency: str = "EUR"
    status: str = "delivered"  # processing | shipped | delivered | cancelled
    ordered_at: datetime
    delivered_at: datetime | None = None
    tracking_confirmed: bool = True
    shipping_address: Address
    billing_address: Address
    payment_method: str = "card"

    @property
    def items_total(self) -> float:
        return round(sum(i.line_total for i in self.items), 2)

    @property
    def total(self) -> float:
        return round(self.items_total + self.shipping_cost, 2)


class Customer(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime
    tier: str = "standard"  # standard | plus | vip


class RefundRecord(BaseModel):
    id: str
    customer_id: str
    order_id: str
    amount: float
    reason: str
    created_at: datetime
    status: str = "completed"


class DecisionAction(str, Enum):
    APPROVE_REFUND = "approve_refund"
    PARTIAL_REFUND = "partial_refund"
    DENY = "deny"
    ESCALATE = "escalate"


class Decision(BaseModel):
    """Structured decision proposed by the LLM reasoning step."""

    action: DecisionAction
    amount: float = Field(default=0.0, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    simulated_latency_ms: float = 0.0


class TraceEvent(BaseModel):
    step: int
    kind: str  # intake | thought | tool_call | rules | decision | guardrail | action
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class Outcome(str, Enum):
    AUTO_APPROVED = "auto_approved"
    AUTO_PARTIAL = "auto_partial"
    AUTO_DENIED = "auto_denied"
    ESCALATED = "escalated"


class Resolution(BaseModel):
    ticket_id: str
    outcome: Outcome
    final_action: DecisionAction
    automated: bool
    proposed: Decision
    amount_refunded: float = 0.0
    guardrail_flags: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    customer_message: str = ""
    refund_id: str | None = None
    escalation_id: str | None = None
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    simulated_latency_ms: float = 0.0
