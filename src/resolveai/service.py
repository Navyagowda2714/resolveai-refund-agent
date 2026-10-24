"""Application service shared by the API and CLI: intake, resolution, human decisions."""

from __future__ import annotations

from typing import Any

from resolveai.actions.services import PaymentError
from resolveai.agent.agent import ResolutionAgent
from resolveai.config import Settings, get_settings
from resolveai.data.db import CommerceDB
from resolveai.intake import normalise
from resolveai.llm import LLMClient, get_llm
from resolveai.models import DecisionAction, Resolution, Ticket
from resolveai.policy import BM25Index, Policy, load_sections


class NotFoundError(LookupError):
    pass


class ResolveService:
    def __init__(
        self,
        db: CommerceDB,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        enforce_guardrails: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db
        self.policy = Policy.load(self.settings.policy_path)
        self.index = BM25Index(load_sections(self.settings.policy_doc_path))
        self.agent = ResolutionAgent(
            db,
            llm or get_llm(),
            self.policy,
            self.index,
            max_tool_calls=self.settings.max_tool_calls,
            enforce_guardrails=self.settings.enforce_guardrails
            if enforce_guardrails is None
            else enforce_guardrails,
        )

    def ingest(self, channel: str, payload: dict[str, Any]) -> Ticket:
        ticket = normalise(channel, payload)
        self.db.save_ticket(ticket)
        return ticket

    def resolve(self, ticket_id: str) -> Resolution:
        found = self.db.get_ticket(ticket_id)
        if found is None:
            raise NotFoundError(ticket_id)
        ticket, status = found
        if status != "open":
            existing = self.db.get_resolution(ticket_id)
            if existing is not None:
                return existing  # idempotent: never double-process a ticket
        resolution = self.agent.resolve(ticket)
        self.db.save_resolution(resolution)
        return resolution

    def human_decide(
        self, esc_id: str, *, action: DecisionAction, amount: float, agent: str, notes: str = ""
    ) -> dict[str, Any]:
        esc = self.db.get_escalation(esc_id)
        if esc is None:
            raise NotFoundError(esc_id)
        if esc["status"] != "open":
            raise ValueError(f"escalation {esc_id} is already {esc['status']}")
        found = self.db.get_ticket(esc["ticket_id"])
        if found is None:
            raise NotFoundError(esc["ticket_id"])
        ticket, _ = found
        order_id = ticket.order_id or (esc["packet"]["findings"]["order"].get("order") or {}).get(
            "id"
        )
        result: dict[str, Any] = {"escalation_id": esc_id, "action": action.value, "amount": amount}
        if action in (DecisionAction.APPROVE_REFUND, DecisionAction.PARTIAL_REFUND):
            order = self.db.get_order(order_id) if order_id else None
            if order is None:
                raise ValueError("cannot refund: no order on this ticket")
            try:
                result["refund_id"] = self.agent.payments.refund(
                    order_id=order.id,
                    customer_id=order.customer_id,
                    amount=amount,
                    reason=f"human:{agent}",
                    idempotency_key=f"human:{esc_id}",
                )
            except PaymentError as exc:
                raise ValueError(str(exc)) from exc
            self.agent.notifier.notify(
                ticket,
                action.value,
                name="there",
                order_id=order.id,
                amount=f"{amount:.2f}",
                currency=order.currency,
                reason_line=notes,
            )
        elif action == DecisionAction.DENY:
            self.agent.notifier.notify(
                ticket, "deny", name="there", order_id=order_id or "-", reason_line=notes
            )
        else:
            raise ValueError("a human decision must be approve_refund, partial_refund or deny")
        self.db.close_escalation(esc_id, by=agent, action=action.value, amount=amount, notes=notes)
        self.db.set_ticket_status(ticket.id, "resolved")
        return result
