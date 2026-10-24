"""ResolutionAgent: investigate -> evaluate policy -> decide -> gate -> act."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from resolveai.actions.services import (
    HumanQueue,
    MockPaymentsService,
    NotificationService,
    PaymentError,
)
from resolveai.agent.guardrails import REFUND_ACTIONS, GateResult, SafetyGate
from resolveai.agent.prompts import (
    DECIDE_SYSTEM,
    INVESTIGATE_SYSTEM,
    NEXT_ACTION_SCHEMA,
    NextAction,
)
from resolveai.data.db import CommerceDB
from resolveai.llm.base import LLMClient, LLMResponse
from resolveai.models import (
    Decision,
    DecisionAction,
    Outcome,
    Resolution,
    Ticket,
    ToolCall,
    TraceEvent,
)
from resolveai.policy.engine import Policy, RuleEvaluation, RulesEngine
from resolveai.policy.retrieval import BM25Index
from resolveai.tools.analyzer import ImageDescriber
from resolveai.tools.registry import TOOLS, ToolContext, execute_tool
from resolveai.tools.schemas import (
    CustomerHistory,
    FraudReport,
    IssueAnalysis,
    OrderDetails,
    PolicyLookup,
)

logger = logging.getLogger(__name__)


class Findings:
    """Typed view over the successful tool results of one investigation."""

    def __init__(self, calls: list[ToolCall]) -> None:
        latest = {c.name: c.result for c in calls if c.result is not None}
        self.order = OrderDetails.model_validate(
            latest.get("fetch_order_details", {"found": False})
        )
        self.analysis = (
            IssueAnalysis.model_validate(latest["analyze_issue"])
            if "analyze_issue" in latest
            else IssueAnalysis(issue_type="other", confidence=0.0)
        )
        self.history = (
            CustomerHistory.model_validate(latest["review_customer_history"])
            if "review_customer_history" in latest
            else None
        )
        self.fraud = (
            FraudReport.model_validate(latest["run_fraud_checks"])
            if "run_fraud_checks" in latest
            else None
        )
        self.policy = (
            PolicyLookup.model_validate(latest["check_refund_policy"])
            if "check_refund_policy" in latest
            else None
        )

    def as_context(self) -> dict[str, Any]:
        o = self.order
        return {
            "order": o.model_dump(mode="json") if o.found else {"found": False},
            "analysis": self.analysis.model_dump(),
            "history": self.history.model_dump() if self.history else None,
            "fraud": self.fraud.model_dump() if self.fraud else None,
            "policy": self.policy.model_dump() if self.policy else None,
        }


class ResolutionAgent:
    REQUIRED_TOOLS = ("fetch_order_details", "analyze_issue")

    def __init__(
        self,
        db: CommerceDB,
        llm: LLMClient,
        policy: Policy,
        index: BM25Index,
        *,
        max_tool_calls: int = 8,
        enforce_guardrails: bool = True,
        describer: ImageDescriber | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.policy = policy
        self.index = index
        self.rules = RulesEngine(policy)
        self.gate = SafetyGate(policy, enforce=enforce_guardrails)
        self.max_tool_calls = max_tool_calls
        self.describer = describer
        self.payments = MockPaymentsService(db)
        self.notifier = NotificationService(db)
        self.queue = HumanQueue(db)
        self._llm_usage: list[LLMResponse] = []

    # ------------------------------------------------------------------ helpers
    def _ctx(self, ticket: Ticket) -> ToolContext:
        ctx = ToolContext(db=self.db, policy=self.policy, index=self.index, as_of=ticket.created_at)
        if self.describer is not None:
            ctx.describer = self.describer
        return ctx

    @staticmethod
    def _ticket_view(ticket: Ticket) -> dict[str, Any]:
        return ticket.model_dump(
            mode="json",
            include={
                "id",
                "channel",
                "customer_email",
                "order_id",
                "subject",
                "body",
                "attachments",
            },
        )

    def _ask(self, system: str, payload: dict[str, Any], schema: dict[str, Any]) -> LLMResponse:
        content = "```json\n" + json.dumps(payload, default=str) + "\n```"
        resp = self.llm.complete(
            [{"role": "user", "content": content}], system=system, json_schema=schema
        )
        self._llm_usage.append(resp)
        return resp

    # ------------------------------------------------------------------ main entry point
    def resolve(self, ticket: Ticket) -> Resolution:
        self._llm_usage: list[LLMResponse] = []
        trace: list[TraceEvent] = []

        def log(kind: str, summary: str, **data: Any) -> None:
            trace.append(TraceEvent(step=len(trace) + 1, kind=kind, summary=summary, data=data))

        log(
            "intake",
            f"{ticket.channel.value} ticket from {ticket.customer_email}",
            order_id=ticket.order_id,
            attachments=len(ticket.attachments),
        )

        calls = self._investigate(ticket, log)
        findings = Findings(calls)
        rules = self.rules.evaluate(analysis=findings.analysis, order=findings.order)
        log(
            "rules",
            f"policy -> {rules.recommended_action.value} {rules.refund_amount:.2f}",
            **rules.model_dump(mode="json"),
        )

        proposed = self._decide(ticket, findings, rules, log)
        gate = self.gate.check(proposed, rules, findings.fraud, findings.policy)
        log(
            "guardrail",
            "automate" if gate.automate else f"escalate: {', '.join(gate.flags)}",
            flags=gate.flags,
        )

        resolution = self._act(ticket, proposed, rules, gate, findings, log)
        resolution.tool_calls = calls
        resolution.trace = trace
        resolution.llm_calls = len(self._llm_usage)
        resolution.llm_input_tokens = sum(r.input_tokens for r in self._llm_usage)
        resolution.llm_output_tokens = sum(r.output_tokens for r in self._llm_usage)
        resolution.simulated_latency_ms = round(
            sum(c.simulated_latency_ms for c in calls) + sum(r.latency_ms for r in self._llm_usage),
            1,
        )
        return resolution

    # ------------------------------------------------------------------ 1. investigation loop
    def _investigate(self, ticket: Ticket, log: Any) -> list[ToolCall]:
        ctx = self._ctx(ticket)
        calls: list[ToolCall] = []
        seen: set[str] = set()
        tool_specs = [t.json_schema() for t in TOOLS.values()]
        for _ in range(self.max_tool_calls):
            payload = {
                "ticket": self._ticket_view(ticket),
                "tools": tool_specs,
                "observations": [
                    {"tool": c.name, "arguments": c.arguments, "result": c.result, "error": c.error}
                    for c in calls
                ],
            }
            try:
                action = NextAction.model_validate(
                    self._ask(INVESTIGATE_SYSTEM, payload, NEXT_ACTION_SCHEMA).json_payload()
                )
            except (ValidationError, ValueError) as exc:
                log("thought", f"unparseable model output, stopping investigation: {exc}")
                break
            if action.thought:
                log("thought", action.thought)
            if action.finish or not action.tool:
                break
            key = action.tool + json.dumps(action.arguments, sort_keys=True, default=str)
            if key in seen:
                log("thought", f"duplicate call to {action.tool} suppressed; finishing")
                break
            seen.add(key)
            call = execute_tool(ctx, action.tool, action.arguments)
            calls.append(call)
            self._log_call(call, log)

        # Completeness guard: the decision step always sees order + issue analysis.
        done = {c.name for c in calls if c.result is not None}
        for name in self.REQUIRED_TOOLS:
            if name in done:
                continue
            args: dict[str, Any]
            if name == "fetch_order_details":
                args = {"order_id": ticket.order_id, "customer_email": ticket.customer_email}
            else:
                order = Findings(calls).order.order
                args = {
                    "text": ticket.text,
                    "attachments": [a.model_dump() for a in ticket.attachments],
                    "order_items": [{"sku": i.sku, "name": i.name} for i in order.items]
                    if order
                    else [],
                }
            call = execute_tool(ctx, name, args)
            calls.append(call)
            log("guardrail", f"completeness guard forced {name}")
            self._log_call(call, log)
        return calls

    @staticmethod
    def _log_call(call: ToolCall, log: Any) -> None:
        if call.error:
            log("tool_call", f"{call.name} failed: {call.error}", arguments=call.arguments)
            return
        r = call.result or {}
        summary = {
            "fetch_order_details": lambda: (
                f"order {r['order']['id']} ({r['order']['status']}, "
                f"{r.get('days_since_delivery')}d since delivery, "
                f"refundable {r['remaining_refundable']:.2f})"
                if r.get("found")
                else "order not found"
            ),
            "analyze_issue": lambda: (
                f"issue={r['issue_type']} conf={r['confidence']:.2f} "
                f"items={r['affected_skus'] or '-'} triggers={r['triggers'] or '-'}"
            ),
            "review_customer_history": lambda: (
                f"account {r['account_age_days']}d, {r['total_orders']} orders, "
                f"{r['refunds_last_90d']} refunds/90d"
                if r.get("found")
                else "customer not found"
            ),
            "run_fraud_checks": lambda: (
                f"score={r['score']} ({r['level']}) "
                f"{[s['name'] for s in r['signals']] or 'no signals'}"
            ),
            "check_refund_policy": lambda: (
                f"window={r['window_days']}d, sections={[s['id'] for s in r['sections']]}"
            ),
        }.get(call.name, lambda: "ok")()
        log(
            "tool_call",
            f"{call.name}: {summary}",
            arguments=call.arguments,
            simulated_latency_ms=call.simulated_latency_ms,
        )

    # ------------------------------------------------------------------ 2. decision
    def _decide(
        self, ticket: Ticket, findings: Findings, rules: RuleEvaluation, log: Any
    ) -> Decision:
        payload = {
            "ticket": self._ticket_view(ticket),
            "findings": findings.as_context(),
            "policy_evaluation": rules.model_dump(mode="json"),
            "autonomy_limits": self.policy.auto_resolution.model_dump(),
        }
        try:
            resp = self._ask(DECIDE_SYSTEM, payload, Decision.model_json_schema())
            decision = Decision.model_validate(resp.json_payload())
        except (ValidationError, ValueError) as exc:
            decision = Decision(
                action=DecisionAction.ESCALATE,
                confidence=0.0,
                reasons=[f"decision step returned invalid output: {exc}"],
            )
        log(
            "decision",
            f"model proposes {decision.action.value} {decision.amount:.2f} "
            f"(confidence {decision.confidence:.2f})",
            **decision.model_dump(mode="json"),
        )
        return decision

    # ------------------------------------------------------------------ 3. action
    def _act(
        self,
        ticket: Ticket,
        proposed: Decision,
        rules: RuleEvaluation,
        gate: GateResult,
        findings: Findings,
        log: Any,
    ) -> Resolution:
        order = findings.order.order
        name = "there"
        if findings.history and findings.history.customer_id:
            cust = self.db.get_customer(findings.history.customer_id)
            name = cust.name.split()[0] if cust else name
        currency = order.currency if order else self.policy.currency
        reason_line = "; ".join(proposed.reasons[:2]).capitalize() + "." if proposed.reasons else ""
        flags = list(gate.flags)

        if gate.automate and proposed.action in REFUND_ACTIONS and order is not None:
            try:
                pay_id = self.payments.refund(
                    order_id=order.id,
                    customer_id=order.customer_id,
                    amount=proposed.amount,
                    reason=findings.analysis.issue_type,
                    idempotency_key=f"refund:{ticket.id}",
                )
            except PaymentError as exc:
                flags.append(f"payment_failed:{exc}")
                log("action", f"payment failed: {exc}")
            else:
                msg = self.notifier.notify(
                    ticket,
                    proposed.action.value,
                    name=name,
                    order_id=order.id,
                    amount=f"{proposed.amount:.2f}",
                    currency=currency,
                    reason_line=reason_line,
                )
                log(
                    "action",
                    f"refunded {proposed.amount:.2f} {currency} ({pay_id}); customer notified",
                )
                self.db.set_ticket_status(ticket.id, "resolved")
                return Resolution(
                    ticket_id=ticket.id,
                    outcome=Outcome.AUTO_APPROVED
                    if proposed.action == DecisionAction.APPROVE_REFUND
                    else Outcome.AUTO_PARTIAL,
                    final_action=proposed.action,
                    automated=True,
                    proposed=proposed,
                    amount_refunded=proposed.amount,
                    guardrail_flags=flags,
                    customer_message=msg,
                    refund_id=pay_id,
                )

        if gate.automate and proposed.action == DecisionAction.DENY:
            msg = self.notifier.notify(
                ticket,
                "deny",
                name=name,
                order_id=order.id if order else "-",
                reason_line=reason_line,
            )
            log("action", "request declined per policy; customer notified")
            self.db.set_ticket_status(ticket.id, "resolved")
            return Resolution(
                ticket_id=ticket.id,
                outcome=Outcome.AUTO_DENIED,
                final_action=DecisionAction.DENY,
                automated=True,
                proposed=proposed,
                guardrail_flags=flags,
                customer_message=msg,
            )

        # Escalate with a full handoff packet.
        packet = self._handoff_packet(ticket, proposed, rules, flags, findings)
        priority = packet["priority"]
        esc_id = self.queue.escalate(ticket.id, packet, priority=priority)
        msg = self.notifier.notify(
            ticket,
            "escalate",
            name=name,
            order_ref=f"order {order.id}" if order else "your request",
        )
        log("action", f"escalated to human queue as {esc_id} (priority {priority})")
        self.db.set_ticket_status(ticket.id, "escalated")
        return Resolution(
            ticket_id=ticket.id,
            outcome=Outcome.ESCALATED,
            final_action=DecisionAction.ESCALATE,
            automated=False,
            proposed=proposed,
            guardrail_flags=flags,
            customer_message=msg,
            escalation_id=esc_id,
        )

    def _handoff_packet(
        self,
        ticket: Ticket,
        proposed: Decision,
        rules: RuleEvaluation,
        flags: list[str],
        findings: Findings,
    ) -> dict[str, Any]:
        a, fraud, order = findings.analysis, findings.fraud, findings.order.order
        urgent = bool(set(a.triggers) & {"legal_threat", "safety_hazard"}) or (
            fraud is not None and fraud.level == "high"
        )
        high = (
            (order is not None and order.total > 500)
            or (findings.history is not None and findings.history.tier == "vip")
            or (fraud is not None and fraud.level == "medium")
        )
        why = rules.escalate_reasons + [f for f in flags if f != "model_requested_escalation"]
        if fraud is not None and fraud.level != "low" and not any("fraud" in w for w in why):
            signals = ", ".join(s.name for s in fraud.signals)
            why.append(f"fraud risk {fraud.level} ({fraud.score}): {signals}")
        summary = (
            f"Customer reports '{a.issue_type}' (classifier confidence {a.confidence:.2f}). "
            f"Policy evaluation: {rules.recommended_action.value}"
            + (f" of {rules.refund_amount:.2f}" if rules.refund_amount else "")
            + f". AI proposed {proposed.action.value}"
            + (f" {proposed.amount:.2f}" if proposed.amount else "")
            + f" at confidence {proposed.confidence:.2f}. "
            + (f"Escalated because: {'; '.join(why)}." if why else "Model chose to escalate.")
        )
        steps = []
        if "carrier" in " ".join(rules.escalate_reasons):
            steps.append("Open a trace with the carrier and request proof-of-delivery photo.")
        if fraud and fraud.level != "low":
            steps.append("Verify identity / review fraud signals before refunding.")
        if a.triggers:
            steps.append(f"Handle with care: triggers {a.triggers}. Consider a phone callback.")
        if "amount_above_auto_limit" in flags:
            steps.append("Amount above automation limit: approve manually if evidence holds.")
        if not steps:
            steps.append("Review evidence and confirm or override the AI proposal.")
        return {
            "ticket": self._ticket_view(ticket),
            "priority": "urgent" if urgent else "high" if high else "normal",
            "reasoning_summary": summary,
            "why_escalated": why or ["model_requested_escalation"],
            "ai_proposal": proposed.model_dump(mode="json"),
            "policy_evaluation": rules.model_dump(mode="json"),
            "findings": findings.as_context(),
            "suggested_next_steps": steps,
        }
