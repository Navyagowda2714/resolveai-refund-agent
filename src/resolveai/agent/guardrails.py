"""Safety gate: decides whether a proposed decision may be executed without a human.

The gate is deliberately independent of the LLM. It encodes autonomy limits (amount, risk,
confidence) and consistency with the deterministic policy evaluation. If any check fails the
ticket is escalated, never silently "corrected".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from resolveai.models import Decision, DecisionAction
from resolveai.policy.engine import Policy, RuleEvaluation
from resolveai.tools.schemas import FraudReport, PolicyLookup

REFUND_ACTIONS = {DecisionAction.APPROVE_REFUND, DecisionAction.PARTIAL_REFUND}


class GateResult(BaseModel):
    automate: bool
    flags: list[str] = Field(default_factory=list)


class SafetyGate:
    def __init__(self, policy: Policy, enforce: bool = True) -> None:
        self.limits = policy.auto_resolution
        self.enforce = enforce

    def check(
        self,
        proposed: Decision,
        rules: RuleEvaluation,
        fraud: FraudReport | None,
        policy_lookup: PolicyLookup | None,
    ) -> GateResult:
        if proposed.action == DecisionAction.ESCALATE:
            return GateResult(automate=False, flags=["model_requested_escalation"])
        if not self.enforce:
            return GateResult(automate=True, flags=["guardrails_disabled"])

        lim = self.limits
        flags: list[str] = []
        if rules.escalate_reasons:
            flags.append("policy_requires_human")
        if proposed.action in REFUND_ACTIONS:
            if not rules.eligible:
                flags.append("policy_conflict:refund_not_eligible")
            elif proposed.amount > rules.refund_amount + 0.01:
                flags.append("amount_exceeds_policy")
            if proposed.amount > lim.max_auto_refund_amount:
                flags.append("amount_above_auto_limit")
            if proposed.amount <= 0:
                flags.append("non_positive_refund")
            if fraud is None:
                flags.append("fraud_check_missing")
            elif fraud.score > lim.max_fraud_score:
                flags.append(f"fraud_risk_{fraud.level}")
            if policy_lookup is None:
                flags.append("policy_check_missing")
            if proposed.confidence < lim.min_confidence:
                flags.append("low_confidence")
        elif proposed.action == DecisionAction.DENY:
            if not rules.deny_reasons:
                flags.append("deny_without_policy_basis")
            if proposed.confidence < lim.min_deny_confidence:
                flags.append("low_confidence")
        return GateResult(automate=not flags, flags=flags)
