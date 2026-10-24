"""Rule + score based fraud checks. Every point in the score is explained by a signal."""

from __future__ import annotations

from resolveai.models import Order
from resolveai.tools.schemas import CustomerHistory, FraudReport, FraudSignal


def fraud_level(score: int, levels: dict[str, int]) -> str:
    if score >= levels.get("high", 60):
        return "high"
    if score >= levels.get("medium", 30):
        return "medium"
    return "low"


def run_fraud_checks(
    history: CustomerHistory,
    order: Order | None,
    *,
    claimed_amount: float,
    issue_type: str,
    levels: dict[str, int] | None = None,
) -> FraudReport:
    signals: list[FraudSignal] = []

    def add(name: str, weight: int, detail: str) -> None:
        signals.append(FraudSignal(name=name, weight=weight, detail=detail))

    # 1. Refund frequency.
    if history.refunds_last_90d >= 3:
        add("refund_frequency", 40, f"{history.refunds_last_90d} refunds in the last 90 days")
    elif history.refunds_last_90d == 2:
        add("refund_frequency", 20, "2 refunds in the last 90 days")
    if history.total_orders >= 4 and history.refund_rate > 0.5:
        add(
            "refund_rate",
            15,
            f"refund rate {history.refund_rate:.0%} over {history.total_orders} orders",
        )

    # 2. Account age.
    if history.account_age_days < 14:
        add("new_account", 30, f"account is {history.account_age_days} days old")
    elif history.account_age_days < 60:
        add("young_account", 12, f"account is {history.account_age_days} days old")

    # 3. Amount vs. history.
    if history.total_orders >= 3 and history.avg_order_value > 0:
        ratio = claimed_amount / history.avg_order_value
        if ratio > 3:
            add("amount_vs_history", 15, f"claim is {ratio:.1f}x the average order value")
    if history.total_orders <= 1 and claimed_amount > 200:
        add("high_value_first_order", 10, f"{claimed_amount:.2f} claimed on a first order")

    # 4. Address mismatch.
    if order is not None:
        ship, bill = order.shipping_address, order.billing_address
        if ship.country != bill.country:
            add(
                "address_country_mismatch", 25, f"shipping {ship.country} vs billing {bill.country}"
            )
        elif ship != bill:
            add("address_mismatch", 8, "shipping and billing addresses differ")
        # 5. Claim contradicts carrier data.
        if (
            issue_type == "not_received"
            and order.status == "delivered"
            and order.tracking_confirmed
        ):
            add("delivery_contradiction", 20, "claims non-receipt but carrier confirms delivery")

    score = min(100, sum(s.weight for s in signals))
    return FraudReport(score=score, level=fraud_level(score, levels or {}), signals=signals)
