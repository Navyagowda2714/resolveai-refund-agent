"""Labelled synthetic evaluation set (~80 tickets).

Each scenario creates its own customers/orders/refunds and a raw channel payload, plus the label a
senior support agent would assign (``approve_refund | partial_refund | deny | escalate``) and the
expected refund amount. Labels encode *intent* (what a careful human would do), written by hand
per scenario, not by calling the agent's rules engine. A slice of deliberately hard tickets
(unusual phrasing of threats, mixed signals, negations) probes where the lexicon-based offline
analysis breaks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from resolveai.data.seed import DEFAULT_ANCHOR, SHORT_NAMES, DataFactory
from resolveai.evaluation.payloads import build_payload
from resolveai.models import Channel, Customer, Order, RefundRecord

CHANNELS = list(Channel)


@dataclass
class Scenario:
    id: str
    name: str
    channel: Channel
    payload: dict[str, Any]
    label: str
    label_amount: float
    hard: bool = False
    customers: list[Customer] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    refunds: list[RefundRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario": self.name,
            "channel": self.channel.value,
            "label": self.label,
            "label_amount": self.label_amount,
            "hard": self.hard,
            "payload": self.payload,
        }


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.f = DataFactory(self.rng, DEFAULT_ANCHOR, prefix="E", order_start=50000)
        self.n = 0
        self.scenarios: list[Scenario] = []

    def add(
        self,
        name: str,
        *,
        customer: Customer,
        order: Order | None,
        text: str,
        label: str,
        amount: float = 0.0,
        reason_code: str = "other",
        photo: str | None = None,
        order_ref: str | None = "auto",
        hard: bool = False,
        extra_orders: list[Order] | None = None,
        refunds: list[RefundRecord] | None = None,
        extra_customers: list[Customer] | None = None,
    ) -> None:
        self.n += 1
        channel = CHANNELS[self.n % len(CHANNELS)]
        ref = (order.id if order else None) if order_ref == "auto" else order_ref
        payload = build_payload(
            channel,
            uid=f"eval-{self.n:03d}",
            email=customer.email,
            name=customer.name,
            order_id=ref,
            text=text,
            created_at=DEFAULT_ANCHOR,
            reason_code=reason_code,
            photo_caption=photo,
        )
        self.scenarios.append(
            Scenario(
                id=f"EV-{self.n:03d}",
                name=name,
                channel=channel,
                payload=payload,
                label=label,
                label_amount=round(amount, 2),
                hard=hard,
                customers=[customer, *(extra_customers or [])],
                orders=([order] if order else []) + (extra_orders or []),
                refunds=refunds or [],
            )
        )

    def cust(self, **kw: Any) -> Customer:
        kw.setdefault("account_age_days", self.rng.randint(200, 1400))
        kw.setdefault("tier", "standard")
        return self.f.customer(**kw)

    def history(self, c: Customer, n: int = 4) -> list[Order]:
        """Some unremarkable past orders so 'amount vs history' has a baseline."""
        return [self.f.order(c, days_ago=self.rng.randint(60, 180)) for _ in range(n)]


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def build_eval_set(seed: int = 2026) -> list[Scenario]:
    b = _Builder(seed)
    f, rng = b.f, b.rng
    item = f.item

    # 1. Damaged, low value -> approve (with/without photo).
    for sku in ["HM-300", "HM-330", "TY-600", "BK-700", "AP-200"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(5, 15))
        name = SHORT_NAMES[sku]
        b.add(
            "damaged_low",
            customer=c,
            order=o,
            label="approve_refund",
            amount=o.items_total,
            reason_code="damaged",
            photo=_pick(rng, [f"{name} with a large crack across the surface", None]),
            text=_pick(
                rng,
                [
                    f"My {name} arrived broken, the box was soaked and it is cracked.",
                    f"The {name} came damaged - one side is smashed. Can I get a refund?",
                    f"Unfortunately the {name} was shattered when I opened the parcel.",
                ],
            ),
            extra_orders=b.history(c, 2),
        )

    # 2. Damaged, high value -> escalate (above automation limit).
    for sku in ["EL-110", "HM-310", "EL-150", "AP-210"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(4, 12))
        name = o.items[0].name.lower()
        b.add(
            "damaged_high_value",
            customer=c,
            order=o,
            label="escalate",
            reason_code="damaged",
            photo=f"{name} with a dented corner",
            text=f"The {name} arrived damaged, the casing is dented and cracked. Please refund.",
            extra_orders=b.history(c),
        )

    # 3. Defective, low value -> approve.
    for sku in ["EL-100", "BT-410", "TY-610", "EL-130"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(8, 20))
        name = SHORT_NAMES[sku]
        b.add(
            "defective_low",
            customer=c,
            order=o,
            label="approve_refund",
            amount=o.items_total,
            reason_code="defective",
            text=_pick(
                rng,
                [
                    f"The {name} stopped working after two days.",
                    f"My {name} won't turn on at all, I tried charging it overnight.",
                    f"The {name} is defective, it keeps turning off.",
                ],
            ),
            extra_orders=b.history(c),
        )

    # 4. Missing named item from a multi-item order -> partial for that item.
    for skus, missing in [
        (["AP-220", "EL-130", "BT-400"], 1),
        (["HM-300", "GR-500"], 1),
        (["TY-600", "BK-700", "GR-510"], 0),
        (["AP-200", "BT-400"], 1),
        (["EL-100", "HM-330"], 0),
    ]:
        c = b.cust()
        o = f.order(c, items=[item(s) for s in skus], days_ago=rng.randint(4, 10))
        it = o.items[missing]
        key = SHORT_NAMES[it.sku]
        b.add(
            "missing_item",
            customer=c,
            order=o,
            label="partial_refund",
            amount=it.line_total,
            reason_code="missing_item",
            text=_pick(
                rng,
                [
                    f"The {key} was missing from my parcel, everything else arrived.",
                    f"I only received part of my order - the {key} wasn't in the box.",
                ],
            ),
            extra_orders=b.history(c),
        )

    # 5. Wrong item -> approve.
    for sku, other in [
        ("AP-220", "a pair of sandals"),
        ("BT-400", "a hand cream"),
        ("HM-330", "a desk fan"),
        ("TY-610", "a puzzle"),
    ]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(5, 12))
        nm = o.items[0].name.lower()
        b.add(
            "wrong_item",
            customer=c,
            order=o,
            label="approve_refund",
            amount=o.items_total,
            reason_code="wrong_item",
            text=f"You sent me {other} instead of the {nm}.",
            extra_orders=b.history(c),
        )

    # 6. Not received, no delivery confirmation, past grace -> approve full order.
    for sku in ["BT-400", "AP-200", "TY-600"]:
        c = b.cust()
        o = f.order(
            c,
            items=[item(sku)],
            days_ago=rng.randint(16, 25),
            status="shipped",
            tracking_confirmed=False,
        )
        b.add(
            "not_received_unconfirmed",
            customer=c,
            order=o,
            label="approve_refund",
            amount=o.total,
            reason_code="not_received",
            text=_pick(
                rng,
                [
                    "My order never arrived and tracking hasn't updated in two weeks.",
                    "I still haven't received my package, it's been weeks.",
                ],
            ),
            extra_orders=b.history(c),
        )

    # 7. Not received, but carrier confirms delivery -> escalate (carrier dispute).
    for sku in ["EL-100", "AP-220", "HM-320"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(6, 14))
        b.add(
            "not_received_confirmed",
            customer=c,
            order=o,
            label="escalate",
            reason_code="not_received",
            text="Tracking says delivered but I never got it. Nothing at my door.",
            extra_orders=b.history(c),
        )

    # 8. Still in transit -> deny (give tracking).
    for sku in ["BK-700", "HM-300", "AP-200"]:
        c = b.cust()
        o = f.order(
            c,
            items=[item(sku)],
            days_ago=rng.randint(2, 5),
            status="shipped",
            tracking_confirmed=False,
        )
        b.add(
            "in_transit",
            customer=c,
            order=o,
            label="deny",
            reason_code="not_received",
            text="I haven't received my order yet, can I get a refund?",
            extra_orders=b.history(c),
        )

    # 9. Change of mind within window -> partial (15% restocking fee).
    for sku in ["AP-200", "AP-220", "HM-330", "TY-600"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(6, 18))
        b.add(
            "changed_mind",
            customer=c,
            order=o,
            label="partial_refund",
            amount=round(o.items_total * 0.85, 2),
            reason_code="changed_mind",
            text=_pick(
                rng,
                [
                    "I changed my mind, I'd like to return it for a refund.",
                    "It doesn't fit and I no longer need it.",
                    "Ordered by mistake, it's unused - can I return it?",
                ],
            ),
            extra_orders=b.history(c),
        )

    # 10. Outside the return window -> deny.
    for sku, days in [("BT-400", 40), ("HM-330", 75), ("TY-600", 60), ("GR-500", 20)]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=days)
        b.add(
            "outside_window",
            customer=c,
            order=o,
            label="deny",
            reason_code="changed_mind",
            text=_pick(
                rng,
                [
                    "I don't need this anymore, please refund me.",
                    "Changed my mind about this purchase, I'd like my money back.",
                ],
            ),
            extra_orders=b.history(c),
        )

    # 11. Final sale, change of mind -> deny.
    for _ in range(3):
        c = b.cust()
        o = f.order(c, items=[item("AP-230")], days_ago=rng.randint(5, 15))
        b.add(
            "final_sale",
            customer=c,
            order=o,
            label="deny",
            reason_code="changed_mind",
            text="The denim jacket doesn't fit, I want to return it.",
            extra_orders=b.history(c),
        )

    # 12. Non-refundable categories -> deny.
    for sku in ["GC-800", "DG-900", "GC-800"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(3, 10))
        b.add(
            "non_refundable",
            customer=c,
            order=o,
            label="deny",
            reason_code="changed_mind",
            text="I don't need this anymore, please refund it.",
            extra_orders=b.history(c),
        )

    # 13. Serial refunder, low value damage -> escalate (fraud risk).
    for sku in ["HM-330", "BT-410", "AP-200", "TY-610"]:
        c = b.cust(account_age_days=rng.randint(120, 400))
        past = b.history(c, 5)
        recent = [f.refund(c, past[i], days_ago=rng.randint(10, 55)) for i in range(3)]
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(4, 10))
        b.add(
            "serial_refunder",
            customer=c,
            order=o,
            label="escalate",
            reason_code="damaged",
            text="It arrived damaged again, the item is broken. Refund please.",
            extra_orders=past,
            refunds=recent,
        )

    # 14. Brand-new account + billing country mismatch -> escalate.
    for sku in ["AP-210", "HM-320", "EL-100"]:
        c = b.cust(account_age_days=rng.randint(2, 9))
        o = f.order(
            c,
            items=[item(sku)],
            days_ago=2 if sku != "AP-210" else 1,
            delivery_days=1,
            billing_country="ES",
        )
        b.add(
            "new_account_mismatch",
            customer=c,
            order=o,
            label="escalate",
            reason_code="damaged",
            text="Item is broken, I want a full refund immediately.",
        )

    # 15. Explicit mandatory-review triggers -> escalate.
    for sku, text in [
        ("BT-410", "The hair dryer caught fire while I was using it! I want a refund."),
        ("EL-130", "The hub overheated and melted the cable. Refund now."),
        (
            "AP-200",
            "The sweater arrived torn. If I don't get a refund I'll open a dispute with my bank.",
        ),
        ("HM-330", "Lamp arrived cracked. This is unacceptable, I'm contacting my lawyer."),
    ]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=rng.randint(5, 12))
        b.add(
            "mandatory_trigger",
            customer=c,
            order=o,
            label="escalate",
            text=text,
            extra_orders=b.history(c),
        )

    # 16. Order belongs to someone else -> escalate.
    for sku in ["EL-100", "AP-220"]:
        owner, c = b.cust(), b.cust()
        o = f.order(owner, items=[item(sku)], days_ago=6)
        b.add(
            "identity_mismatch",
            customer=c,
            order=o,
            label="escalate",
            reason_code="damaged",
            text="My item arrived broken, please refund.",
            extra_customers=[owner],
            extra_orders=b.history(c, 2),
        )

    # 17. Unrecognised charge -> escalate.
    for _ in range(2):
        c = b.cust()
        o = f.order(c, items=[item("EL-120")], days_ago=3)
        b.add(
            "unauthorized_charge",
            customer=c,
            order=o,
            label="escalate",
            text="I don't recognise this charge on my card. I never placed this order.",
            extra_orders=b.history(c),
        )

    # 18. Already refunded in full -> deny.
    for sku in ["HM-330", "BK-700"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=12)
        b.add(
            "already_refunded",
            customer=c,
            order=o,
            label="deny",
            reason_code="damaged",
            text="The item was damaged, I want my refund.",
            refunds=[f.refund(c, o, days_ago=5, amount=o.total)],
            extra_orders=b.history(c),
        )

    # 19. Vague -> escalate (needs clarification).
    for text in [
        "Hi, I have a problem with my order, please help.",
        "Can someone call me about my last purchase?",
    ]:
        c = b.cust()
        o = f.order(c, items=[item("HM-300")], days_ago=8)
        b.add("vague", customer=c, order=o, label="escalate", text=text, extra_orders=b.history(c))

    # 20. Missing item, not identifiable -> escalate.
    for skus in [["TY-600", "BK-700", "GR-510"], ["AP-200", "BT-400", "HM-330"]]:
        c = b.cust()
        o = f.order(c, items=[item(s) for s in skus], days_ago=6)
        b.add(
            "missing_unidentified",
            customer=c,
            order=o,
            label="escalate",
            reason_code="missing_item",
            text="Something was missing from my parcel.",
            extra_orders=b.history(c),
        )

    # 21. Late delivery, product fine -> partial (shipping refund or 10% goodwill).
    for sku in ["BK-700", "HM-320"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=12, delivery_days=9)
        amount = o.shipping_cost if o.shipping_cost > 0 else round(o.items_total * 0.10, 2)
        b.add(
            "late_delivery",
            customer=c,
            order=o,
            label="partial_refund",
            amount=amount,
            text="My order arrived a week late, after the promised delivery date.",
            extra_orders=b.history(c),
        )

    # ---------------------------------------------------------------- hard cases
    # Threats phrased outside the lexicon: a careful human escalates these.
    for sku, text in [
        (
            "HM-330",
            "The lamp is cracked. Refund it or I'll ask my card issuer to claw the money back.",
        ),
        (
            "EL-130",
            "The charging hub stopped working after it got scorching hot and left a scorch "
            "mark on my desk.",
        ),
        (
            "TY-600",
            "Box arrived crushed and pieces are broken. My solicitor says you have to refund.",
        ),
    ]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=7)
        b.add(
            "hard_hidden_trigger",
            customer=c,
            order=o,
            label="escalate",
            text=text,
            hard=True,
            extra_orders=b.history(c),
        )

    # Negation: "not broken, I just don't like it" -> change of mind (partial).
    for sku in ["AP-200", "HM-330"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=9)
        b.add(
            "hard_negation",
            customer=c,
            order=o,
            label="partial_refund",
            amount=round(o.items_total * 0.85, 2),
            hard=True,
            text="It's not broken or anything, I just don't like the colour. Can I return it?",
            extra_orders=b.history(c),
        )

    # Mixed signals: dented *box* + late arrival, product fine -> late-delivery partial.
    for sku in ["HM-330", "BT-400"]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=10, delivery_days=8)
        amount = o.shipping_cost if o.shipping_cost > 0 else round(o.items_total * 0.10, 2)
        b.add(
            "hard_mixed_signals",
            customer=c,
            order=o,
            label="partial_refund",
            amount=amount,
            hard=True,
            text="The box was dented and it arrived days late. The product itself "
            "works fine, but I'd like some compensation for the delay.",
            extra_orders=b.history(c),
        )

    # No order id given: the agent must infer the latest order from history.
    for sku in ["BT-400", "BK-700"]:
        c = b.cust()
        older = b.history(c, 3)
        o = f.order(c, items=[item(sku)], days_ago=4)
        b.add(
            "hard_no_order_id",
            customer=c,
            order=o,
            label="approve_refund",
            amount=o.items_total,
            order_ref=None,
            hard=True,
            reason_code="damaged",
            text=f"My {SHORT_NAMES[sku]} from last week arrived damaged.",
            extra_orders=older,
        )

    # Low-value item, but text shows the customer used it for weeks (abuse) -> deny.
    c = b.cust()
    o = f.order(c, items=[item("AP-220")], days_ago=25)
    b.add(
        "hard_used_item",
        customer=c,
        order=o,
        label="deny",
        hard=True,
        text="I've worn these running shoes for 3 weeks of training and changed my mind, "
        "they're a bit worn now but I want a refund.",
        extra_orders=b.history(c),
    )

    # Defect after window but within warranty -> escalate (warranty claim).
    for sku, days in [("EL-100", 120), ("HM-310", 200)]:
        c = b.cust()
        o = f.order(c, items=[item(sku)], days_ago=days)
        b.add(
            "warranty",
            customer=c,
            order=o,
            label="escalate",
            reason_code="defective",
            text="It stopped working last week, it's defective.",
            extra_orders=b.history(c),
        )

    return b.scenarios
