"""Deterministic synthetic commerce dataset generator.

Produces customers with realistic order histories (mixed categories, price ranges, delivery
timing, address data) and refund histories with a long-tail of "serial refunders". Everything is
derived from a seeded ``random.Random`` so datasets are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from resolveai.data.db import CommerceDB
from resolveai.models import Address, Customer, Order, OrderItem, RefundRecord

# (sku, name, category, base price, final_sale)
CATALOG: list[tuple[str, str, str, float, bool]] = [
    ("EL-100", "Wireless Earbuds Pro", "electronics", 89.0, False),
    ("EL-110", "Noise-Cancelling Headphones", "electronics", 249.0, False),
    ("EL-120", "Smartwatch S3", "electronics", 199.0, False),
    ("EL-130", "USB-C Charging Hub", "electronics", 39.0, False),
    ("EL-140", "14-inch Ultrabook", "electronics", 1149.0, False),
    ("EL-150", "Mirrorless Camera Body", "electronics", 899.0, False),
    ("AP-200", "Merino Wool Sweater", "apparel", 79.0, False),
    ("AP-210", "Waterproof Hiking Jacket", "apparel", 159.0, False),
    ("AP-220", "Running Shoes", "apparel", 119.0, False),
    ("AP-230", "Clearance Denim Jacket", "apparel", 35.0, True),
    ("HM-300", "Ceramic Dinner Set (12pc)", "home", 64.0, False),
    ("HM-310", "Espresso Machine", "home", 329.0, False),
    ("HM-320", "Linen Bedding Set", "home", 119.0, False),
    ("HM-330", "Glass Table Lamp", "home", 54.0, False),
    ("BT-400", "Vitamin C Serum", "beauty", 29.0, False),
    ("BT-410", "Hair Dryer Ionic", "beauty", 69.0, False),
    ("GR-500", "Organic Coffee Beans 1kg", "grocery", 24.0, False),
    ("GR-510", "Olive Oil Gift Box", "grocery", 42.0, False),
    ("TY-600", "Building Blocks Castle", "toys", 59.0, False),
    ("TY-610", "Remote Control Car", "toys", 74.0, False),
    ("BK-700", "Hardcover Cookbook", "books", 32.0, False),
    ("GC-800", "Gift Card 50", "gift_cards", 50.0, True),
    ("DG-900", "Photo Editing Software Licence", "digital", 79.0, True),
]
CATALOG_BY_SKU = {c[0]: c for c in CATALOG}
# How customers actually refer to products in messages.
SHORT_NAMES = {
    "EL-100": "earbuds",
    "EL-110": "headphones",
    "EL-120": "smartwatch",
    "EL-130": "charging hub",
    "EL-140": "laptop",
    "EL-150": "camera",
    "AP-200": "sweater",
    "AP-210": "jacket",
    "AP-220": "running shoes",
    "AP-230": "denim jacket",
    "HM-300": "dinner set",
    "HM-310": "espresso machine",
    "HM-320": "bedding set",
    "HM-330": "lamp",
    "BT-400": "serum",
    "BT-410": "hair dryer",
    "GR-500": "coffee beans",
    "GR-510": "olive oil",
    "TY-600": "building blocks",
    "TY-610": "remote control car",
    "BK-700": "cookbook",
    "GC-800": "gift card",
    "DG-900": "software licence",
}

FIRST = [
    "Maya",
    "Luca",
    "Aisha",
    "Tom",
    "Giulia",
    "Arjun",
    "Sofia",
    "Ben",
    "Elena",
    "Kenji",
    "Chiara",
    "Omar",
    "Lea",
    "Marco",
    "Priya",
    "Noah",
    "Sara",
    "Ivan",
    "Nadia",
    "Leo",
]
LAST = [
    "Rossi",
    "Patel",
    "Müller",
    "Bianchi",
    "Khan",
    "Novak",
    "Silva",
    "Martin",
    "Ricci",
    "Kowalski",
    "Dubois",
    "Moretti",
    "Tanaka",
    "Costa",
    "Fischer",
    "Greco",
]
CITIES = [
    ("Milan", "20121", "IT"),
    ("Rome", "00184", "IT"),
    ("Turin", "10121", "IT"),
    ("Munich", "80331", "DE"),
    ("Lyon", "69002", "FR"),
    ("Madrid", "28013", "ES"),
    ("Vienna", "1010", "AT"),
    ("Bologna", "40121", "IT"),
]
STREETS = [
    "Via Roma",
    "Corso Italia",
    "Hauptstraße",
    "Rue de la Paix",
    "Calle Mayor",
    "Via Garibaldi",
]

DEFAULT_ANCHOR = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


class DataFactory:
    """Builds individual synthetic records. Used by the seeder, the demo and the eval set."""

    def __init__(
        self,
        rng: random.Random,
        anchor: datetime = DEFAULT_ANCHOR,
        *,
        prefix: str = "",
        order_start: int = 10000,
    ) -> None:
        self.rng = rng
        self.anchor = anchor
        self.prefix = prefix  # keeps IDs/emails of separate populations from colliding
        self._cust = 0
        self._order = order_start
        self._refund = 0

    def address(self, country: str | None = None) -> Address:
        city, pc, ctry = self.rng.choice(
            [c for c in CITIES if country is None or c[2] == country] or CITIES
        )
        return Address(
            line1=f"{self.rng.choice(STREETS)} {self.rng.randint(1, 180)}",
            city=city,
            postcode=pc,
            country=ctry,
        )

    def customer(
        self,
        *,
        account_age_days: int | None = None,
        tier: str | None = None,
        email: str | None = None,
        name: str | None = None,
        cid: str | None = None,
    ) -> Customer:
        self._cust += 1
        first, last = self.rng.choice(FIRST), self.rng.choice(LAST)
        name = name or f"{first} {last}"
        slug = name.lower().replace(" ", ".").replace("ü", "u")
        age = account_age_days if account_age_days is not None else self.rng.randint(40, 1500)
        return Customer(
            id=cid or f"CUS-{self.prefix}{self._cust:04d}",
            email=email or f"{slug}.{self.prefix.lower()}{self._cust}@example.com",
            name=name,
            created_at=self.anchor - timedelta(days=age),
            tier=tier or self.rng.choices(["standard", "plus", "vip"], [0.75, 0.2, 0.05])[0],
        )

    def item(self, sku: str | None = None, category: str | None = None, qty: int = 1) -> OrderItem:
        if sku is None:
            pool = [c for c in CATALOG if category is None or c[2] == category]
            pool = [c for c in pool if not c[4]] if category is None else pool
            sku = self.rng.choice(pool)[0]
        s, name, cat, price, final = CATALOG_BY_SKU[sku]
        return OrderItem(
            sku=s, name=name, category=cat, unit_price=price, quantity=qty, final_sale=final
        )

    def order(
        self,
        customer: Customer,
        *,
        items: list[OrderItem] | None = None,
        days_ago: int = 10,
        delivery_days: int | None = 3,
        status: str = "delivered",
        tracking_confirmed: bool = True,
        billing_country: str | None = None,
        shipping_differs: bool = False,
        oid: str | None = None,
    ) -> Order:
        self._order += 1
        ordered = self.anchor - timedelta(days=days_ago, hours=self.rng.randint(0, 10))
        shipping = self.address()
        billing = shipping
        if shipping_differs or billing_country:
            billing = self.address(billing_country)
            if billing == shipping:
                billing = Address(**{**shipping.model_dump(), "line1": "Postfach 12"})
        items = items or [self.item() for _ in range(self.rng.choice([1, 1, 1, 2, 3]))]
        delivered = (
            ordered + timedelta(days=delivery_days)
            if status == "delivered" and delivery_days is not None
            else None
        )
        return Order(
            id=oid or f"ORD-{self._order}",
            customer_id=customer.id,
            items=items,
            shipping_cost=4.99 if sum(i.line_total for i in items) < 50 else 0.0,
            ordered_at=ordered,
            delivered_at=delivered,
            status=status,
            tracking_confirmed=tracking_confirmed,
            shipping_address=shipping,
            billing_address=billing,
            payment_method=self.rng.choice(["card", "card", "paypal", "apple_pay"]),
        )

    def refund(
        self,
        customer: Customer,
        order: Order,
        *,
        days_ago: int,
        amount: float | None = None,
        reason: str = "damaged",
    ) -> RefundRecord:
        self._refund += 1
        return RefundRecord(
            id=f"RFD-{self.prefix}{self._refund:05d}",
            customer_id=customer.id,
            order_id=order.id,
            amount=round(amount if amount is not None else order.items[0].line_total, 2),
            reason=reason,
            created_at=self.anchor - timedelta(days=days_ago),
        )


@dataclass
class SeedSummary:
    customers: int
    orders: int
    refunds: int


def seed_database(
    db: CommerceDB, *, n_customers: int = 150, seed: int = 7, anchor: datetime = DEFAULT_ANCHOR
) -> SeedSummary:
    """Populate ``db`` with a realistic background population of customers/orders/refunds."""
    rng = random.Random(seed)
    f = DataFactory(rng, anchor)
    n_orders = n_refunds = 0
    for _ in range(n_customers):
        c = f.customer()
        db.add_customer(c)
        age = (anchor - c.created_at).days
        serial_refunder = rng.random() < 0.06
        for _ in range(rng.randint(1, 9)):
            o = f.order(c, days_ago=rng.randint(1, min(age, 400)))
            db.add_order(o)
            n_orders += 1
            p_refund = 0.55 if serial_refunder else 0.07
            if rng.random() < p_refund:
                days = max(0, (anchor - o.ordered_at).days - rng.randint(2, 10))
                db.add_refund(
                    f.refund(
                        c,
                        o,
                        days_ago=days,
                        reason=rng.choice(["damaged", "changed_mind", "not_received", "defective"]),
                    )
                )
                n_refunds += 1
    return SeedSummary(customers=n_customers, orders=n_orders, refunds=n_refunds)
