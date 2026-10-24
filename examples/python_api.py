"""Use ResolveAI as a library: seed an in-memory store, ingest one ticket, resolve it.

Run: python examples/python_api.py   (offline by default; set LLM_PROVIDER/ANTHROPIC_API_KEY
to use a real model)
"""

from __future__ import annotations

from resolveai.data.db import CommerceDB
from resolveai.data.seed import seed_database
from resolveai.demo import build_showcase
from resolveai.service import ResolveService

db = CommerceDB(":memory:")
seed_database(db, n_customers=20)
channel, payload = build_showcase(db)[1]  # the live-chat "missing item" ticket

service = ResolveService(db)
ticket = service.ingest(channel, payload)
resolution = service.resolve(ticket.id)

print(f"ticket   : {ticket.id} via {ticket.channel.value}")
print(f"outcome  : {resolution.outcome.value} ({resolution.amount_refunded:.2f} EUR)")
print(f"proposal : {resolution.proposed.model_dump_json(indent=2)}")
for event in resolution.trace:
    print(f"  {event.step:>2} {event.kind:<10} {event.summary}")
