"""Drive the REST API: create tickets, resolve them, then work the human queue.

Start the server first:  make serve   (seeds data/resolveai.db and the showcase payloads)
Then run:                python examples/http_walkthrough.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
showcase = json.loads(Path("data/resolveai.showcase.json").read_text())

with httpx.Client(base_url=BASE, timeout=30) as client:
    for channel, payload in showcase:
        ticket = client.post("/tickets", json={"channel": channel, "payload": payload}).json()
        tid = ticket["ticket"]["id"]
        res = client.post(f"/tickets/{tid}/resolve").json()
        print(f"{tid} {channel:<7} -> {res['outcome']:<14} {res['amount_refunded']:>8.2f}")

    queue = client.get("/queue").json()
    print(f"\n{len(queue)} ticket(s) waiting for a human")
    for entry in queue:
        item = client.get(f"/queue/{entry['id']}").json()
        packet = item["packet"]
        print(f"\n{entry['id']} [{entry['priority']}] {packet['reasoning_summary']}")
        print("next steps:", json.dumps(packet["suggested_next_steps"]))
        # A human reviews the packet; here we accept the AI's refund proposal when it made one.
        proposal = packet["ai_proposal"]
        if proposal["action"] in ("approve_refund", "partial_refund"):
            decision = {
                "action": proposal["action"],
                "amount": proposal["amount"],
                "agent": "agent-42",
                "notes": "Reviewed photos and order history.",
            }
        else:
            decision = {
                "action": "deny",
                "agent": "agent-42",
                "notes": "Carrier proof of delivery confirmed.",
            }
        print(
            "human decision ->", client.post(f"/queue/{entry['id']}/decision", json=decision).json()
        )
