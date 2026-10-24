from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from resolveai.agent.offline_policy import OFFLINE_HANDLERS
from resolveai.data.db import CommerceDB
from resolveai.data.seed import DataFactory
from resolveai.llm import ScriptedLLM
from resolveai.service import ResolveService

ANCHOR = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db() -> CommerceDB:
    d = CommerceDB(":memory:")
    yield d
    d.close()


@pytest.fixture
def factory() -> DataFactory:
    return DataFactory(random.Random(1), ANCHOR, prefix="T", order_start=70000)


@pytest.fixture
def service(db: CommerceDB) -> ResolveService:
    return ResolveService(db, llm=ScriptedLLM(OFFLINE_HANDLERS))


def chat_payload(email: str, text: str, order_id: str | None, ts: datetime = ANCHOR) -> dict:
    return {
        "session_id": f"s-{abs(hash(text)) % 10**8}",
        "customer": {"email": email},
        "order_id": order_id,
        "messages": [{"sender": "customer", "text": text, "ts": ts.isoformat()}],
    }
