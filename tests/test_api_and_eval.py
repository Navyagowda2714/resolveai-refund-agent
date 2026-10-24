from fastapi.testclient import TestClient

from resolveai.api import create_app
from resolveai.data.seed import seed_database
from resolveai.demo import build_showcase
from resolveai.evaluation.dataset import build_eval_set
from resolveai.evaluation.runner import Row, metrics


def test_api_end_to_end(service, db):
    seed_database(db, n_customers=10)
    showcase = build_showcase(db)
    client = TestClient(create_app(service))
    assert client.get("/health").json()["status"] == "ok"

    outcomes = []
    for channel, payload in showcase:
        r = client.post("/tickets", json={"channel": channel, "payload": payload})
        assert r.status_code == 201
        tid = r.json()["ticket"]["id"]
        res = client.post(f"/tickets/{tid}/resolve").json()
        outcomes.append(res["outcome"])
    assert outcomes == ["auto_approved", "auto_partial", "escalated", "escalated", "auto_denied"]

    queue = client.get("/queue").json()
    assert len(queue) == 2
    esc_id = queue[0]["id"]
    assert client.get(f"/queue/{esc_id}").json()["packet"]["reasoning_summary"]
    r = client.post(
        f"/queue/{esc_id}/decision",
        json={"action": "deny", "agent": "agent-1", "notes": "carrier proof"},
    )
    assert r.status_code == 200
    assert len(client.get("/queue").json()) == 1
    assert client.post("/tickets/TKT-NOPE/resolve").status_code == 404
    bad = client.post("/tickets", json={"channel": "email", "payload": {"body": "x"}})
    assert bad.status_code == 422


def test_eval_set_shape():
    s = build_eval_set()
    assert len(s) == 80
    assert len({x.id for x in s}) == 80
    labels = {x.label for x in s}
    assert labels == {"approve_refund", "partial_refund", "deny", "escalate"}
    assert sum(x.hard for x in s) >= 10


def _row(label, pred, automated, **kw):
    base = dict(
        id="x",
        scenario="s",
        channel="chat",
        hard=False,
        label=label,
        label_amount=10.0,
        predicted=pred,
        proposed=pred,
        automated=automated,
        amount=10.0 if automated else 0,
        correct=label == pred,
        amount_ok=True,
        unsafe_approval=False,
        over_refund=False,
        flags=[],
        tool_calls=3,
        llm_calls=4,
        simulated_latency_ms=100.0,
    )
    base.update(kw)
    return Row(**base)


def test_metrics_definitions():
    rows = [
        _row("approve_refund", "approve_refund", True),
        _row("escalate", "approve_refund", True, unsafe_approval=True),
        _row("escalate", "escalate", False),
        _row("deny", "escalate", False),
    ]
    m = metrics(rows)
    assert m["autonomous_resolution_pct"] == 50.0
    assert m["decision_accuracy_pct"] == 50.0
    assert m["unsafe_approvals"] == 1 and m["unsafe_approval_rate_pct"] == 25.0
    assert m["escalation_precision_pct"] == 50.0  # 1 of 2 escalations was truly 'escalate'
    assert m["escalation_recall_pct"] == 50.0
