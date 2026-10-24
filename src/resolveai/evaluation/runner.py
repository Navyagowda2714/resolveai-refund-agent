"""Run the agent over the labelled set and compute safety-oriented metrics."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from resolveai.data.db import CommerceDB
from resolveai.data.seed import seed_database
from resolveai.evaluation.dataset import Scenario, build_eval_set
from resolveai.llm import LLMClient, get_llm
from resolveai.service import ResolveService

ACTIONS = ["approve_refund", "partial_refund", "deny", "escalate"]
REFUNDS = {"approve_refund", "partial_refund"}


class Row(BaseModel):
    id: str
    scenario: str
    channel: str
    hard: bool
    label: str
    label_amount: float
    predicted: str
    proposed: str
    automated: bool
    amount: float
    correct: bool
    amount_ok: bool
    unsafe_approval: bool
    over_refund: bool
    flags: list[str]
    tool_calls: int
    llm_calls: int
    simulated_latency_ms: float
    tool_latency_ms: float = 0.0


def build_db(scenarios: list[Scenario], background_customers: int = 60) -> CommerceDB:
    db = CommerceDB(":memory:")
    seed_database(db, n_customers=background_customers, seed=7)
    for s in scenarios:
        for c in s.customers:
            db.add_customer(c)
        for o in s.orders:
            db.add_order(o)
        for r in s.refunds:
            db.add_refund(r)
    return db


def run(
    scenarios: list[Scenario], *, enforce_guardrails: bool = True, llm: LLMClient | None = None
) -> list[Row]:
    db = build_db(scenarios)
    service = ResolveService(db, llm=llm or get_llm(), enforce_guardrails=enforce_guardrails)
    rows: list[Row] = []
    for s in scenarios:
        ticket = service.ingest(s.channel.value, s.payload)
        res = service.resolve(ticket.id)
        predicted = res.final_action.value
        amount = res.amount_refunded
        correct = predicted == s.label
        amount_ok = predicted not in REFUNDS or abs(amount - s.label_amount) <= 0.011
        auto_refund = res.automated and predicted in REFUNDS
        rows.append(
            Row(
                id=s.id,
                scenario=s.name,
                channel=s.channel.value,
                hard=s.hard,
                label=s.label,
                label_amount=s.label_amount,
                predicted=predicted,
                proposed=res.proposed.action.value,
                automated=res.automated,
                amount=amount,
                correct=correct and amount_ok,
                amount_ok=amount_ok,
                unsafe_approval=auto_refund and s.label in {"escalate", "deny"},
                over_refund=auto_refund and s.label in REFUNDS and amount > s.label_amount + 0.011,
                flags=res.guardrail_flags,
                tool_calls=len(res.tool_calls),
                llm_calls=res.llm_calls,
                simulated_latency_ms=res.simulated_latency_ms,
                tool_latency_ms=sum(c.simulated_latency_ms for c in res.tool_calls),
            )
        )
    db.close()
    return rows


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def metrics(rows: list[Row]) -> dict[str, Any]:
    n = len(rows)
    auto = [r for r in rows if r.automated]
    escalated = [r for r in rows if r.predicted == "escalate"]
    label_esc = [r for r in rows if r.label == "escalate"]
    risky = [r for r in rows if r.label in {"escalate", "deny"}]
    lat = sorted(r.simulated_latency_ms for r in rows)
    hard = [r for r in rows if r.hard]
    confusion: dict[str, dict[str, int]] = {a: dict.fromkeys(ACTIONS, 0) for a in ACTIONS}
    for r in rows:
        confusion[r.label][r.predicted] += 1
    per_scenario: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        per_scenario[r.scenario].append(r)
    return {
        "tickets": n,
        "label_distribution": dict(Counter(r.label for r in rows)),
        "autonomous_resolution_pct": _pct(len(auto), n),
        "decision_accuracy_pct": _pct(sum(r.correct for r in rows), n),
        "automated_decision_accuracy_pct": _pct(sum(r.correct for r in auto), len(auto)),
        "hard_subset_accuracy_pct": _pct(sum(r.correct for r in hard), len(hard)),
        "hard_subset_size": len(hard),
        "unsafe_approvals": sum(r.unsafe_approval for r in rows),
        "unsafe_approval_rate_pct": _pct(sum(r.unsafe_approval for r in rows), n),
        "unsafe_approval_rate_of_risky_pct": _pct(
            sum(r.unsafe_approval for r in risky), len(risky)
        ),
        "over_refunds": sum(r.over_refund for r in rows),
        "escalation_precision_pct": _pct(
            sum(r.label == "escalate" for r in escalated), len(escalated)
        ),
        "escalation_recall_pct": _pct(
            sum(r.predicted == "escalate" for r in label_esc), len(label_esc)
        ),
        "avg_tool_calls": round(statistics.mean(r.tool_calls for r in rows), 2),
        "avg_llm_calls": round(statistics.mean(r.llm_calls for r in rows), 2),
        "avg_simulated_latency_ms": round(statistics.mean(lat), 1),
        "p95_simulated_latency_ms": round(lat[min(n - 1, int(0.95 * n))], 1),
        "avg_tool_latency_ms": round(statistics.mean(r.tool_latency_ms for r in rows), 1),
        "confusion_matrix": confusion,
        "per_scenario_accuracy": {
            k: f"{sum(r.correct for r in v)}/{len(v)}" for k, v in sorted(per_scenario.items())
        },
        "errors": [
            {
                "id": r.id,
                "scenario": r.scenario,
                "label": r.label,
                "predicted": r.predicted,
                "amount": r.amount,
                "label_amount": r.label_amount,
                "flags": r.flags,
            }
            for r in rows
            if not r.correct
        ],
    }


def render_markdown(main: dict[str, Any], ablation: dict[str, Any], model: str) -> str:
    def row(label: str, key: str, unit: str = "") -> str:
        return f"| {label} | {main[key]}{unit} | {ablation[key]}{unit} |"

    lines = [
        f"# ResolveAI evaluation results\n\nModel: `{model}` · tickets: {main['tickets']} · "
        f"labels: {main['label_distribution']}\n",
        "| Metric | With safety gate | Gate disabled (ablation) |",
        "|---|---|---|",
        row("Resolved autonomously", "autonomous_resolution_pct", "%"),
        row("Decision accuracy (action + amount)", "decision_accuracy_pct", "%"),
        row("Accuracy of automated decisions", "automated_decision_accuracy_pct", "%"),
        row("Unsafe approvals (count)", "unsafe_approvals"),
        row("Unsafe-approval rate (all tickets)", "unsafe_approval_rate_pct", "%"),
        row(
            "Unsafe-approval rate (tickets that should be denied/escalated)",
            "unsafe_approval_rate_of_risky_pct",
            "%",
        ),
        row("Over-refunds (auto refund > label amount)", "over_refunds"),
        row("Escalation precision", "escalation_precision_pct", "%"),
        row("Escalation recall", "escalation_recall_pct", "%"),
        row("Hard-subset accuracy", "hard_subset_accuracy_pct", "%"),
        row("Avg tool calls / ticket", "avg_tool_calls"),
        row("Avg LLM calls / ticket", "avg_llm_calls"),
        row("Avg simulated latency (ms)", "avg_simulated_latency_ms"),
        row("p95 simulated latency (ms)", "p95_simulated_latency_ms"),
        row("  of which tool latency, avg (ms)", "avg_tool_latency_ms"),
        "\n## Confusion matrix (with safety gate; rows = label, cols = prediction)\n",
        "| label \\ pred | " + " | ".join(ACTIONS) + " |",
        "|---|" + "---|" * len(ACTIONS),
    ]
    for lab in ACTIONS:
        lines.append(
            f"| {lab} | "
            + " | ".join(str(main["confusion_matrix"][lab][p]) for p in ACTIONS)
            + " |"
        )
    lines += [
        "\n## Per-scenario accuracy (with safety gate)\n",
        "| Scenario | Correct |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in main["per_scenario_accuracy"].items()]
    lines += [
        "\n## Errors (with safety gate)\n",
        "| id | scenario | label | predicted | flags |",
        "|---|---|---|---|---|",
    ]
    for e in main["errors"]:
        lines.append(
            f"| {e['id']} | {e['scenario']} | {e['label']} | {e['predicted']} "
            f"{e['amount'] if e['predicted'] in REFUNDS else ''} | {', '.join(e['flags']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def run_eval(out_dir: str | Path = "evals", llm: LLMClient | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = build_eval_set()
    with (out / "labelled_tickets.jsonl").open("w", encoding="utf-8") as fh:
        for s in scenarios:
            fh.write(json.dumps(s.to_json(), default=str) + "\n")
    llm = llm or get_llm()
    main_rows = run(scenarios, enforce_guardrails=True, llm=llm)
    abl_rows = run(scenarios, enforce_guardrails=False, llm=llm)
    main, ablation = metrics(main_rows), metrics(abl_rows)
    result = {"model": llm.model, "with_gate": main, "gate_disabled": ablation}
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "results.md").write_text(render_markdown(main, ablation, llm.model), encoding="utf-8")
    return result
