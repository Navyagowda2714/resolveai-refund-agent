"""Readable terminal rendering of resolution traces (rich)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from resolveai.models import Outcome, Resolution, Ticket

KIND_STYLE = {
    "intake": "cyan",
    "thought": "dim italic",
    "tool_call": "blue",
    "rules": "magenta",
    "decision": "yellow",
    "guardrail": "bold red",
    "action": "bold green",
}
OUTCOME_STYLE = {
    Outcome.AUTO_APPROVED: "bold green",
    Outcome.AUTO_PARTIAL: "bold green",
    Outcome.AUTO_DENIED: "bold yellow",
    Outcome.ESCALATED: "bold red",
}


def render_resolution(console: Console, ticket: Ticket, res: Resolution, title: str = "") -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("step", width=10)
    table.add_column("detail", overflow="fold")
    for ev in res.trace:
        style = KIND_STYLE.get(ev.kind, "")
        table.add_row(str(ev.step), f"[{style}]{ev.kind}[/]", ev.summary)
    p = res.proposed
    footer = (
        f"[{OUTCOME_STYLE[res.outcome]}]{res.outcome.value.upper()}[/]  "
        f"proposed={p.action.value} {p.amount:.2f}  confidence={p.confidence:.2f}  "
        f"citations={', '.join(p.policy_citations) or '-'}\n"
        f"tool calls={len(res.tool_calls)}  llm calls={res.llm_calls}  "
        f"simulated latency={res.simulated_latency_ms / 1000:.2f}s"
        + (f"  escalation={res.escalation_id}" if res.escalation_id else "")
        + (f"  refund={res.refund_id}" if res.refund_id else "")
    )
    console.print(
        Panel.fit(
            table,
            title=f"[bold]{title or ticket.id}[/]  ({ticket.channel.value}, {ticket.id})",
            subtitle=None,
            border_style="grey50",
        )
    )
    console.print(footer)
    console.print(Panel(res.customer_message, title="message to customer", border_style="grey35"))
    console.print()
