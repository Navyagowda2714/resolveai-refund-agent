"""Command-line interface: ``resolveai demo | eval | seed | resolve | queue | serve``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from resolveai.config import configure_logging, get_settings
from resolveai.data.db import CommerceDB
from resolveai.data.seed import seed_database
from resolveai.demo import SHOWCASE_TITLES, build_showcase
from resolveai.render import render_resolution
from resolveai.service import ResolveService

app = typer.Typer(help="ResolveAI: customer issues, resolved (safely).", no_args_is_help=True)
console = Console()


def _service(db: CommerceDB) -> ResolveService:
    return ResolveService(db)


@app.command()
def demo(
    show_packets: bool = typer.Option(False, help="Print the escalation handoff packets."),
) -> None:
    """Run five showcase tickets (one per channel) end to end with readable traces."""
    configure_logging("WARNING")
    db = CommerceDB(":memory:")
    seed_database(db, n_customers=60)
    tickets = build_showcase(db)
    svc = _service(db)
    console.rule(
        f"[bold]ResolveAI demo[/]  model={svc.agent.llm.model}  policy={svc.policy.version}"
    )
    summary = Table("ticket", "channel", "outcome", "amount", "conf", "tools", title="Summary")
    for title, (channel, payload) in zip(SHOWCASE_TITLES, tickets, strict=True):
        ticket = svc.ingest(channel, payload)
        res = svc.resolve(ticket.id)
        render_resolution(console, ticket, res, title=title)
        summary.add_row(
            title,
            channel,
            res.outcome.value,
            f"{res.amount_refunded:.2f}",
            f"{res.proposed.confidence:.2f}",
            str(len(res.tool_calls)),
        )
    console.print(summary)
    queue = db.list_escalations("open")
    console.print(f"\n[bold]Human queue:[/] {len(queue)} open escalation(s)")
    for e in queue:
        console.print(f"  {e['id']} [{e['priority']}] {e['packet']['reasoning_summary']}")
        if show_packets:
            console.print_json(json.dumps(e["packet"], default=str))


@app.command(name="eval")
def run_eval_cmd(
    out_dir: Path = typer.Option(Path("evals"), help="Where to write results."),
) -> None:
    """Run the labelled evaluation set (with and without the safety gate)."""
    from resolveai.evaluation.runner import run_eval

    configure_logging("WARNING")
    result = run_eval(out_dir)
    console.print((out_dir / "results.md").read_text(encoding="utf-8"))
    console.print(f"[dim]model={result['model']} -> {out_dir}/results.json[/]")


@app.command()
def seed(
    customers: int = typer.Option(150, help="Number of background customers."),
    with_showcase: bool = typer.Option(True, help="Also insert the 5 showcase customers."),
) -> None:
    """Create/populate the SQLite database at RESOLVEAI_DB_PATH."""
    db = CommerceDB(get_settings().db_path)
    s = seed_database(db, n_customers=customers)
    if with_showcase:
        path = Path(get_settings().db_path).with_suffix(".showcase.json")
        path.write_text(json.dumps(build_showcase(db), indent=2, default=str), encoding="utf-8")
        console.print(f"showcase payloads written to {path}")
    console.print(
        f"seeded {s.customers} customers, {s.orders} orders, {s.refunds} refunds "
        f"-> {get_settings().db_path}"
    )


@app.command()
def resolve(
    channel: str = typer.Argument(..., help="email | chat | phone | in_app | portal"),
    payload_file: Path = typer.Argument(..., exists=True, help="JSON payload file."),
) -> None:
    """Ingest one raw channel payload and resolve it against the configured database."""
    db = CommerceDB(get_settings().db_path)
    svc = _service(db)
    ticket = svc.ingest(channel, json.loads(payload_file.read_text(encoding="utf-8")))
    render_resolution(console, ticket, svc.resolve(ticket.id))


@app.command()
def queue() -> None:
    """List open escalations in the human queue."""
    db = CommerceDB(get_settings().db_path)
    table = Table("id", "priority", "ticket", "why escalated")
    for e in db.list_escalations("open"):
        table.add_row(
            e["id"], e["priority"], e["ticket_id"], "; ".join(e["packet"]["why_escalated"])
        )
    console.print(table)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("resolveai.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
