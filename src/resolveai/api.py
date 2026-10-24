"""FastAPI service: ticket intake, resolution and the human escalation queue."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from resolveai import __version__
from resolveai.config import get_settings
from resolveai.data.db import CommerceDB
from resolveai.intake import IntakeError
from resolveai.models import Channel, DecisionAction, Resolution, Ticket
from resolveai.service import NotFoundError, ResolveService


class TicketIn(BaseModel):
    channel: Channel
    payload: dict[str, Any] = Field(description="Raw payload exactly as the channel delivers it.")


class TicketOut(BaseModel):
    ticket: Ticket
    status: str


class HumanDecisionIn(BaseModel):
    action: DecisionAction
    amount: float = Field(0.0, ge=0)
    agent: str = Field(description="Human agent id")
    notes: str = ""


def create_app(service: ResolveService | None = None) -> FastAPI:
    app = FastAPI(
        title="ResolveAI",
        version=__version__,
        description="Customer issues, resolved (safely). Refund agent with a human queue.",
    )
    state: dict[str, ResolveService] = {}

    def get_service() -> ResolveService:
        if "svc" not in state:
            state["svc"] = service or ResolveService(CommerceDB(get_settings().db_path))
        return state["svc"]

    @app.get("/health")
    def health(svc: ResolveService = Depends(get_service)) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": svc.agent.llm.model,
            "policy": svc.policy.version,
            "db": svc.db.counts(),
        }

    @app.post("/tickets", response_model=TicketOut, status_code=201)
    def create_ticket(body: TicketIn, svc: ResolveService = Depends(get_service)) -> TicketOut:
        try:
            ticket = svc.ingest(body.channel.value, body.payload)
        except IntakeError as exc:
            raise HTTPException(422, str(exc)) from exc
        return TicketOut(ticket=ticket, status="open")

    @app.get("/tickets", response_model=list[TicketOut])
    def list_tickets(status: str | None = None, svc: ResolveService = Depends(get_service)) -> Any:
        return [TicketOut(ticket=t, status=s) for t, s in svc.db.list_tickets(status)]

    @app.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: str, svc: ResolveService = Depends(get_service)) -> dict[str, Any]:
        found = svc.db.get_ticket(ticket_id)
        if not found:
            raise HTTPException(404, "ticket not found")
        res = svc.db.get_resolution(ticket_id)
        return {
            "ticket": found[0].model_dump(mode="json"),
            "status": found[1],
            "resolution": res.model_dump(mode="json") if res else None,
        }

    @app.post("/tickets/{ticket_id}/resolve", response_model=Resolution)
    def resolve(ticket_id: str, svc: ResolveService = Depends(get_service)) -> Resolution:
        try:
            return svc.resolve(ticket_id)
        except NotFoundError as exc:
            raise HTTPException(404, "ticket not found") from exc

    @app.get("/queue")
    def queue(svc: ResolveService = Depends(get_service)) -> list[dict[str, Any]]:
        return [
            {
                "id": e["id"],
                "ticket_id": e["ticket_id"],
                "priority": e["priority"],
                "created_at": e["created_at"],
                "why": e["packet"]["why_escalated"],
                "summary": e["packet"]["reasoning_summary"],
            }
            for e in svc.db.list_escalations("open")
        ]

    @app.get("/queue/{esc_id}")
    def queue_item(esc_id: str, svc: ResolveService = Depends(get_service)) -> dict[str, Any]:
        item = svc.db.get_escalation(esc_id)
        if not item:
            raise HTTPException(404, "escalation not found")
        return item

    @app.post("/queue/{esc_id}/decision")
    def decide(
        esc_id: str, body: HumanDecisionIn, svc: ResolveService = Depends(get_service)
    ) -> dict[str, Any]:
        try:
            return svc.human_decide(
                esc_id, action=body.action, amount=body.amount, agent=body.agent, notes=body.notes
            )
        except NotFoundError as exc:
            raise HTTPException(404, "escalation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return app


app = create_app()
