"""Heuristic handlers that let ``ScriptedLLM`` play the LLM's role deterministically offline.

They implement a sensible investigation order and a decision heuristic that follows the policy
evaluation and discounts confidence for ambiguity. They are a stand-in so the full agent loop runs
without an API key; they are NOT a claim about real-model quality.
"""

from __future__ import annotations

from typing import Any

from resolveai.llm.offline import Handler

EARLY_EXIT_ISSUES = {"unauthorized_charge", "other"}


def _result(ctx: dict[str, Any], tool: str) -> dict[str, Any] | None:
    for obs in ctx.get("observations", []):
        if obs["tool"] == tool and obs.get("result") is not None:
            return obs["result"]
    return None


def _called(ctx: dict[str, Any], tool: str) -> bool:
    return any(o["tool"] == tool for o in ctx.get("observations", []))


def next_action(ctx: dict[str, Any]) -> dict[str, Any]:
    ticket = ctx["ticket"]
    email = ticket["customer_email"]

    if not _called(ctx, "fetch_order_details"):
        return {
            "thought": "Start by fetching the order the customer is asking about.",
            "tool": "fetch_order_details",
            "arguments": {"order_id": ticket.get("order_id"), "customer_email": email},
        }
    order_res = _result(ctx, "fetch_order_details") or {"found": False}
    order = order_res.get("order") or {}
    items = [{"sku": i["sku"], "name": i["name"]} for i in order.get("items", [])]

    if not _called(ctx, "analyze_issue"):
        return {
            "thought": "Classify the issue and check which items are affected.",
            "tool": "analyze_issue",
            "arguments": {
                "text": f"{ticket.get('subject', '')}\n{ticket['body']}",
                "attachments": ticket.get("attachments", []),
                "order_items": items,
            },
        }
    analysis = _result(ctx, "analyze_issue") or {"issue_type": "other", "triggers": []}
    issue = analysis["issue_type"]

    if (
        analysis.get("triggers")
        or issue in EARLY_EXIT_ISSUES
        or not order_res.get("found")
        or not order_res.get("belongs_to_customer")
    ):
        return {"thought": "This clearly needs a human; stop investigating.", "finish": True}

    affected = [i for i in order.get("items", []) if i["sku"] in analysis.get("affected_skus", [])]
    focus = affected or order.get("items", [])
    category = focus[0]["category"] if focus else "default"

    if not _called(ctx, "check_refund_policy"):
        return {
            "thought": f"Check the refund policy for {category} / {issue}.",
            "tool": "check_refund_policy",
            "arguments": {"category": category, "issue_type": issue},
        }
    pol = _result(ctx, "check_refund_policy") or {}
    days = order_res.get("days_since_delivery")
    if (
        issue not in {"defective", "not_received"}
        and days is not None
        and days > pol.get("window_days", 30)
    ):
        return {"thought": "Outside the return window; no refund possible.", "finish": True}

    if not _called(ctx, "review_customer_history"):
        return {
            "thought": "Review the customer's order and refund history.",
            "tool": "review_customer_history",
            "arguments": {"customer_email": email},
        }
    if not _called(ctx, "run_fraud_checks"):
        claimed = round(sum(i["unit_price"] * i["quantity"] for i in focus), 2)
        return {
            "thought": "Run fraud checks before any money moves.",
            "tool": "run_fraud_checks",
            "arguments": {
                "customer_email": email,
                "order_id": order.get("id"),
                "claimed_amount": claimed,
                "issue_type": issue,
            },
        }
    return {"thought": "I have enough evidence to decide.", "finish": True}


def decide(ctx: dict[str, Any]) -> dict[str, Any]:
    rules = ctx["policy_evaluation"]
    analysis = ctx["findings"].get("analysis") or {"confidence": 0.3, "issue_type": "other"}
    fraud = ctx["findings"].get("fraud")
    a_conf = float(analysis.get("confidence", 0.3))

    if rules["escalate_reasons"]:
        return {
            "action": "escalate",
            "amount": 0.0,
            "confidence": 0.9,
            "reasons": rules["escalate_reasons"],
            "policy_citations": rules["citations"],
        }
    if rules["deny_reasons"]:
        conf = round(min(0.97, 0.55 + 0.45 * a_conf), 3)
        return {
            "action": "deny",
            "amount": 0.0,
            "confidence": conf,
            "reasons": rules["deny_reasons"],
            "policy_citations": rules["citations"],
        }

    reasons = [f"issue classified as {analysis['issue_type']} (confidence {a_conf:.2f})"]
    conf = a_conf
    if any("not identified" in n for n in rules.get("notes", [])):
        conf -= 0.15
        reasons.append("affected item unclear")
    if analysis.get("secondary_issues"):
        conf -= 0.05
        reasons.append(f"secondary signals: {', '.join(analysis['secondary_issues'])}")
    if any(not f.endswith("no description available") for f in analysis.get("image_findings", [])):
        conf += 0.05
        reasons.append("photo evidence provided")
    if fraud is None:
        return {
            "action": "escalate",
            "amount": 0.0,
            "confidence": 0.5,
            "reasons": ["fraud check missing"],
            "policy_citations": rules["citations"],
        }
    if fraud["level"] == "high":
        return {
            "action": "escalate",
            "amount": 0.0,
            "confidence": 0.85,
            "reasons": [f"high fraud risk ({fraud['score']})"],
            "policy_citations": ["POL-9.1"],
        }
    if fraud["level"] == "medium":
        conf -= 0.1
        reasons.append(f"medium fraud risk ({fraud['score']})")
    reasons += rules.get("notes", [])
    return {
        "action": rules["recommended_action"],
        "amount": rules["refund_amount"],
        "confidence": round(max(0.0, min(0.99, conf)), 3),
        "reasons": reasons,
        "policy_citations": rules["citations"],
    }


OFFLINE_HANDLERS: dict[str, Handler] = {"next_action": next_action, "decide": decide}
