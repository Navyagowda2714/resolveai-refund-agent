"""Issue analysis: lexicon classifier with negation handling + image-description hook.

This is the *deterministic* analysis tool the agent calls. It is intentionally transparent: every
classification lists the evidence phrases that produced it, so a human reviewer can audit it.
A real LLM in the decision step still sees the raw ticket text and may disagree with it.
"""

from __future__ import annotations

import re
from typing import Protocol

from resolveai.models import Attachment
from resolveai.tools.schemas import IssueAnalysis

ISSUE_LEXICON: dict[str, list[str]] = {
    "damaged": [
        "damaged",
        "broken",
        "cracked",
        "shattered",
        "smashed",
        "crushed",
        "dented",
        "scratched",
        "torn",
        "ripped",
        "leaking",
        "in pieces",
        "chipped",
        "bent",
    ],
    "defective": [
        "defective",
        "faulty",
        "stopped working",
        "doesn't work",
        "does not work",
        "won't turn on",
        "will not turn on",
        "not working",
        "malfunction",
        "keeps disconnecting",
        "won't charge",
        "doesn't charge",
        "dead on arrival",
        "keeps turning off",
        "stopped charging",
        "error code",
    ],
    "wrong_item": [
        "wrong item",
        "wrong size",
        "wrong colour",
        "wrong color",
        "different item",
        "not what i ordered",
        "instead of",
        "sent me the wrong",
        "wrong product",
    ],
    "missing_item": [
        "missing",
        "wasn't in the box",
        "not in the box",
        "not in the package",
        "only received",
        "didn't include",
        "left out",
        "only one of",
    ],
    "not_received": [
        "never arrived",
        "not received",
        "haven't received",
        "didn't receive",
        "never got",
        "hasn't arrived",
        "not delivered",
        "lost package",
        "no sign of",
        "never showed up",
        "still waiting for",
    ],
    "late_delivery": [
        "late",
        "delayed",
        "took weeks",
        "arrived after",
        "days late",
        "past the delivery date",
        "missed the delivery date",
    ],
    "changed_mind": [
        "changed my mind",
        "no longer need",
        "don't need",
        "don't want",
        "doesn't fit",
        "does not fit",
        "too small",
        "too big",
        "don't like",
        "not for me",
        "ordered by mistake",
        "found it cheaper",
        "don't really need",
        "not what i expected",
    ],
    "unauthorized_charge": [
        "didn't order",
        "did not order",
        "don't recognise",
        "don't recognize",
        "unauthorized",
        "unauthorised",
        "fraudulent charge",
        "someone used my card",
        "never placed",
    ],
}

TRIGGER_LEXICON: dict[str, list[str]] = {
    "legal_threat": [
        "lawyer",
        "legal action",
        "sue you",
        "court",
        "attorney",
        "consumer protection",
        "small claims",
    ],
    "chargeback_threat": [
        "chargeback",
        "charge back",
        "dispute with my bank",
        "dispute the charge",
        "call my bank",
        "contact my bank",
        "reverse the charge",
        "credit card company",
        "open a dispute",
    ],
    "safety_hazard": [
        "caught fire",
        "fire",
        "smoke",
        "smoking",
        "burned",
        "burnt",
        "sparks",
        "overheated",
        "overheating",
        "electric shock",
        "injured",
        "injury",
        "melted",
    ],
}

NEGATIONS = {
    "not",
    "no",
    "never",
    "nothing",
    "without",
    "isn't",
    "wasn't",
    "aren't",
    "weren't",
    "hasn't",
    "didn't",
    "doesn't",
    "nor",
}
NEGATION_EXEMPT = {"missing", "late"}  # "not missing anything" is still handled by the window
ANGRY = [
    "ridiculous",
    "unacceptable",
    "furious",
    "terrible",
    "worst",
    "angry",
    "disgusting",
    "scam",
    "!!!",
]
_WORD = re.compile(r"[a-z0-9']+")


class ImageDescriber(Protocol):
    """Hook for a vision model. Return a caption for an image attachment, or None."""

    def describe(self, attachment: Attachment) -> str | None: ...


class AltTextDescriber:
    """Default describer: uses the caption/alt text supplied by the channel, if any."""

    def describe(self, attachment: Attachment) -> str | None:
        return attachment.description


def _find(text: str, phrase: str) -> list[int]:
    pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    return [m.start() for m in re.finditer(pattern, text)]


def _negated(text: str, pos: int) -> bool:
    preceding = _WORD.findall(text[max(0, pos - 40) : pos])[-3:]
    return any(w in NEGATIONS or w.endswith("n't") for w in preceding)


def score_text(text: str, weight: float = 1.0) -> tuple[dict[str, float], list[str]]:
    text = text.lower()
    scores: dict[str, float] = {}
    evidence: list[str] = []
    for issue, phrases in ISSUE_LEXICON.items():
        for phrase in phrases:
            for pos in _find(text, phrase):
                if issue not in NEGATION_EXEMPT and _negated(text, pos):
                    evidence.append(f"negated:{phrase}")
                    continue
                w = weight * (1.5 if " " in phrase else 1.0)
                scores[issue] = scores.get(issue, 0.0) + w
                evidence.append(f"{issue}:{phrase}")
                break  # count each phrase once
    return scores, evidence


def detect_triggers(text: str) -> list[str]:
    text = text.lower()
    found = []
    for trig, phrases in TRIGGER_LEXICON.items():
        if any(_find(text, p) and not _negated(text, _find(text, p)[0]) for p in phrases):
            found.append(trig)
    return found


def match_items(text: str, items: list[dict[str, str]]) -> list[str]:
    tokens = set(_WORD.findall(text.lower()))
    stems = {t.rstrip("s") for t in tokens}
    hits = []
    for item in items:
        words = [w for w in _WORD.findall(item["name"].lower()) if len(w) > 3 and not w.isdigit()]
        if any(w in tokens or w.rstrip("s") in stems for w in words):
            hits.append(item["sku"])
    return hits


_CLAUSE_SPLIT = re.compile(r"[.;!?\n]+|,|\bbut\b|\bhowever\b")


def affected_items(text: str, issue: str, items: list[dict[str, str]]) -> list[str]:
    """Attribute the issue to items mentioned in the same clause as the issue evidence.

    "The shoes and serum are there but the charging hub wasn't in the box" -> only the hub.
    Falls back to every item mentioned anywhere in the text.
    """
    if not items:
        return []
    phrases = ISSUE_LEXICON.get(issue, [])
    clauses = [c for c in _CLAUSE_SPLIT.split(text.lower()) if c and c.strip()]
    local: list[str] = []
    for clause in clauses:
        if any(_find(clause, p) for p in phrases):
            local += [sku for sku in match_items(clause, items) if sku not in local]
    return local or match_items(text, items)


def analyze(
    text: str,
    attachments: list[Attachment] | None = None,
    order_items: list[dict[str, str]] | None = None,
    describer: ImageDescriber | None = None,
) -> IssueAnalysis:
    describer = describer or AltTextDescriber()
    scores, evidence = score_text(text)
    image_findings: list[str] = []
    for att in attachments or []:
        if not att.content_type.startswith("image/"):
            continue
        caption = describer.describe(att)
        if not caption:
            image_findings.append(f"{att.filename}: no description available")
            continue
        image_findings.append(f"{att.filename}: {caption}")
        img_scores, img_ev = score_text(caption, weight=0.75)
        for k, v in img_scores.items():
            scores[k] = scores.get(k, 0.0) + v
        evidence += [f"image:{e}" for e in img_ev]

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        issue, confidence = "other", 0.3
    else:
        issue, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = s1 / (s1 + s2)
        confidence = min(0.96, 0.7 + 0.1 * s1) * (0.55 + 0.45 * margin)
    lowered = text.lower()
    angry = sum(lowered.count(w) for w in ANGRY)
    if any(w in lowered for w in ("replacement", "replace it", "exchange")):
        resolution = "replacement"
    elif any(w in lowered for w in ("refund", "money back", "reimburse")):
        resolution = "refund"
    else:
        resolution = "unspecified"
    return IssueAnalysis(
        issue_type=issue,
        confidence=round(confidence, 3),
        scores={k: round(v, 2) for k, v in ranked},
        secondary_issues=[k for k, v in ranked[1:] if v >= 1.0],
        affected_skus=affected_items(text, issue, order_items or []),
        triggers=detect_triggers(text),
        sentiment="negative" if angry >= 1 else "neutral",
        requested_resolution=resolution,
        image_findings=image_findings,
        evidence=evidence,
    )
