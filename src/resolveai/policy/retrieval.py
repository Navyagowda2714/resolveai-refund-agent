"""Policy document retrieval: a dependency-free BM25 index over policy sections."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "is",
    "are",
    "be",
    "for",
    "in",
    "on",
    "by",
    "with",
    "if",
    "that",
    "this",
    "as",
    "it",
    "its",
    "after",
    "within",
    "their",
    "any",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class PolicySection(BaseModel):
    id: str
    title: str
    text: str


class RetrievedSection(BaseModel):
    id: str
    title: str
    excerpt: str
    score: float


def load_sections(path: str | Path) -> list[PolicySection]:
    sections: list[PolicySection] = []
    current: PolicySection | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(POL-[\d.]+)\s+(.*)$", line)
        if m:
            if current:
                sections.append(current)
            current = PolicySection(id=m.group(1), title=m.group(2).strip(), text="")
        elif current and line.strip():
            current.text = f"{current.text} {line.strip()}".strip()
    if current:
        sections.append(current)
    return sections


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over policy sections (title boosted by repetition)."""

    def __init__(self, sections: list[PolicySection], k1: float = 1.5, b: float = 0.75) -> None:
        self.sections = sections
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{s.title} {s.title} {s.text}") for s in sections]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[RetrievedSection]:
        q = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:k]:
            s = self.sections[i]
            out.append(
                RetrievedSection(
                    id=s.id, title=s.title, excerpt=s.text[:280], score=round(score, 3)
                )
            )
        return out

    def get(self, section_id: str) -> PolicySection | None:
        return next((s for s in self.sections if s.id == section_id), None)
