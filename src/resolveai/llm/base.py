"""Provider-agnostic LLM interface."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class LLMResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    simulated: bool = Field(
        default=False, description="True when latency/tokens are estimated (offline mode)."
    )

    def json_payload(self) -> Any:
        """Parse the response text as JSON (tolerates ```json fences)."""
        return parse_json_block(self.text)


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_block(text: str) -> Any:
    text = text.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for offline accounting."""
    return max(1, len(text) // 4)
