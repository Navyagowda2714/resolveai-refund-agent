"""Deterministic offline stand-in for an LLM.

``ScriptedLLM`` routes each request to a *handler* chosen by a ``<task>name</task>`` tag in the
system prompt. Handlers receive the JSON context embedded in the last user message and return a
JSON-serialisable answer. This keeps tests and the offline demo fully deterministic while
exercising exactly the same prompt/parse code path as a real provider.

Offline mode measures the *pipeline* (tools, rules, guardrails, routing); it does not measure
the reasoning quality of a real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from resolveai.llm.base import LLMResponse, Message, estimate_tokens, parse_json_block

Handler = Callable[[dict[str, Any]], dict[str, Any]]

_TASK = re.compile(r"<task>(\w+)</task>")


class ScriptedLLM:
    """Deterministic, heuristic LLM stand-in."""

    def __init__(
        self,
        handlers: dict[str, Handler],
        model: str = "offline-scripted",
        base_latency_ms: float = 400.0,
        ms_per_output_token: float = 8.0,
    ) -> None:
        self.handlers = handlers
        self.model = model
        self.base_latency_ms = base_latency_ms
        self.ms_per_output_token = ms_per_output_token
        self.calls: list[str] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        match = _TASK.search(system or "")
        if not match or match.group(1) not in self.handlers:
            raise ValueError(f"ScriptedLLM has no handler for task in system prompt: {system!r}")
        task = match.group(1)
        self.calls.append(task)
        context = parse_json_block(messages[-1]["content"]) if messages else {}
        answer = self.handlers[task](context)
        text = json.dumps(answer)
        prompt_text = (system or "") + "".join(m["content"] for m in messages)
        out_tokens = estimate_tokens(text)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=out_tokens,
            latency_ms=self.base_latency_ms + self.ms_per_output_token * out_tokens,
            model=self.model,
            simulated=True,
        )
