"""Real LLM providers. SDKs are imported lazily so the package works without them."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from resolveai.llm.base import LLMResponse, Message


class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK (``pip install resolveai[anthropic]``)."""

    DEFAULT_MODEL = "claude-sonnet-5-5"
    SMALL_MODEL = "claude-haiku-5-5"

    def __init__(self, model: str | None = None, max_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the extra: pip install 'resolveai[anthropic]'") from exc
        self._client = anthropic.Anthropic()
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": list(messages),
        }
        if system:
            kwargs["system"] = system
        if json_schema is not None:
            # Structured output via a forced tool call: the tool input *is* the JSON object.
            kwargs["tools"] = [
                {
                    "name": "respond",
                    "description": "Return the answer.",
                    "input_schema": json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "respond"}
        start = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        text = ""
        for block in resp.content:
            if block.type == "tool_use":
                text = json.dumps(block.input)
                break
            if block.type == "text":
                text += block.text
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            model=self.model,
        )


class OpenAIProvider:
    """OpenAI chat completions (``pip install resolveai[openai]``)."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the extra: pip install 'resolveai[openai]'") from exc
        self._client = openai.OpenAI()
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": temperature}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model=self.model,
        )


class OllamaProvider:
    """Local models through the Ollama HTTP API."""

    DEFAULT_MODEL = "llama3.1"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        start = time.perf_counter()
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
        )
