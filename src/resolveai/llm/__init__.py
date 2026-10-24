"""LLM abstraction: protocol, real providers, deterministic offline stand-in, factory."""

from __future__ import annotations

import logging
import os

from resolveai.llm.base import LLMClient, LLMResponse, Message, parse_json_block
from resolveai.llm.offline import Handler, ScriptedLLM

logger = logging.getLogger(__name__)

__all__ = [
    "Handler",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ScriptedLLM",
    "get_llm",
    "parse_json_block",
]

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def get_llm(
    provider: str | None = None, offline_handlers: dict[str, Handler] | None = None
) -> LLMClient:
    """Build the LLM client selected by ``LLM_PROVIDER`` (anthropic|openai|ollama|offline).

    Falls back to the offline ``ScriptedLLM`` with a warning when the selected provider's API key
    is missing, so the project always runs end to end.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "")).lower().strip()
    if not provider:
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "offline"
    key_env = _KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        logger.warning("%s not set; falling back to offline ScriptedLLM", key_env)
        provider = "offline"

    if provider == "anthropic":
        from resolveai.llm.providers import AnthropicProvider

        return AnthropicProvider()
    if provider == "openai":
        from resolveai.llm.providers import OpenAIProvider

        return OpenAIProvider()
    if provider == "ollama":
        from resolveai.llm.providers import OllamaProvider

        return OllamaProvider()
    if provider != "offline":
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    if offline_handlers is None:
        from resolveai.agent.offline_policy import OFFLINE_HANDLERS

        offline_handlers = OFFLINE_HANDLERS
    return ScriptedLLM(offline_handlers)
