"""Which LLM the agent uses: the provider and model are configuration, not code.

    ISR_FORGE_LLM_PROVIDER   openrouter | anthropic   (unset: whichever API key is set, OpenRouter first)
    ISR_FORGE_MODEL          the model id in that provider's naming, e.g. openai/gpt-5.6-sol on OpenRouter
    OPENROUTER_API_KEY / ANTHROPIC_API_KEY

These are set only in .env (Docker Compose) or the shell environment; the CLI and the web form cannot
override them, so whoever uses the form cannot pick a model billed to the operator's key. Every adapter implements `StructuredLLM`, so the agent nodes never know which vendor answered.
"""

import os
import re
from dataclasses import dataclass
from typing import Callable, Final, Optional

from src.agent.llm import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
from src.agent.llm import MODEL_ENV, StructuredLLM

PROVIDER_ENV: Final[str] = "ISR_FORGE_LLM_PROVIDER"
MODEL_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}")


class LLMConfigError(ValueError):
    """The provider or model setting is invalid, or the provider's API key is missing."""


def _claude(model: str) -> StructuredLLM:
    from src.agent.llm import ClaudeLLM

    return ClaudeLLM(model)


def _openrouter(model: str) -> StructuredLLM:
    from src.agent.openrouter import OpenRouterLLM  # imported lazily: the openai SDK is only needed here

    return OpenRouterLLM(model)


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str
    default_model: str
    build: Callable[[str], StructuredLLM]


PROVIDERS: Final[dict[str, Provider]] = {
    "openrouter": Provider("openrouter", "OPENROUTER_API_KEY", "anthropic/claude-opus-5.5", _openrouter),
    "anthropic": Provider("anthropic", "ANTHROPIC_API_KEY", CLAUDE_DEFAULT_MODEL, _claude),
}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _has_key(provider: Provider) -> bool:
    return bool(_env(provider.key_env))


def resolve_provider(name: Optional[str] = None) -> Provider:
    """`name`, else ISR_FORGE_LLM_PROVIDER, else the first provider whose API key is set (OpenRouter first)."""
    chosen: str = (name or _env(PROVIDER_ENV)).lower()
    if chosen:
        if chosen not in PROVIDERS:
            raise LLMConfigError(f"unknown LLM provider {chosen!r}; choose one of {', '.join(PROVIDERS)}")
        return PROVIDERS[chosen]
    return next((p for p in PROVIDERS.values() if _has_key(p)), next(iter(PROVIDERS.values())))


def resolve_model(provider: Provider, model: Optional[str] = None) -> str:
    """`model`, else ISR_FORGE_MODEL, else the provider's default; validated as a plain model id."""
    chosen: str = (model or "").strip() or _env(MODEL_ENV) or provider.default_model
    if not MODEL_ID_PATTERN.fullmatch(chosen):
        raise LLMConfigError(f"invalid model id {chosen[:60]!r}: use letters, digits and . _ : / @ - only")
    return chosen


def config_problem(name: Optional[str] = None) -> Optional[str]:
    """Why no LLM can be built (a missing key or a bad provider name), or None when one can."""
    try:
        provider = resolve_provider(name)
    except LLMConfigError as exc:
        return str(exc)
    if _has_key(provider):
        return None
    if name or _env(PROVIDER_ENV):
        return f"set {provider.key_env} for the {provider.name} provider"
    return " or ".join(f"set {p.key_env}" for p in PROVIDERS.values())


def describe(name: Optional[str] = None, model: Optional[str] = None) -> str:
    """`provider · model` for display; raises LLMConfigError on a bad setting."""
    provider = resolve_provider(name)
    return f"{provider.name} · {resolve_model(provider, model)}"


def make_llm(name: Optional[str] = None, model: Optional[str] = None) -> StructuredLLM:
    """Build the configured LLM; raises LLMConfigError instead of failing later at the first request."""
    problem: Optional[str] = config_problem(name)
    if problem:
        raise LLMConfigError(problem)
    provider = resolve_provider(name)
    return provider.build(resolve_model(provider, model))
