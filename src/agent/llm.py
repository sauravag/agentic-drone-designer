"""The only place ISR-Forge talks to an LLM: schema-constrained calls via the Anthropic SDK.

Nodes depend on the `StructuredLLM` protocol, so tests and offline runs can inject a scripted model.
Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth login` profile); nothing is
hard-coded. The model is configurable with ISR_FORGE_MODEL.
"""

import logging
import os
from typing import Final, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-opus-5-5"
MODEL_ENV: Final[str] = "ISR_FORGE_MODEL"
MAX_TOKENS: Final[int] = 16_000
FALLBACK_BETA: Final[str] = "server-side-fallback-2026-07-01"  # re-run a policy-declined request on a fallback model

Effort = Literal["low", "medium", "high"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM call failed, was refused, or returned nothing that matches the schema."""


class StructuredLLM(Protocol):
    def parse(self, system: str, prompt: str, schema: type[SchemaT], effort: Effort = "medium") -> SchemaT:
        """Return an instance of `schema` for `prompt` under `system` instructions."""
        ...


class ClaudeLLM:
    """Claude via `client.beta.messages.parse` (structured outputs + server-side refusal fallback)."""

    def __init__(self, model: str | None = None, client: anthropic.Anthropic | None = None) -> None:
        self.model: str = model or os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        self._client: anthropic.Anthropic = client or anthropic.Anthropic()

    def parse(self, system: str, prompt: str, schema: type[SchemaT], effort: Effort = "medium") -> SchemaT:
        try:
            response = self._client.beta.messages.parse(
                model=self.model, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": prompt}], output_format=schema,
                output_config={"effort": effort}, betas=[FALLBACK_BETA], fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"cannot reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:  # auth, bad request, rate limit after SDK retries, 5xx
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except (anthropic.AnthropicError, TypeError, ValueError) as exc:
            # missing credentials surface as TypeError at request time; schema/JSON problems as ValueError
            # (pydantic.ValidationError is a ValueError)
            raise LLMError(f"Claude call failed ({type(exc).__name__}): {exc}") from exc
        logger.info("[Tool] llm.parse model=%s schema=%s request_id=%s stop=%s", self.model, schema.__name__,
                    response._request_id, response.stop_reason)  # _request_id is the SDK's public request-id accessor
        if response.stop_reason == "refusal":
            raise LLMError("the model declined the request")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError(f"no {schema.__name__} in the response (stop_reason={response.stop_reason})")
        return response.parsed_output
