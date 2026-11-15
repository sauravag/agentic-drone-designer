"""OpenRouter adapter: any OpenRouter model behind the `StructuredLLM` protocol.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so this uses the `openai` SDK pointed at
OpenRouter's base URL. `chat.completions.parse(response_format=Schema)` sends the Pydantic schema as a strict
JSON schema and validates the reply; `provider.require_parameters` makes OpenRouter route only to upstream
hosts that honour `response_format`, so a model can never silently ignore the schema.
The key comes from OPENROUTER_API_KEY; nothing is hard-coded.
"""

import logging
import os
from typing import Any, Final

import openai
from pydantic import ValidationError

from src.agent.llm import MAX_TOKENS, Effort, LLMError, SchemaT

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
APP_TITLE: Final[str] = "ISR-Forge"  # shown on the OpenRouter activity page
ROUTING: Final[dict[str, Any]] = {"provider": {"require_parameters": True}}
MAX_UPSTREAM_TEXT: Final[int] = 200  # upstream error text is shown to the user: keep it short


def _clip(text: object) -> str:
    flat: str = " ".join(str(text).split())
    return flat if len(flat) <= MAX_UPSTREAM_TEXT else f"{flat[:MAX_UPSTREAM_TEXT]}…"


def _validation_summary(exc: ValidationError) -> str:
    """Field locations only: the offending values are model output and stay out of the message."""
    fields: str = ", ".join(".".join(str(p) for p in err["loc"]) or "<root>" for err in exc.errors())
    return f"{exc.error_count()} schema error(s) at {_clip(fields)}"


def _raise_body_error(body: Any) -> None:
    """OpenRouter can answer HTTP 200 with {"error": {...}} and no choices (an upstream failure mid-route)."""
    error = body.get("error") if isinstance(body, dict) else None
    if error:
        detail = error if not isinstance(error, dict) else f"{error.get('code', '?')}: {error.get('message', '')}"
        raise LLMError(f"OpenRouter upstream error {_clip(detail)}")


class OpenRouterLLM:
    """Schema-constrained calls to `model` (an OpenRouter slug such as `anthropic/claude-opus-5.5`).

    `effort` is accepted for protocol compatibility but not sent: OpenRouter's `reasoning` parameter would
    exclude non-reasoning models under `require_parameters`."""

    def __init__(self, model: str, client: openai.OpenAI | None = None) -> None:
        self.model: str = model
        self._client: openai.OpenAI = client or openai.OpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=os.environ.get(OPENROUTER_KEY_ENV, "").strip(),
            default_headers={"X-Title": APP_TITLE})

    def parse(self, system: str, prompt: str, schema: type[SchemaT], effort: Effort = "medium") -> SchemaT:
        completion = self._request(system, prompt, schema)
        if not completion.choices:
            raise LLMError(f"OpenRouter returned no choices for {self.model}")
        choice = completion.choices[0]
        logger.info("[Tool] llm.parse provider=openrouter model=%s schema=%s id=%s finish=%s", self.model,
                    schema.__name__, completion.id, choice.finish_reason)
        if choice.message.refusal:
            raise LLMError("the model declined the request")
        if choice.message.parsed is None:
            raise LLMError(f"no {schema.__name__} in the response (finish_reason={choice.finish_reason})")
        return choice.message.parsed

    def _request(self, system: str, prompt: str, schema: type[SchemaT]) -> Any:
        try:
            raw = self._client.chat.completions.with_raw_response.parse(
                model=self.model, max_tokens=MAX_TOKENS, response_format=schema, extra_body=ROUTING,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            _raise_body_error(raw.http_response.json())  # before parse(), which would fail on choices=None
            return raw.parse()
        except openai.APIConnectionError as exc:
            raise LLMError(f"cannot reach OpenRouter: {exc}") from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            # the response body can echo key details: keep it in the server log, show a fixed message
            logger.error("OpenRouter rejected the credentials (%s): %s", exc.status_code, exc.message)
            raise LLMError(f"OpenRouter rejected the credentials (HTTP {exc.status_code}); check "
                           f"{OPENROUTER_KEY_ENV}") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"OpenRouter has no endpoint for model {self.model!r}: either the id is wrong or no "
                           "host of that model supports structured outputs (see https://openrouter.ai/models)"
                           ) from exc
        except openai.APIStatusError as exc:  # 402 no credits, 429 after SDK retries, 5xx
            raise LLMError(f"OpenRouter error {exc.status_code}: {_clip(exc.message)}") from exc
        except openai.LengthFinishReasonError as exc:
            raise LLMError(f"no {schema.__name__} in the response (finish_reason=length)") from exc
        except openai.ContentFilterFinishReasonError as exc:
            raise LLMError("the model declined the request (content filter)") from exc
        except ValidationError as exc:
            raise LLMError(f"OpenRouter call failed: the reply did not match {schema.__name__} "
                           f"({_validation_summary(exc)})") from exc
        except (openai.OpenAIError, TypeError, ValueError) as exc:  # malformed payloads, non-JSON bodies
            raise LLMError(f"OpenRouter call failed ({type(exc).__name__}): {_clip(exc)}") from exc
