"""Interchangeable LLMs: provider/model resolution, and the OpenRouter adapter driven through the real `openai`
SDK against a mock HTTP transport (no network, no cost)."""

import json
from typing import Any, Callable

import httpx
import openai
import pytest
from pydantic import BaseModel

from src.agent import providers
from src.agent.agent_schemas import (Adjustment, CriticDecision, DesignerTurn, FieldChange, ProposedChange,
                                     ReportNarrative)
from src.agent.llm import ClaudeLLM, LLMError
from src.agent.openrouter import OPENROUTER_BASE_URL, OpenRouterLLM
from src.agent.providers import LLMConfigError
from tests.agent_fakes import SRS_DRAFT

OR_KEY: str = "sk-or-test-key"
Handler = Callable[[httpx.Request], httpx.Response]


# ---------------------------------------------------------- provider choice ---
@pytest.mark.parametrize(("keys", "expected"), [
    ({"OPENROUTER_API_KEY": OR_KEY}, "openrouter"),
    ({"ANTHROPIC_API_KEY": "sk-ant"}, "anthropic"),
    ({"OPENROUTER_API_KEY": OR_KEY, "ANTHROPIC_API_KEY": "sk-ant"}, "openrouter"),
])
def test_the_provider_follows_whichever_key_is_set(monkeypatch: pytest.MonkeyPatch, keys: dict[str, str],
                                                   expected: str) -> None:
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    assert providers.resolve_provider().name == expected and providers.config_problem() is None


def test_without_any_key_the_problem_names_both_keys() -> None:
    problem = providers.config_problem()
    assert problem is not None and "OPENROUTER_API_KEY" in problem and "ANTHROPIC_API_KEY" in problem
    with pytest.raises(LLMConfigError):
        providers.make_llm()


def test_an_explicit_provider_needs_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv(providers.PROVIDER_ENV, "Anthropic")  # case-insensitive
    assert providers.config_problem() == "set ANTHROPIC_API_KEY for the anthropic provider"
    assert providers.config_problem("openrouter") is None  # the CLI flag beats the environment


def test_an_unknown_provider_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(providers.PROVIDER_ENV, "bedrock")
    assert "unknown LLM provider 'bedrock'" in (providers.config_problem() or "")
    with pytest.raises(LLMConfigError, match="openrouter, anthropic"):
        providers.make_llm()


def test_model_precedence_is_argument_then_environment_then_provider_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    openrouter = providers.PROVIDERS["openrouter"]
    assert providers.resolve_model(openrouter) == "anthropic/claude-opus-5.5"
    assert providers.resolve_model(providers.PROVIDERS["anthropic"]) == "claude-opus-5-5"
    monkeypatch.setenv("ISR_FORGE_MODEL", "google/gemini-3.8-flash")
    assert providers.resolve_model(openrouter) == "google/gemini-3.8-flash"
    assert providers.resolve_model(openrouter, "  openai/gpt-5.6-sol  ") == "openai/gpt-5.6-sol"
    assert providers.resolve_model(openrouter, "   ") == "google/gemini-3.8-flash"  # blank = no override


@pytest.mark.parametrize("model", ["gpt; rm -rf /", "a b", "-leading-dash", "x" * 201, "mo\ndel", "m/é"])
def test_model_ids_are_validated_at_the_boundary(model: str) -> None:
    with pytest.raises(LLMConfigError, match="invalid model id"):
        providers.resolve_model(providers.PROVIDERS["openrouter"], model)


@pytest.mark.parametrize("model", ["openai/gpt-5.6-sol", "anthropic/claude-opus-5.5:batch", "meta-llama/llama-4@x",
                                   "claude-opus-5-5"])
def test_real_model_ids_are_accepted(model: str) -> None:
    assert providers.resolve_model(providers.PROVIDERS["openrouter"], model) == model


def test_the_factory_builds_the_chosen_adapter_with_the_chosen_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    routed = providers.make_llm(model="openai/gpt-5.6-sol")
    assert isinstance(routed, OpenRouterLLM) and routed.model == "openai/gpt-5.6-sol"
    assert str(routed._client.base_url).startswith(OPENROUTER_BASE_URL) and routed._client.api_key == OR_KEY
    claude = providers.make_llm("anthropic", "claude-sonnet-5-5")
    assert isinstance(claude, ClaudeLLM) and claude.model == "claude-sonnet-5-5"
    assert providers.describe("anthropic") == "anthropic · claude-opus-5-5"


# ------------------------------------------------------- OpenRouter adapter ---
def _completion(content: str | None, finish: str = "stop", refusal: str | None = None) -> dict[str, Any]:
    message = {"role": "assistant", "content": content, "refusal": refusal}
    return {"id": "gen-123", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}]}


def _llm(handler: Handler, model: str = "openai/gpt-5.6-sol") -> OpenRouterLLM:
    client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OR_KEY, max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return OpenRouterLLM(model, client)


def _reply(payload: dict[str, Any], status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, json=payload)


def test_openrouter_sends_a_strict_schema_with_parameter_routing_and_parses_the_reply() -> None:
    seen: list[httpx.Request] = []
    decision = CriticDecision(adjustments=[Adjustment(knob="aspect_ratio", direction="increase", reason="L/D")],
                              summary="raise AR")

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(decision.model_dump_json()))

    assert _llm(handler).parse("system text", "prompt text", CriticDecision, effort="low") == decision
    (request,) = seen
    body = json.loads(request.content)
    assert request.url.host == "openrouter.ai" and request.url.path == "/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {OR_KEY}"
    assert body["model"] == "openai/gpt-5.6-sol" and body["provider"] == {"require_parameters": True}
    assert body["messages"] == [{"role": "system", "content": "system text"},
                                {"role": "user", "content": "prompt text"}]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "CriticDecision"


@pytest.mark.parametrize("expected", [
    SRS_DRAFT, CriticDecision(summary="nothing to change"),
    ReportNarrative(executive_summary="e", trade_offs="t", risks_and_next_steps="r"),
])
def test_every_agent_schema_round_trips_through_the_openai_sdk(expected: BaseModel) -> None:
    llm = _llm(_reply(_completion(expected.model_dump_json())))
    assert llm.parse("s", "p", type(expected)) == expected


@pytest.mark.parametrize(("handler", "match"), [
    (_reply(_completion(None, refusal="I can't help with that")), "declined"),
    (_reply(_completion('{"summary": "cut', finish="length")), "finish_reason=length"),
    (_reply(_completion('{"adjustments": [{"knob": "wingspan", "direction": "up", "reason": "x"}], '
                        '"summary": "s"}')), "OpenRouter call failed"),
    (_reply(_completion(None)), "no CriticDecision"),
    (_reply({**_completion("{}"), "choices": []}), "no choices"),
    (_reply({"error": {"message": "No endpoints found that support the provided parameters"}}, 404),
     "no endpoint for model 'openai/gpt-5.6-sol'"),
    (_reply({"error": {"message": "Insufficient credits"}}, 402), "OpenRouter error 402"),
    (_reply({"id": "gen-9", "error": {"code": 502, "message": "Upstream provider overloaded"}}),
     "upstream error 502: Upstream provider overloaded"),  # HTTP 200 carrying an error and no choices
    (_reply({"error": {"message": "x" * 5000}}, 500), r"^OpenRouter error 500: .{1,200}…$"),
])
def test_openrouter_failures_become_llm_errors(handler: Handler, match: str) -> None:
    with pytest.raises(LLMError, match=match):
        _llm(handler).parse("s", "p", CriticDecision)


def test_schema_mismatches_name_the_fields_but_not_the_model_output() -> None:
    reply = '{"adjustments": [{"knob": "SECRET-VALUE", "direction": "up", "reason": "x"}], "summary": "s"}'
    with pytest.raises(LLMError, match=r"did not match CriticDecision \(2 schema error\(s\) at adjustments\.0\.knob"
                       ) as info:
        _llm(_reply(_completion(reply))).parse("s", "p", CriticDecision)
    assert "SECRET-VALUE" not in str(info.value)


def test_the_key_is_stripped_before_it_reaches_the_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", f"  {OR_KEY}\n")
    assert OpenRouterLLM("openai/gpt-5.6-sol")._client.api_key == OR_KEY


def test_openrouter_connection_failures_become_llm_errors() -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(LLMError, match="cannot reach OpenRouter"):
        _llm(offline).parse("s", "p", CriticDecision)


def test_rejected_openrouter_credentials_give_a_fixed_message_without_the_body() -> None:
    handler = _reply({"error": {"message": f"No auth credentials found for {OR_KEY}-SECRET"}}, 401)
    with pytest.raises(LLMError, match="rejected the credentials") as info:
        _llm(handler).parse("s", "p", CriticDecision)
    assert "SECRET" not in str(info.value) and "OPENROUTER_API_KEY" in str(info.value)


def test_designer_turn_round_trips_through_the_openai_sdk() -> None:
    turn = DesignerTurn(message="m", what_ifs=[], proposal=None, done=True)
    assert _llm(_reply(_completion(turn.model_dump_json()))).parse("s", "p", DesignerTurn) == turn


def test_a_designer_turn_with_changes_round_trips_under_a_strict_schema() -> None:
    seen: list[httpx.Request] = []
    change = ProposedChange(label="AR 14", changes=[FieldChange(field="endurance_hours", value="4"),
                                                    FieldChange(field="aspect_ratio", value="14")])
    turn = DesignerTurn(message="m", what_ifs=[change, change], proposal=change, done=False)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(turn.model_dump_json()))

    assert _llm(handler).parse("s", "p", DesignerTurn) == turn
    schema = json.loads(seen[0].content)["response_format"]["json_schema"]
    assert schema["strict"] is True and set(schema["schema"]["required"]) == {"message", "what_ifs", "proposal", "done"}


def _nullable_or_union(node: Any) -> int:
    """How many schema nodes in `node` are unions (anyOf/oneOf or a list of types)."""
    if isinstance(node, list):
        return sum(_nullable_or_union(item) for item in node)
    if not isinstance(node, dict):
        return 0
    own = int("anyOf" in node or "oneOf" in node or isinstance(node.get("type"), list))
    return own + sum(_nullable_or_union(value) for value in node.values())


def test_the_designer_turn_schema_claude_receives_has_at_most_two_unions() -> None:
    from anthropic.lib._parse._transform import to_json_output_format  # what beta.messages.parse sends

    schema = to_json_output_format(DesignerTurn)["schema"]
    assert _nullable_or_union(schema) <= 2
    assert "ChangeSet" not in json.dumps(schema)  # the nested all-optional ChangeSet is never sent to an LLM
