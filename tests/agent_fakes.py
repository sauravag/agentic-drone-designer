"""Scripted StructuredLLM for agent tests: deterministic stand-in for Claude (no network, no cost)."""

from typing import Any, Callable, Optional

from pydantic import BaseModel

from src.agent.agent_schemas import (CriticDecision, Distance, Duration, Mass, OptionsExplanation, ReportNarrative,
                                     RequirementsDraft)
from src.agent.llm import LLMError

SRS_PROMPT: str = ("Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, 6-hour "
                   "endurance, and 80 km datalink radius")
SRS_DRAFT = RequirementsDraft(
    mission_name="Group 2 VTOL ISR", payload_mass=Mass(value=3.5, unit="kg"), payload_sensor="eo_ir",
    endurance=Duration(value=6, unit="h"), radius=Distance(value=80, unit="km"), runway_available=False,
    vtol_required=True, dod_group=2,
)


class ScriptedLLM:
    """Answers each schema with a fixed object (or a callable of the prompt); records every call."""

    def __init__(self, draft: Optional[RequirementsDraft] = SRS_DRAFT,
                 critic: Callable[[str], CriticDecision] | CriticDecision | None = None,
                 fail_on: tuple[type[BaseModel], ...] = (),
                 options: Optional[OptionsExplanation] = None) -> None:
        self.options = options or OptionsExplanation(why_it_failed="Scripted failure reason.",
                                                     trade_offs="Scripted option trade-offs.")
        self.draft = draft
        self.critic = critic or CriticDecision(summary="defer to the rule table")
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def parse(self, system: str, prompt: str, schema: type[Any], effort: str = "medium") -> Any:
        self.calls.append((schema.__name__, prompt))
        if schema in self.fail_on:
            raise LLMError(f"scripted failure for {schema.__name__}")
        if schema is RequirementsDraft:
            return self.draft
        if schema is CriticDecision:
            return self.critic(prompt) if callable(self.critic) else self.critic
        if schema is ReportNarrative:
            return ReportNarrative(executive_summary="Scripted summary.", trade_offs="Scripted trade-offs.",
                                   risks_and_next_steps="Scripted next steps.")
        if schema is OptionsExplanation:
            return self.options
        raise AssertionError(f"unexpected schema {schema}")
