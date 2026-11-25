"""Scripted StructuredLLM for agent tests: deterministic stand-in for Claude (no network, no cost)."""

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel

from src.agent.agent_schemas import (CriticDecision, DesignerTurn, Distance, Duration, Mass, OptionsExplanation,
                                     ReportNarrative, RequirementsDraft)
from src.agent.llm import LLMError
from src.agent.nodes import AgentContext
from src.tools.database.seed_data import seed_database

SRS_PROMPT: str = ("Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, 6-hour "
                   "endurance, and 80 km datalink radius")
SRS_DRAFT = RequirementsDraft(
    mission_name="Group 2 VTOL ISR", payload_mass=Mass(value=3.5, unit="kg"), payload_sensor="eo_ir",
    endurance=Duration(value=6, unit="h"), radius=Distance(value=80, unit="km"), runway_available=False,
    vtol_required=True, dod_group=2,
)


class ScriptedLLM:
    """Answers each schema with a fixed object (or a callable of the prompt); records every call.
    `DesignerTurn` answers come from `designer_turns` in order (one per call)."""

    def __init__(self, draft: Optional[RequirementsDraft] = SRS_DRAFT,
                 critic: Callable[[str], CriticDecision] | CriticDecision | None = None,
                 fail_on: tuple[type[BaseModel], ...] = (),
                 options: Optional[OptionsExplanation] = None,
                 designer_turns: Sequence[DesignerTurn] = ()) -> None:
        self.options = options or OptionsExplanation(why_it_failed="Scripted failure reason.",
                                                     trade_offs="Scripted option trade-offs.")
        self.draft = draft
        self.critic = critic or CriticDecision(summary="defer to the rule table")
        self.fail_on = fail_on
        self.designer_turns: list[DesignerTurn] = list(designer_turns)
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
        if schema is DesignerTurn:
            assert self.designer_turns, "no scripted DesignerTurn left"
            return self.designer_turns.pop(0)
        raise AssertionError(f"unexpected schema {schema}")


def agent_context(tmp_path: Path, db: Path, llm: Any, **kw: Any) -> AgentContext:
    """An agent context for host tests: aero/CAD and SITL off, outputs under tmp_path."""
    return AgentContext(llm=llm, outputs_root=tmp_path / "outputs", db_path=db, with_aero_cad=False,
                        with_sitl=False, **kw)


def seeded_catalog(directory: Path) -> Path:
    """A freshly seeded catalogue database in `directory`."""
    path: Path = directory / "catalog.db"
    seed_database(path)
    return path
