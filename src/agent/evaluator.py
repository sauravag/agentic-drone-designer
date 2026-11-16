"""Closure evaluator (spec 6.2): does a candidate design close, judged by sizing + COTS catalog only.

The same gates the pipeline's sizing stage uses (`critic.evaluate_sizing_gates`) decide closure, with
both reserve fractions floored at `RESERVE_MIN` so a hybrid is sized with the reserve the gate demands.
Design errors are returned as a non-closing result, never raised. A requested reserve the floor raised is
recorded (`reserve_floor_applied`), never silently clamped (spec 7.3).

Facts handed to an LLM carry solver masses rounded to `FACT_DECIMALS`, the precision the UI shows, so the
numbers the LLM may cite are the numbers the designer sees.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Optional

from src.agent.critic import RESERVE_MIN, evaluate_sizing_gates
from src.pipeline import DESIGN_ERRORS
from src.schemas.assumptions import SERIES_HYBRID, DesignAssumptions
from src.schemas.changes import ChangeSet
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import SizingOutput
from src.tools.database.matcher import select_bom
from src.tools.sizing.mission_solver import component_needs, size_aircraft

FACT_DECIMALS: Final[int] = 2  # solver masses (kg) in LLM facts and on screen
PERCENT: Final[float] = 100.0
FLOOR_NOTE: Final[str] = "sized with the {floor} % reserve floor (requested {requested} %)"


def round_fact(value: Optional[float]) -> Optional[float]:
    """A solver number at display precision (None stays None)."""
    return None if value is None else round(value, FACT_DECIMALS)


def fact_text(value: Optional[float]) -> str:
    """A solver number exactly as it appears in the facts ('-' when missing)."""
    return "-" if value is None else str(round_fact(value))


def _percent(fraction: float) -> str:
    return f"{round(fraction * PERCENT, FACT_DECIMALS):g}"


def floor_note(requested: Optional[float]) -> str:
    """Why a result was sized with a higher reserve than requested ('' if the floor changed nothing)."""
    return "" if requested is None else FLOOR_NOTE.format(floor=_percent(RESERVE_MIN), requested=_percent(requested))


def reserve_field(propulsion: str) -> str:
    return "fuel_reserve_fraction" if propulsion == SERIES_HYBRID else "battery_reserve_fraction"


@dataclass(frozen=True)
class Evaluation:
    closes: bool
    mtow_kg: Optional[float]
    dod_group: Optional[int]
    propulsion: Optional[str]
    battery_mass_kg: Optional[float]
    fuel_mass_kg: Optional[float]
    reason: str
    reserve_floor_applied: Optional[float] = None  # the requested reserve the floor raised, if any

    @property
    def floor_note(self) -> str:
        return floor_note(self.reserve_floor_applied)

    def as_facts(self) -> dict[str, object]:
        return {
            "closes": self.closes, "mtow_kg": round_fact(self.mtow_kg), "dod_group": self.dod_group,
            "propulsion": self.propulsion, "battery_mass_kg": round_fact(self.battery_mass_kg),
            "fuel_mass_kg": round_fact(self.fuel_mass_kg), "reason": self.reason,
            "reserve_floor_applied": self.reserve_floor_applied,
        }


def with_reserve_floor(change: ChangeSet, result: Evaluation, assumptions: DesignAssumptions) -> ChangeSet:
    """Spec 6.2: carry the reserve floor `change` was checked with into it, for the powertrain it was sized with.
    `assumptions` are the design's before the change; a reserve the change itself sets wins over them."""
    if result.propulsion is None:
        return change
    field = reserve_field(result.propulsion)
    requested = getattr(change.assumptions, field)
    if (getattr(assumptions, field) if requested is None else requested) >= RESERVE_MIN:
        return change
    return change.model_copy(update={"assumptions": change.assumptions.model_copy(update={field: RESERVE_MIN})})


def _failure(reason: str) -> Evaluation:
    return Evaluation(False, None, None, None, None, None, reason)


def evaluate(req: OperationalRequirements, assumptions: DesignAssumptions, knobs: Mapping[str, float],
             db_path: Path) -> Evaluation:
    """Size and match `req`; `closes` only if every sizing gate passes. Never raises for design errors."""
    floored = assumptions.with_reserve_floor(RESERVE_MIN)
    try:
        sizing: SizingOutput = size_aircraft(req, dict(knobs), floored)
        bom: SelectedBOM = select_bom(db_path, component_needs(sizing, req), floored)
    except DESIGN_ERRORS as exc:
        return _failure(str(exc))
    failed = [gate.detail for gate in evaluate_sizing_gates(req, sizing, bom) if not gate.passed]
    requested = assumptions.reserve_for(sizing.propulsion_type)
    return Evaluation(
        closes=not failed, mtow_kg=sizing.mtow_kg, dod_group=sizing.dod_group, propulsion=sizing.propulsion_type,
        battery_mass_kg=sizing.battery_mass_kg, fuel_mass_kg=sizing.fuel_mass_kg, reason="; ".join(failed),
        reserve_floor_applied=requested if requested < RESERVE_MIN else None,
    )
