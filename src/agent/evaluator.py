"""Closure evaluator (spec 6.2): does a candidate design close, judged by sizing + COTS catalog only.

The same gates the pipeline's sizing stage uses (`critic.evaluate_sizing_gates`) decide closure, with
both reserve fractions floored at `RESERVE_MIN` so a hybrid is sized with the reserve the gate demands.
Design errors are returned as a non-closing result, never raised.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from src.agent.critic import RESERVE_MIN, evaluate_sizing_gates
from src.pipeline import DESIGN_ERRORS
from src.schemas.assumptions import DesignAssumptions
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import SizingOutput
from src.tools.database.matcher import select_bom
from src.tools.sizing.mission_solver import component_needs, size_aircraft


@dataclass(frozen=True)
class Evaluation:
    closes: bool
    mtow_kg: Optional[float]
    dod_group: Optional[int]
    propulsion: Optional[str]
    battery_mass_kg: Optional[float]
    fuel_mass_kg: Optional[float]
    reason: str

    def as_facts(self) -> dict[str, object]:
        return {
            "closes": self.closes, "mtow_kg": self.mtow_kg, "dod_group": self.dod_group,
            "propulsion": self.propulsion, "battery_mass_kg": self.battery_mass_kg,
            "fuel_mass_kg": self.fuel_mass_kg, "reason": self.reason,
        }


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
    return Evaluation(
        closes=not failed, mtow_kg=sizing.mtow_kg, dod_group=sizing.dod_group, propulsion=sizing.propulsion_type,
        battery_mass_kg=sizing.battery_mass_kg, fuel_mass_kg=sizing.fuel_mass_kg, reason="; ".join(failed),
    )
