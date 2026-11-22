"""Perturbed plants for the SITL check: the gains are generated from the nominal model, then flown against
two off-nominal plants so a design that only works on its own model is caught.

  * HOT: strong control power, low damping, light inertia, CG aft (the band minimum). Quickest and least damped,
    so it exposes the worst roll overshoot.
  * SLUGGISH: weak control power, high damping, heavy inertia, CG forward (the band maximum). Slowest to respond,
    so it exposes the worst settling time.

Only the roll/pitch terms (Cl_delta_a, Cm_delta_e, Cl_p, Cm_q), the inertia and the static margin change. Yaw/rudder
terms and the VTOL lift rotors are not perturbed, and the lift-rotor geometry stays referenced to the nominal CG, so
the hover sees the inertia change but no CG imbalance. The transition gate uses the nominal flight.

Known gap: mixed corners (e.g. high control power with high damping) are not flown.

The CG shift keeps tail incidence fixed: Cm_alpha = -CL_alpha * SM' and Cm0 moves by CL0 * (SM0 - SM').
"""

import math
from dataclasses import dataclass, replace
from typing import Final, Literal

from src.tools.flight_control.flight_model import FlightModel
from src.tools.sizing.layouts import StaticMarginBand


@dataclass(frozen=True)
class PlantCase:
    name: str
    control_power: float  # multiplies Cl_delta_a and Cm_delta_e
    damping: float  # multiplies Cl_p and Cm_q
    inertia: float  # multiplies Ixx, Iyy, Izz, Ixz
    margin_end: Literal["minimum", "maximum"]  # which end of the layout's static-margin band the CG moves to

    def __post_init__(self) -> None:
        for field in ("control_power", "damping", "inertia"):
            if not (math.isfinite(getattr(self, field)) and getattr(self, field) > 0):
                raise ValueError(f"PlantCase {self.name!r}: {field} must be finite and > 0, got {getattr(self, field)}")
        if self.margin_end not in ("minimum", "maximum"):
            raise ValueError(f"PlantCase {self.name!r}: margin_end must be 'minimum' or 'maximum', got {self.margin_end!r}")


HOT: Final[PlantCase] = PlantCase("hot", 1.3, 0.7, 0.8, "minimum")
SLUGGISH: Final[PlantCase] = PlantCase("sluggish", 0.7, 1.3, 1.2, "maximum")
PERTURBED_CASES: Final[tuple[PlantCase, ...]] = (HOT, SLUGGISH)
NOMINAL_CASE: Final[str] = "nominal"


def perturbed_model(model: FlightModel, case: PlantCase, band: StaticMarginBand) -> FlightModel:
    """Return a copy of `model` with the case's scalings and CG shift applied; `model` is not changed."""
    a = model.aero
    margin: float = getattr(band, case.margin_end)
    aero = replace(
        a,
        cl_delta_a=a.cl_delta_a * case.control_power, cm_delta_e=a.cm_delta_e * case.control_power,
        cl_p=a.cl_p * case.damping, cm_q=a.cm_q * case.damping,
        cm_alpha=-a.cl_alpha * margin,
        cm0=a.cm0 + a.cl0 * (model.static_margin - margin),
    )
    # x_cg_m deliberately stays nominal: the JSBSim exporter puts AERORP and the CG both at x_cg_m, so the CG
    # move is carried entirely by cm_alpha/cm0. Moving x_cg_m too would double-count it.
    return replace(
        model, aero=aero, static_margin=margin,
        ixx_kgm2=model.ixx_kgm2 * case.inertia, iyy_kgm2=model.iyy_kgm2 * case.inertia,
        izz_kgm2=model.izz_kgm2 * case.inertia, ixz_kgm2=model.ixz_kgm2 * case.inertia,
    )
