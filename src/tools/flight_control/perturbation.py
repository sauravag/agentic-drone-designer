"""Perturbed plants for the SITL check: the gains are generated from the nominal model, then flown against
two off-nominal plants so a design that only works on its own model is caught.

  * HOT: strong control power, low damping, light inertia, CG aft (SM_MIN). Quickest and least damped,
    so it exposes the worst roll overshoot.
  * SLUGGISH: weak control power, high damping, heavy inertia, CG forward (SM_MAX). Slowest to respond,
    so it exposes the worst settling time.

Known gap: mixed corners (e.g. high control power with high damping) are not flown.

The CG shift keeps tail incidence fixed: Cm_alpha = -CL_alpha * SM' and Cm0 moves by CL0 * (SM0 - SM').
"""

from dataclasses import dataclass, replace
from typing import Final

from src.tools.aero_cad.stability import SM_MAX, SM_MIN
from src.tools.flight_control.flight_model import FlightModel


@dataclass(frozen=True)
class PlantCase:
    name: str
    control_power: float  # multiplies Cl_delta_a and Cm_delta_e
    damping: float  # multiplies Cl_p and Cm_q
    inertia: float  # multiplies Ixx, Iyy, Izz, Ixz
    static_margin: float  # fractions of MAC


HOT: Final[PlantCase] = PlantCase("hot", 1.3, 0.7, 0.8, SM_MIN)
SLUGGISH: Final[PlantCase] = PlantCase("sluggish", 0.7, 1.3, 1.2, SM_MAX)
PERTURBED_CASES: Final[tuple[PlantCase, ...]] = (HOT, SLUGGISH)
NOMINAL_CASE: Final[str] = "nominal"


def perturbed_model(model: FlightModel, case: PlantCase) -> FlightModel:
    """Return a copy of `model` with the case's scalings and CG shift applied; `model` is not changed."""
    a = model.aero
    aero = replace(
        a,
        cl_delta_a=a.cl_delta_a * case.control_power, cm_delta_e=a.cm_delta_e * case.control_power,
        cl_p=a.cl_p * case.damping, cm_q=a.cm_q * case.damping,
        cm_alpha=-a.cl_alpha * case.static_margin,
        cm0=a.cm0 + a.cl0 * (model.static_margin - case.static_margin),
    )
    return replace(
        model, aero=aero, static_margin=case.static_margin,
        ixx_kgm2=model.ixx_kgm2 * case.inertia, iyy_kgm2=model.iyy_kgm2 * case.inertia,
        izz_kgm2=model.izz_kgm2 * case.inertia, ixz_kgm2=model.ixz_kgm2 * case.inertia,
    )
