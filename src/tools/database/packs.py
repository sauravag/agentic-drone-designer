"""Battery packs built from catalogue cells (spec 6.3): S cells in series set the bus voltage, P strings in
parallel are the fewest that meet the energy and the peak current at the nominal pack voltage.

Mass and box come from `pack_envelope` with the run's specific energy and energy density, so the editable
assumption still governs pack mass; the C-rate follows from the cell's continuous current.
"""

import math
from typing import Final

from src.schemas.assumptions import Chemistry, DesignAssumptions
from src.schemas.components import Battery, Cell
from src.tools.database.seed_data import pack_envelope
from src.units import WH_TO_J

# Lowest per-cell voltage under load at the end of the reserve (controller's call, spec 6.3): the climb and hover
# points are re-checked at this sagged bus voltage.
LI_ION_LOADED_MIN_V_CELL: Final[float] = 3.3
LIPO_LOADED_MIN_V_CELL: Final[float] = 3.5
LOADED_MIN_V_CELL: Final[dict[str, float]] = {"li_ion": LI_ION_LOADED_MIN_V_CELL, "lipo": LIPO_LOADED_MIN_V_CELL}
# The Battery part schema's mass bound (read from the schema, so the two never drift apart).
PACK_MASS_LIMIT_KG: Final[float] = next(m.le for m in Battery.model_fields["mass_kg"].metadata
                                        if getattr(m, "le", None) is not None)


class PackLimitError(ValueError):
    """The pack for this energy and power would break the Battery part's mass limit (the message says so)."""
# Relative slack on the parallel-count ceiling: an exact fit must not round up on float noise.
_CEIL_SLACK: Final[float] = 1e-9


def _require_series(series: int) -> int:
    if isinstance(series, bool) or not isinstance(series, int) or series < 1:
        raise ValueError(f"series cell count must be a positive integer, got {series!r}")
    return series


def _require_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def loaded_min_voltage_v(chemistry: Chemistry, series: int) -> float:
    """End-of-reserve loaded pack voltage: series x the chemistry's loaded minimum cell voltage."""
    return _require_series(series) * LOADED_MIN_V_CELL[chemistry]


def charged_voltage_v(cell: Cell, series: int) -> float:
    """Fully charged pack voltage: series x the cell's max voltage."""
    return _require_series(series) * cell.max_voltage_v


def bus_window_v(cell: Cell, series: int) -> tuple[float, float]:
    """(end-of-reserve loaded, fully charged) pack voltage: the window a hybrid's generator output must sit in."""
    return loaded_min_voltage_v(cell.chemistry, series), charged_voltage_v(cell, series)


def _strings(required: float, per_string: float) -> int:
    return max(1, math.ceil(required / per_string * (1.0 - _CEIL_SLACK)))


def build_pack(cell: Cell, series: int, energy_j: float, peak_power_w: float,
               assumptions: DesignAssumptions) -> Battery:
    """The pack of `series` cells per string with the fewest parallel strings that hold `energy_j` and supply
    `peak_power_w` at the nominal pack voltage. No safety margin is added here (the caller owns margins)."""
    _require_series(series)
    energy_wh: float = _require_positive(energy_j, "energy_j") / WH_TO_J
    peak_w: float = _require_positive(peak_power_w, "peak_power_w")
    voltage: float = series * cell.nominal_voltage_v
    parallel: int = max(_strings(energy_wh, series * cell.capacity_wh), _strings(peak_w / voltage, cell.max_current_a))
    capacity_wh: float = series * parallel * cell.capacity_wh
    chem: Chemistry = cell.chemistry
    mass, length, width, height = pack_envelope(
        capacity_wh, assumptions.specific_energy_wh_kg(chem), assumptions.energy_density_wh_l(chem)
    )
    if mass > PACK_MASS_LIMIT_KG:
        raise PackLimitError(f"battery pack over the {PACK_MASS_LIMIT_KG:g} kg part limit: {cell.name} "
                             f"{series}S{parallel}P would weigh {mass:.1f} kg for {energy_wh:.0f} Wh")
    cell_capacity_ah: float = cell.capacity_wh / cell.nominal_voltage_v
    return Battery(
        name=f"{cell.name} {series}S{parallel}P", chemistry=chem, capacity_wh=capacity_wh,
        max_c_rate=cell.max_current_a / cell_capacity_ah, voltage_v=voltage, mass_kg=mass,
        length_m=length, width_m=width, height_m=height, cell_name=cell.name, series=series, parallel=parallel,
    )
