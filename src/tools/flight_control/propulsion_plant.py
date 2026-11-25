"""The simulator's propulsors (spec 6.6): one `Propulsor` per propeller role, built from the BOM parts.

Matched records (motor constants, an ESC, a cell-built pack) go straight through `propulsor_from_parts`, the same
model the propulsion matching used. Records written before propulsion matching (commit 9d63da9) carry neither motor
constants nor ESCs nor a cell build, and still have to fly. Their LEGACY PATH, part by part:

* motor constants (rm, I0, max current): the catalogue's own estimate rule (`motor_constants`) applied to the
  record's ratings. In those records `voltage_v` is the winding's NOMINAL voltage (the seed column before 9d63da9),
  which is what the rule takes. A lift motor's KV comes from its "-<KV>KV" name suffix, its continuous power is the
  catalogue's VTOL_CONTINUOUS_FRACTION of its peak;
* ESC efficiency: the catalogue's generic ESC class for the pack voltage (`legacy_esc_efficiency`);
* bus: the pack's recorded voltage; series count (battery failsafes): the voltage over the chemistry's nominal cell
  voltage (`pack_series`).

No fixed propulsive efficiency is used on either path: thrust always comes from the propeller tables.
Simplification (ruled in scope): past zero-thrust advance ratio the propulsor gives 0 N, not the negative thrust
of a windmilling propeller, so the plant has no windmilling drag at low throttle and high airspeed.
"""

from typing import Final, Optional

from src.schemas.components import Battery, CruiseMotor, Esc, Propeller, VtolMotor
from src.tools.database.catalog_variants import (ESC_EFFICIENCY_HV, ESC_EFFICIENCY_SMALL, ESC_HV_THRESHOLD_V,
                                                 MAX_CELL_V, VTOL_CONTINUOUS_FRACTION, kv_from_name, motor_constants)
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion.prop_tables import propeller_table
from src.tools.propulsion.propulsor import MOTOR_PARAMS, Propulsor, propulsor_from_parts

# Nominal cell voltage per chemistry (the catalogue cells: Li-ion 3.6 V, LiPo 3.7 V); legacy packs only.
LEGACY_NOMINAL_V_CELL: Final[dict[str, float]] = {"li_ion": 3.6, "lipo": 3.7}


def legacy_esc_efficiency(bus_voltage_v: float) -> float:
    """The catalogue's generic ESC class efficiency for a pack voltage (a record without an ESC)."""
    return ESC_EFFICIENCY_HV if bus_voltage_v > ESC_HV_THRESHOLD_V else ESC_EFFICIENCY_SMALL


def pack_series(battery: Battery) -> int:
    """Cells in series: the cell build's count, or (legacy pack) the voltage over the nominal cell voltage."""
    if battery.series is not None:
        return battery.series
    nominal_v: float | None = LEGACY_NOMINAL_V_CELL.get(battery.chemistry)
    if nominal_v is None:
        raise ValueError(f"pack {battery.name!r} has no cell build and no nominal cell voltage is known for chemistry "
                         f"{battery.chemistry!r} (known: {', '.join(sorted(LEGACY_NOMINAL_V_CELL))})")
    return max(1, round(battery.voltage_v / nominal_v))


def charged_cell_voltage_v(battery: Battery) -> float:
    """Fully charged cell voltage: the pack's catalogue cell, or (legacy pack / cell not in the catalogue) the
    catalogue's MAX_CELL_V."""
    cell = next((c for c in catalog_rows("cells") if c.name == battery.cell_name), None)
    return MAX_CELL_V if cell is None else cell.max_voltage_v


def is_legacy_motor(motor: CruiseMotor | VtolMotor) -> bool:
    return any(getattr(motor, name) is None for name in ("kv", *MOTOR_PARAMS))


def _legacy_constants(motor: CruiseMotor | VtolMotor) -> dict[str, float]:
    """kv, rm, I0 and max current of a pre-matching motor record (estimates; see the module docstring)."""
    if isinstance(motor, VtolMotor):
        continuous_w: float = VTOL_CONTINUOUS_FRACTION * motor.max_power_w
        kv: float = motor.kv if motor.kv is not None else kv_from_name(motor.name)
    else:
        continuous_w, kv = motor.continuous_power_w, motor.kv
    return {"kv": kv, **motor_constants(continuous_w, motor.max_power_w, motor.voltage_v)}


def plant_propulsor(motor: CruiseMotor | VtolMotor, propeller: Propeller, esc: Optional[Esc], battery: Battery,
                    installation_factor: float = 1.0) -> Propulsor:
    """The propulsor the simulator flies: the matched parts as-is, or the legacy estimates for what a pre-matching
    record lacks."""
    if esc is not None and not is_legacy_motor(motor):
        return propulsor_from_parts(motor, propeller, esc, battery, installation_factor)
    constants: dict[str, float] = (_legacy_constants(motor) if is_legacy_motor(motor) else
                                   {name: getattr(motor, name) for name in ("kv", *MOTOR_PARAMS)})
    efficiency: float = esc.efficiency if esc is not None else legacy_esc_efficiency(battery.voltage_v)
    return Propulsor(
        table=propeller_table(propeller.diameter_m, propeller.pitch_m), esc_efficiency=efficiency,
        bus_voltage_v=battery.voltage_v, installation_factor=installation_factor,
        role="lift" if propeller.application == "vtol" else "cruise", **constants,
    )
