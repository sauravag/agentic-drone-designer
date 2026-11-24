"""COTS hardware snapping: the matched propulsion set (sizing's PropulsionMatch) taken by name, the pack built from
the matched cell and series count, the remaining parts the lightest that meet a requirement with a 15% safety
margin; assemble a SelectedBOM, and check MTOW convergence.

Catalog tables are small (<100 rows), so each lookup loads the table and filters in Python.
That keeps queries free of dynamic SQL and re-validates every row read from disk.
"""

import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable, Final, Literal, Optional, TypeVar

from pydantic import BaseModel

from src.schemas.components import (
    Battery,
    Cell,
    CompanionComputer,
    ComponentNeeds,
    CruiseMotor,
    Esc,
    FlightController,
    IceGenerator,
    Payload,
    Propeller,
    SelectedBOM,
    SensorType,
    VtolMotor,
)
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.payload_catalog import payload_display_name, payload_spec
from src.schemas.requirements import ARK_JETSON_PAB, CUBE_ORANGE_PLUS
from src.tools.database.packs import build_pack, bus_window_v
from src.tools.database.seed_data import CATALOG_TABLES

SAFETY_MARGIN: float = 1.15
MTOW_TOLERANCE: float = 0.03

# Avionics names and reference prices (display only; masses and watts are DesignAssumptions).
# Sources: arkelectron.com, docs.px4.io, GetFPV/RobotShop/Lumenier, NVIDIA developer forum.
AVIONICS_RESEARCH_DATE: Final[str] = "2026-11-18"
COMPANION: Final[str] = "companion"
AVIONICS_REFERENCE: Final[dict[str, tuple[str, str]]] = {
    CUBE_ORANGE_PLUS: ("CubePilot Cube Orange+ (ADS-B carrier)", "≈ $450–546"),
    ARK_JETSON_PAB: ("ARK Jetson PAB Carrier + ARKV6X + Jetson Orin NX 16 GB",
                     "≈ $1,750–2,600 (carrier $800, ARKV6X $400, Orin NX 16 GB $599 1KU–$1,400 retail)"),
    COMPANION: ("NVIDIA Jetson Orin NX + carrier + heatsink", "≈ $1,000–1,400 (Orin NX 16 GB retail) + carrier"),
}

T = TypeVar("T", bound=BaseModel)


class ComponentNotFoundError(LookupError):
    """No catalog part satisfies the requirement."""


def _require_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def load_catalog(db_path: Path, table: str) -> list:
    """Load and validate every row of a catalog table (read-only connection)."""
    model: Optional[type[BaseModel]] = CATALOG_TABLES.get(table)
    if model is None:
        raise ValueError(f"Unknown catalog table {table!r}; expected one of {sorted(CATALOG_TABLES)}")
    if not db_path.is_file():
        raise FileNotFoundError(f"COTS catalog not found at {db_path}; run `python -m src.tools.database.seed_data`")
    # mode=ro stops sqlite from silently creating an empty database on a bad path.
    # closing(): sqlite3's own context manager only commits; it never closes the handle.
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows: list[sqlite3.Row] = conn.execute(f"SELECT * FROM {table}").fetchall()  # table is whitelisted
    return [model(**dict(row)) for row in rows]


def part_by_name(db_path: Path, table: str, name: str) -> BaseModel:
    """The catalog row called `name`; ComponentNotFoundError names the table and the part."""
    for row in load_catalog(db_path, table):
        if row.name == name:
            return row
    raise ComponentNotFoundError(f"No part in {table} named {name!r}")


def _lightest(db_path: Path, table: str, predicate: Callable[[T], bool], requirement: str) -> T:
    candidates: list[T] = [item for item in load_catalog(db_path, table) if predicate(item)]
    if not candidates:
        raise ComponentNotFoundError(f"No part in {table} satisfies {requirement}")
    return min(candidates, key=lambda item: item.mass_kg)


def match_cruise_motor(db_path: Path, required_power_w: float) -> CruiseMotor:
    need: float = _require_positive(required_power_w, "required_power_w") * SAFETY_MARGIN
    return _lightest(db_path, "cruise_motors", lambda m: m.continuous_power_w >= need,
                     f"continuous power >= {need:.0f} W (incl. margin)")


def match_vtol_motor(db_path: Path, required_thrust_n: float) -> VtolMotor:
    need: float = _require_positive(required_thrust_n, "required_thrust_n") * SAFETY_MARGIN
    return _lightest(db_path, "vtol_motors", lambda m: m.max_thrust_n >= need,
                     f"max thrust >= {need:.0f} N (incl. margin)")


def match_ice_generator(
    db_path: Path, required_power_w: float, bus_window_v: Optional[tuple[float, float]] = None
) -> IceGenerator:
    """Lightest genset with the power margin. With a bus window (end-of-reserve loaded to fully charged pack
    voltage), only gensets whose regulated output lies inside it qualify; without one, the choice is unchanged
    (base rows first, and a voltage variant weighs the same as its base, so the tie keeps the base row)."""
    need: float = _require_positive(required_power_w, "required_power_w") * SAFETY_MARGIN
    requirement: str = f"continuous power >= {need:.0f} W (incl. margin)"
    if bus_window_v is None:
        return _lightest(db_path, "ice_generators", lambda g: g.continuous_power_w >= need, requirement)
    low, high = bus_window_v
    _require_positive(low, "bus window low")
    if not math.isfinite(high) or high < low:
        raise ValueError(f"bus window must be (low, high) with low <= high, got {bus_window_v!r}")
    return _lightest(
        db_path, "ice_generators",
        lambda g: g.continuous_power_w >= need and g.output_voltage_v is not None
        and low <= g.output_voltage_v <= high,
        f"{requirement} and output voltage within {low:.1f}-{high:.1f} V",
    )


def match_battery(db_path: Path, cell_name: str, series: int, required_energy_j: float,
                  required_peak_power_w: float, assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS) -> Battery:
    """The pack of the matched cell and series count with the fewest parallel strings for the sized energy and peak
    power (spec 6.3). No extra margin: sizing already holds the reserve, and rounding P up is the only step."""
    cell: Cell = part_by_name(db_path, "cells", cell_name)  # type: ignore[assignment]
    return build_pack(cell, series, _require_positive(required_energy_j, "required_energy_j"),
                      _require_positive(required_peak_power_w, "required_peak_power_w"), assumptions)


def match_propeller(db_path: Path, motor_max_power_w: float, application: Literal["cruise", "vtol"]) -> Propeller:
    """A propeller must absorb its motor's full max power, so no extra margin is applied here."""
    need: float = _require_positive(motor_max_power_w, "motor_max_power_w")
    return _lightest(db_path, "propellers", lambda p: p.application == application and p.max_power_w >= need,
                     f"{application} prop rated >= {need:.0f} W")


def match_payload(
    db_path: Path,
    target_mass_kg: float,
    sensor_type: Optional[SensorType] = None,
    max_power_w: Optional[float] = None,
) -> Payload:
    """Pick the lightest payload at or above the requested mass (never under-deliver the sensor
    class the user asked for, and never under-estimate MTOW); ties go to the lower-power unit."""
    target: float = _require_positive(target_mass_kg, "target_mass_kg")
    candidates: list[Payload] = [
        p for p in load_catalog(db_path, "payloads")
        if p.vendor is None  # generic sensor classes only: a named gimbal is chosen by name, never by mass
        and p.mass_kg >= target
        and (sensor_type is None or p.sensor_type == sensor_type)
        and (max_power_w is None or p.power_w <= max_power_w)
    ]
    if not candidates:
        raise ComponentNotFoundError(
            f"No part in payloads with mass >= {target:.2f} kg, sensor_type={sensor_type or 'any'}, "
            f"power <= {max_power_w or 'any'} W"
        )
    return min(candidates, key=lambda p: (p.mass_kg, p.power_w))


def payload_part(db_path: Path, needs: ComponentNeeds) -> Payload:
    """The payload line: a named catalogue model carries the designer's mass and power (suffixed when they differ
    from its datasheet); Custom is matched by mass against the generic sensor classes, exactly as before."""
    spec: Optional[Payload] = payload_spec(needs.payload_model)
    if spec is None:
        return match_payload(db_path, needs.payload_mass_kg, needs.payload_sensor_type, needs.payload_power_w)
    power: float = spec.power_w if needs.payload_power_w is None else needs.payload_power_w
    return spec.model_copy(update={"mass_kg": needs.payload_mass_kg, "power_w": power,
                                   "name": payload_display_name(spec.name, needs.payload_mass_kg, power)})


def avionics_parts(
    needs: ComponentNeeds, assumptions: DesignAssumptions
) -> tuple[FlightController, Optional[CompanionComputer]]:
    """The chosen flight-controller stack and (Cube only) the separate Jetson, from the design assumptions.
    The Cube's own power sits inside the shared avionics allowance; the ARK line carries the Jetson's power."""
    if needs.flight_controller == ARK_JETSON_PAB:
        name, price = AVIONICS_REFERENCE[ARK_JETSON_PAB]
        return FlightController(name=name, mass_kg=assumptions.ark_pab_stack_mass_kg,
                                power_w=assumptions.jetson_power_w, reference_price_usd=price), None
    name, price = AVIONICS_REFERENCE[CUBE_ORANGE_PLUS]
    cube = FlightController(name=name, mass_kg=assumptions.cube_orange_plus_mass_kg, reference_price_usd=price)
    if not needs.companion_computer:
        return cube, None
    name, price = AVIONICS_REFERENCE[COMPANION]
    return cube, CompanionComputer(name=name, mass_kg=assumptions.companion_computer_mass_kg,
                                   power_w=assumptions.jetson_power_w, reference_price_usd=price)


MATCHED_NAME_FIELDS: Final[tuple[str, ...]] = (
    "cruise_motor_name", "cruise_propeller_name", "cruise_esc_name", "cell_name", "battery_series",
)
MATCHED_LIFT_FIELDS: Final[tuple[str, ...]] = ("vtol_motor_name", "vtol_propeller_name", "vtol_esc_name")


def _require_matched(needs: ComponentNeeds) -> None:
    """ValueError unless the needs carry the matched propulsion set (and its lift set when rotors are asked for)."""
    fields: tuple[str, ...] = MATCHED_NAME_FIELDS + (
        MATCHED_LIFT_FIELDS if needs.vtol_thrust_per_motor_n is not None else ())
    missing: list[str] = [name for name in fields if getattr(needs, name) is None]
    if missing:
        raise ValueError(f"component needs lack the matched propulsion set ({', '.join(missing)}); "
                         "size the design with propulsion matching first")


def _lift_parts(db_path: Path, needs: ComponentNeeds) -> dict[str, object]:
    if needs.vtol_thrust_per_motor_n is None:
        return {}
    return {"vtol_motor": part_by_name(db_path, "vtol_motors", str(needs.vtol_motor_name)),
            "vtol_propeller": part_by_name(db_path, "propellers", str(needs.vtol_propeller_name)),
            "vtol_esc": part_by_name(db_path, "escs", str(needs.vtol_esc_name)),
            "vtol_motor_count": needs.vtol_motor_count}


def _battery_and_generator(db_path: Path, needs: ComponentNeeds,
                           assumptions: DesignAssumptions) -> tuple[Battery, Optional[IceGenerator]]:
    """The cell-built pack; for a hybrid, the lightest genset whose output sits in that pack's bus window."""
    cell: Cell = part_by_name(db_path, "cells", str(needs.cell_name))  # type: ignore[assignment]
    if needs.battery_chemistry is not None and cell.chemistry != needs.battery_chemistry:
        raise ValueError(f"matched cell {cell.name} is {cell.chemistry}, the design needs {needs.battery_chemistry}")
    series: int = int(needs.battery_series or 0)
    battery: Battery = match_battery(db_path, cell.name, series, needs.battery_energy_j, needs.battery_peak_power_w,
                                     assumptions)
    if needs.generator_power_w is None:
        return battery, None
    return battery, match_ice_generator(db_path, needs.generator_power_w, bus_window_v(cell, series))


def select_bom(
    db_path: Path, needs: ComponentNeeds, assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS
) -> SelectedBOM:
    """The BOM of a matched design: the propulsion set by name, the pack from the matched cell x series, the genset
    inside the bus window (hybrids), the payload and avionics. VTOL rotors and the genset appear only when the
    sizing engine asked for them."""
    _require_matched(needs)
    battery, generator = _battery_and_generator(db_path, needs, assumptions)
    flight_controller, companion = avionics_parts(needs, assumptions)
    cruise_esc: Esc = part_by_name(db_path, "escs", str(needs.cruise_esc_name))  # type: ignore[assignment]
    return SelectedBOM(
        cruise_motor=part_by_name(db_path, "cruise_motors", str(needs.cruise_motor_name)),
        cruise_propeller=part_by_name(db_path, "propellers", str(needs.cruise_propeller_name)),
        cruise_esc=cruise_esc,
        battery=battery,
        payload=payload_part(db_path, needs),
        ice_generator=generator,
        flight_controller=flight_controller,
        companion_computer=companion,
        **_lift_parts(db_path, needs),
    )


def check_mtow_convergence(sized_mtow_kg: float, actual_mtow_kg: float, tolerance: float = MTOW_TOLERANCE) -> bool:
    """True when the discrete-hardware MTOW is within ±tolerance of the sized MTOW."""
    sized: float = _require_positive(sized_mtow_kg, "sized_mtow_kg")
    actual: float = _require_positive(actual_mtow_kg, "actual_mtow_kg")
    return abs(actual - sized) / sized <= tolerance
