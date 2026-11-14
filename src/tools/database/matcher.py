"""COTS hardware snapping: pick the lightest catalog part that meets a requirement with a 15%
safety margin, assemble a SelectedBOM, and check MTOW convergence.

Catalog tables are small (<100 rows), so each lookup loads the table and filters in Python.
That keeps queries free of dynamic SQL and re-validates every row read from disk.
"""

import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

from pydantic import BaseModel

from src.schemas.components import (
    Battery,
    ComponentNeeds,
    CruiseMotor,
    IceGenerator,
    Payload,
    Propeller,
    SelectedBOM,
    SensorType,
    VtolMotor,
)
from src.tools.database.seed_data import CATALOG_TABLES
from src.units import WH_TO_J

SAFETY_MARGIN: float = 1.15
MTOW_TOLERANCE: float = 0.03

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


def match_ice_generator(db_path: Path, required_power_w: float) -> IceGenerator:
    need: float = _require_positive(required_power_w, "required_power_w") * SAFETY_MARGIN
    return _lightest(db_path, "ice_generators", lambda g: g.continuous_power_w >= need,
                     f"continuous power >= {need:.0f} W (incl. margin)")


def match_battery(
    db_path: Path,
    required_energy_j: float,
    required_peak_power_w: float,
    chemistry: Optional[Literal["li_ion", "lipo"]] = None,
) -> Battery:
    energy: float = _require_positive(required_energy_j, "required_energy_j") * SAFETY_MARGIN
    peak: float = _require_positive(required_peak_power_w, "required_peak_power_w") * SAFETY_MARGIN

    def fits(b: Battery) -> bool:
        return b.capacity_j >= energy and b.max_power_w >= peak and (chemistry is None or b.chemistry == chemistry)

    return _lightest(db_path, "batteries", fits,
                     f"energy >= {energy / WH_TO_J:.0f} Wh, peak >= {peak:.0f} W, chemistry={chemistry or 'any'}")


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
        if p.mass_kg >= target
        and (sensor_type is None or p.sensor_type == sensor_type)
        and (max_power_w is None or p.power_w <= max_power_w)
    ]
    if not candidates:
        raise ComponentNotFoundError(
            f"No part in payloads with mass >= {target:.2f} kg, sensor_type={sensor_type or 'any'}, "
            f"power <= {max_power_w or 'any'} W"
        )
    return min(candidates, key=lambda p: (p.mass_kg, p.power_w))


def select_bom(db_path: Path, needs: ComponentNeeds) -> SelectedBOM:
    """Snap every SI requirement to a discrete COTS part. VTOL rotors and the ICE generator are
    selected only when the sizing engine asked for them."""
    cruise_motor: CruiseMotor = match_cruise_motor(db_path, needs.cruise_power_w)
    vtol_motor: Optional[VtolMotor] = None
    vtol_prop: Optional[Propeller] = None
    if needs.vtol_thrust_per_motor_n is not None:
        vtol_motor = match_vtol_motor(db_path, needs.vtol_thrust_per_motor_n)
        vtol_prop = match_propeller(db_path, vtol_motor.max_power_w, "vtol")
    generator: Optional[IceGenerator] = (
        match_ice_generator(db_path, needs.generator_power_w) if needs.generator_power_w is not None else None
    )
    return SelectedBOM(
        cruise_motor=cruise_motor,
        cruise_propeller=match_propeller(db_path, cruise_motor.max_power_w, "cruise"),
        battery=match_battery(db_path, needs.battery_energy_j, needs.battery_peak_power_w, needs.battery_chemistry),
        payload=match_payload(db_path, needs.payload_mass_kg, needs.payload_sensor_type, needs.payload_power_w),
        vtol_motor=vtol_motor,
        vtol_propeller=vtol_prop,
        vtol_motor_count=needs.vtol_motor_count if vtol_motor is not None else 0,
        ice_generator=generator,
    )


def check_mtow_convergence(sized_mtow_kg: float, actual_mtow_kg: float, tolerance: float = MTOW_TOLERANCE) -> bool:
    """True when the discrete-hardware MTOW is within ±tolerance of the sized MTOW."""
    sized: float = _require_positive(sized_mtow_kg, "sized_mtow_kg")
    actual: float = _require_positive(actual_mtow_kg, "actual_mtow_kg")
    return abs(actual - sized) / sized <= tolerance
