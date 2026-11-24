"""Rows generated from the base motor / propeller / generator tables: KV windings, pitch and folding
propeller variants, generic ESC classes and regulated-output generator voltage variants.

ALL constants below are catalogue ESTIMATES, not datasheet values (see seed_data's header note).
Ordering contract: every base row stays first, in its original order, with generated rows appended
after, so the lightest-part matchers (ties -> first row) keep returning exactly the base parts.
"""

import math
import re
from typing import Final

from src.schemas.components import CruiseMotor, Esc, IceGenerator, Propeller, VtolMotor
from src.units import INCH_TO_M

NOMINAL_CELL_V: Final[float] = 3.7
MAX_CELL_V: Final[float] = 4.2

# Winding factors f: KV x f at rated voltage / f (same power, same no-load rpm at rated voltage).
WINDING_FACTORS: Final[tuple[float, ...]] = (0.75, 1.33)
# Motor constants (ESTIMATE rules): copper loss 7 % of continuous power at continuous current,
# iron/friction loss 3 % of continuous power at nominal voltage.
COPPER_LOSS_FRACTION: Final[float] = 0.07
IRON_LOSS_FRACTION: Final[float] = 0.03
# Lift motors list only a peak power; assume their continuous rating is 70 % of it (ESTIMATE).
VTOL_CONTINUOUS_FRACTION: Final[float] = 0.7
_KV_SUFFIX: Final[re.Pattern[str]] = re.compile(r"-(\d+)KV$")
_CONSTANT_DIGITS: Final[int] = 6

PITCH_RATIOS: Final[dict[str, tuple[float, ...]]] = {
    "cruise": (0.5, 0.6, 0.7, 0.8, 1.0),
    "vtol": (0.3, 0.4, 0.5, 0.6),
}
PITCH_DUPLICATE_TOLERANCE: Final[float] = 0.05  # P/D closer than this to the base pitch is a duplicate
FOLDING_MASS_FACTOR: Final[float] = 1.3  # hub and hinge mass (ESTIMATE)
FOLDING_SUFFIX: Final[str] = " folding"

# ESC classes (ESTIMATES): small packs by cell count, high-voltage classes above 60 V.
ESC_SMALL_CELLS: Final[tuple[int, ...]] = (2, 3, 4, 6, 12, 14)
ESC_HV_CLASSES_V: Final[tuple[float, ...]] = (125.0, 250.0, 500.0, 750.0)
ESC_CURRENT_LADDER_A: Final[tuple[float, ...]] = (10, 20, 30, 40, 60, 80, 120, 160, 200, 300, 400)
ESC_CURRENT_MARGIN: Final[float] = 1.2  # continuous rating >= 1.2 x the motor's max current
ESC_BURST_FACTOR: Final[float] = 1.5
ESC_EFFICIENCY_SMALL: Final[float] = 0.95
ESC_EFFICIENCY_HV: Final[float] = 0.97
ESC_HV_THRESHOLD_V: Final[float] = 60.0
_VOLTAGE_EPS_V: Final[float] = 1e-6  # n*4.2 and round(n*4.2, 1) differ by float noise
ESC_BASE_MASS_KG: Final[float] = 0.01
ESC_MASS_KG_PER_KW: Final[float] = 0.025  # of continuous power (current x max voltage)
ESC_DENSITY_KG_M3: Final[float] = 1200.0
ESC_ASPECT: Final[tuple[float, float, float]] = (3.0, 2.0, 1.0)  # L : W : H

# Generator voltage variants (ESTIMATES): a regulated DC output at a pack's nominal voltage (cells x 3.7 V), so it
# sits inside the bus window of that series count for both chemistries. Same mass as the base genset (the
# rectifier/regulator is part of the genset). HV classes (>= 24S) only on the big gensets; any class whose output
# current would exceed GENSET_MAX_OUTPUT_CURRENT_A is not offered.
GENSET_SERIES_CLASSES: Final[tuple[int, ...]] = (6, 8, 10, 12, 14, 24, 48, 96)
GENSET_HV_MIN_SERIES: Final[int] = 24
GENSET_HV_MIN_POWER_W: Final[float] = 5_000.0
GENSET_MAX_OUTPUT_CURRENT_A: Final[float] = 250.0


def rated_max_voltage(nominal_v: float) -> float:
    """Max rated voltage of a winding: rated cell count (nominal / 3.7 V) x 4.2 V."""
    return max(1, round(nominal_v / NOMINAL_CELL_V)) * MAX_CELL_V


def _constants(continuous_w: float, max_w: float, nominal_v: float) -> dict[str, float]:
    i_cont: float = continuous_w / nominal_v
    return {
        "rm_ohm": round(COPPER_LOSS_FRACTION * continuous_w / i_cont**2, _CONSTANT_DIGITS),
        "i0_a": round(IRON_LOSS_FRACTION * continuous_w / nominal_v, _CONSTANT_DIGITS),
        "max_current_a": round(max_w / nominal_v, _CONSTANT_DIGITS),
    }


def _winding_name(name: str, kv: int) -> str:
    return _KV_SUFFIX.sub(f"-{kv}KV", name) if _KV_SUFFIX.search(name) else f"{name}-{kv}KV"


def _kv_from_name(name: str) -> float:
    match = _KV_SUFFIX.search(name)
    if match is None:
        raise ValueError(f"cannot derive KV from motor name {name!r}")
    return float(match.group(1))


def _windings(name: str, kv: float, nominal_v: float) -> list[tuple[str, float, float]]:
    """(name, kv, nominal_v) for the base winding first, then each variant (duplicates dropped)."""
    out: list[tuple[str, float, float]] = [(name, kv, nominal_v)]
    seen: set[int] = {round(kv)}
    for factor in WINDING_FACTORS:
        new_kv: int = round(kv * factor)
        if new_kv in seen:
            continue
        seen.add(new_kv)
        out.append((_winding_name(name, new_kv), float(new_kv), nominal_v * kv / new_kv))
    return out


def cruise_motor_rows(raw: list[tuple]) -> list[CruiseMotor]:
    """(name, mass_kg, max_power_w, continuous_power_w, kv, nominal_v, diameter_m, length_m) -> base + windings."""
    bases: list[CruiseMotor] = []
    variants: list[CruiseMotor] = []
    for name, mass, max_w, cont_w, kv, volts, dia, length in raw:
        for i, (w_name, w_kv, w_v) in enumerate(_windings(name, kv, volts)):
            row = CruiseMotor(
                name=w_name, mass_kg=mass, max_power_w=max_w, continuous_power_w=cont_w, kv=w_kv,
                voltage_v=rated_max_voltage(w_v), diameter_m=dia, length_m=length,
                **_constants(cont_w, max_w, w_v),
            )
            (bases if i == 0 else variants).append(row)
    return bases + variants


def vtol_motor_rows(raw: list[tuple]) -> list[VtolMotor]:
    """(name, mass_kg, max_thrust_n, max_power_w, nominal_v, diameter_m, length_m) -> base + windings."""
    bases: list[VtolMotor] = []
    variants: list[VtolMotor] = []
    for name, mass, thrust, max_w, volts, dia, length in raw:
        for i, (w_name, w_kv, w_v) in enumerate(_windings(name, _kv_from_name(name), volts)):
            row = VtolMotor(
                name=w_name, mass_kg=mass, max_thrust_n=thrust, max_power_w=max_w,
                voltage_v=rated_max_voltage(w_v), diameter_m=dia, length_m=length, kv=w_kv,
                **_constants(max_w * VTOL_CONTINUOUS_FRACTION, max_w, w_v),
            )
            (bases if i == 0 else variants).append(row)
    return bases + variants


def _prop_name(application: str, dia_in: float, pitch_in: float) -> str:
    return f"Carbon {application} prop {dia_in:g}x{pitch_in:g}in"


def propeller_rows(tables: dict[str, list[tuple]]) -> list[Propeller]:
    """Base props (original order), then pitch variants, then folding cruise variants.

    `tables` maps application -> (diameter_in, pitch_in, mass_kg, max_power_w) rows.
    """
    bases: list[Propeller] = []
    pitch_variants: list[Propeller] = []
    for application, table in tables.items():
        for dia_in, pitch_in, mass, power in table:
            def make(pitch: float) -> Propeller:
                return Propeller(
                    name=_prop_name(application, dia_in, pitch), diameter_m=round(dia_in * INCH_TO_M, 4),
                    pitch_m=round(pitch * INCH_TO_M, 4), mass_kg=mass, max_power_w=power, application=application,
                )

            bases.append(make(pitch_in))
            for ratio in PITCH_RATIOS[application]:
                if abs(ratio - pitch_in / dia_in) >= PITCH_DUPLICATE_TOLERANCE:
                    pitch_variants.append(make(round(ratio * dia_in, 1)))
    folding: list[Propeller] = [
        p.model_copy(update={"name": p.name + FOLDING_SUFFIX, "mass_kg": p.mass_kg * FOLDING_MASS_FACTOR,
                             "is_folding": True})
        for p in [*bases, *pitch_variants] if p.application == "cruise"
    ]
    return bases + pitch_variants + folding


def _esc_row(current_a: float, max_voltage_v: float, label: str) -> Esc:
    # ESTIMATE: mass scales with continuous power; a 3:2:1 box at ESC_DENSITY_KG_M3.
    mass: float = ESC_BASE_MASS_KG + ESC_MASS_KG_PER_KW * current_a * max_voltage_v / 1000.0
    unit: float = (mass / ESC_DENSITY_KG_M3 / math.prod(ESC_ASPECT)) ** (1.0 / 3.0)
    return Esc(
        name=f"ESC {current_a:g}A {label}", mass_kg=round(mass, 4), continuous_current_a=current_a,
        burst_current_a=ESC_BURST_FACTOR * current_a, max_voltage_v=max_voltage_v,
        efficiency=ESC_EFFICIENCY_SMALL if max_voltage_v <= ESC_HV_THRESHOLD_V else ESC_EFFICIENCY_HV,
        length_m=round(ESC_ASPECT[0] * unit, 4), width_m=round(ESC_ASPECT[1] * unit, 4),
        height_m=round(ESC_ASPECT[2] * unit, 4),
    )


def esc_rows(motors: list[CruiseMotor | VtolMotor]) -> list[Esc]:
    """Generic ESC classes: for each voltage class, every ladder current up to the first rung that
    covers 1.2 x the largest motor current wound for that class."""
    classes: list[tuple[float, str]] = [(round(c * MAX_CELL_V, 1), f"{c}S") for c in ESC_SMALL_CELLS]
    classes += [(v, f"{v:g}V") for v in ESC_HV_CLASSES_V]
    rows: list[Esc] = []
    previous_v: float = 0.0
    for class_v, label in classes:
        in_class: list[CruiseMotor | VtolMotor] = [
            m for m in motors if previous_v + _VOLTAGE_EPS_V < m.voltage_v <= class_v + _VOLTAGE_EPS_V
        ]
        need: float = max((ESC_CURRENT_MARGIN * (m.max_current_a or 0.0) for m in in_class), default=0.0)
        for rung in ESC_CURRENT_LADDER_A:
            rows.append(_esc_row(rung, class_v, label))
            if rung >= need:
                break
        previous_v = class_v
    return rows


def _genset_offers(genset: IceGenerator, series: int) -> bool:
    output_v: float = series * NOMINAL_CELL_V
    if series >= GENSET_HV_MIN_SERIES and genset.continuous_power_w < GENSET_HV_MIN_POWER_W:
        return False
    return genset.continuous_power_w / output_v <= GENSET_MAX_OUTPUT_CURRENT_A


def generator_rows(bases: list[IceGenerator]) -> list[IceGenerator]:
    """Base gensets (unregulated, output_voltage_v None, original order), then each one's regulated voltage
    variants named "<base> <S>S". Same mass as the base, so the lightest-part matcher's tie keeps the base row."""
    variants: list[IceGenerator] = [
        g.model_copy(update={"name": f"{g.name} {s}S", "output_voltage_v": round(s * NOMINAL_CELL_V, 1)})
        for g in bases for s in GENSET_SERIES_CLASSES if _genset_offers(g, s)
    ]
    return [*bases, *variants]
