"""Generates data/cots_catalog.db: a representative COTS catalog spanning Group 1 (~2 kg) to
Group 3 (~500 kg) UAVs.

NOTE: names are generic product classes and figures are representative of typical vendor
datasheets, not quotes for a specific SKU. Verify against real datasheets before procurement.
Run directly: `python -m src.tools.database.seed_data`
"""

import functools
import math
import sqlite3
import types
from contextlib import closing
import typing
from pathlib import Path
from typing import Final, Optional

from pydantic import BaseModel

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.components import Cell, CruiseMotor, Esc, IceGenerator, Payload, Propeller, VtolMotor
from src.schemas.payload_catalog import PAYLOADS
from src.tools.database.catalog_variants import (cruise_motor_rows, esc_rows, generator_rows, propeller_rows,
                                                 vtol_motor_rows)

DEFAULT_DB_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "cots_catalog.db"
# Bump whenever a table's columns or seeded rows change: ensure_catalog reseeds a file with another version.
# 2: researched NextVision/SIYI gimbals and the payload vendor / reference-price columns.
# 3: motor windings + equivalent-circuit constants, propeller pitch/folding variants, ESC table.
# 4: cells table, battery cell columns, generator output voltage + regulated voltage variants.
# 5: the fixed `batteries` pack table is gone: every pack is built from a cell (`packs.build_pack`).
# 6: motor frames' KV, max current and lift thrust derived from the propulsor model (motor_ratings).
CATALOG_VERSION: Final[int] = 6

CATALOG_TABLES: Final[dict[str, type[BaseModel]]] = {
    "cruise_motors": CruiseMotor,
    "vtol_motors": VtolMotor,
    "ice_generators": IceGenerator,
    "cells": Cell,
    "propellers": Propeller,
    "escs": Esc,
    "payloads": Payload,
}

MASS_DECIMALS_SCALE: Final[float] = 1_000.0  # pack masses are reported to the gram
# Pack-level properties used to derive battery mass and volume from capacity.
PACK_SPECIFIC_ENERGY_WH_KG: Final[dict[str, float]] = {
    chem: DEFAULT_ASSUMPTIONS.specific_energy_wh_kg(chem) for chem in ("li_ion", "lipo")
}
PACK_ENERGY_DENSITY_WH_L: Final[dict[str, float]] = {
    chem: DEFAULT_ASSUMPTIONS.energy_density_wh_l(chem) for chem in ("li_ion", "lipo")
}
PACK_ASPECT: Final[tuple[float, float, float]] = (3.0, 1.5, 1.0)  # L : W : H

# fmt: off
# (name, mass_kg, max_power_w, continuous_power_w, kv, voltage_v, diameter_m, length_m)
CRUISE_MOTORS: Final[list[tuple]] = [
    ("Outrunner 1806-2300KV", 0.025,   150,   100, 2300,   7.4, 0.023, 0.020),
    ("Outrunner 2212-1400KV", 0.056,   300,   200, 1400,  11.1, 0.028, 0.026),
    ("Outrunner 2814-900KV",  0.105,   550,   400,  900,  14.8, 0.035, 0.030),
    ("Outrunner 3520-520KV",  0.210,  1100,   800,  520,  22.2, 0.042, 0.038),
    ("Outrunner 4120-560KV",  0.300,  1500,  1100,  560,  22.2, 0.050, 0.040),
    ("Outrunner 4325-400KV",  0.420,  2200,  1600,  400,  22.2, 0.053, 0.045),
    ("Outrunner 5025-330KV",  0.550,  3000,  2200,  330,  44.4, 0.059, 0.050),
    ("Outrunner 5330-260KV",  0.750,  4000,  3000,  260,  44.4, 0.063, 0.055),
    ("Outrunner 6335-200KV",  1.050,  5500,  4000,  200,  44.4, 0.070, 0.062),
    ("Outrunner 7050-160KV",  1.500,  8000,  6000,  160,  50.0, 0.080, 0.070),
    ("Outrunner 8020-120KV",  2.100, 11000,  8000,  120,  50.0, 0.090, 0.075),
    ("Axial-flux AX-15",      3.200, 15000, 11000,  100, 100.0, 0.150, 0.060),
    ("Axial-flux AX-25",      5.000, 25000, 18000,   80, 200.0, 0.180, 0.070),
    ("Axial-flux AX-40",      8.500, 40000, 30000,   60, 300.0, 0.210, 0.080),
    ("Axial-flux AX-60",     12.000, 60000, 45000,   45, 400.0, 0.240, 0.090),
    ("Axial-flux AX-90",     17.000, 90000, 70000,   35, 400.0, 0.270, 0.100),
]

# (name, mass_kg, max_thrust_n, max_power_w, voltage_v, diameter_m, length_m)
VTOL_MOTORS: Final[list[tuple]] = [
    ("Lift 2306-1700KV",   0.033,    8,   180,  14.8, 0.028, 0.025),
    ("Lift 2807-1300KV",   0.050,   12,   250,  14.8, 0.034, 0.028),
    ("Lift 3110-900KV",    0.080,   20,   400,  22.2, 0.037, 0.030),
    ("Lift 4010-370KV",    0.110,   30,   500,  22.2, 0.046, 0.025),
    ("Lift 4114-400KV",    0.150,   45,   750,  22.2, 0.046, 0.030),
    ("Lift 5010-360KV",    0.180,   60,   900,  22.2, 0.056, 0.030),
    ("Lift 6215-180KV",    0.380,  120,  1800,  44.4, 0.068, 0.036),
    ("Lift 7215-135KV",    0.550,  180,  2600,  44.4, 0.075, 0.038),
    ("Lift 8318-100KV",    0.750,  250,  3500,  44.4, 0.088, 0.040),
    ("Lift 10025-80KV",    1.200,  400,  5800,  50.0, 0.105, 0.050),
    ("Lift 12030-60KV",    2.000,  650,  9000,  50.0, 0.125, 0.055),
    ("Lift 13530-50KV",    3.000,  900, 13000, 100.0, 0.140, 0.060),
    ("Lift 15040-40KV",    4.500, 1300, 19000, 100.0, 0.160, 0.065),
    ("Lift 18050-30KV",    6.500, 1800, 27000, 200.0, 0.190, 0.075),
    ("Lift 20060-25KV",    9.000, 2400, 36000, 400.0, 0.210, 0.085),
    ("Lift 24070-20KV",   12.500, 3200, 48000, 400.0, 0.250, 0.095),
]

# Frame ratings DERIVED from the propulsor model (fix ruling L1; `python -m src.tools.propulsion.motor_ratings`
# regenerates them, tests/test_motor_ratings.py checks every row): frame -> (KV, max current A, lift max thrust N).
# Each frame reaches its rating on its catalogue-paired propeller at MAX_MATCH_THROTTLE on its nominal voltage, at
# sea level, static. Lift thrust is the listed thrust or, where that needs more than max power on the paired rotor,
# what max power delivers. ESTIMATES. "Axial-flux AX-90" keeps its listed values: no catalogue propeller absorbs
# 90 kW static below tip Mach 0.80.
DERIVED_MOTOR_RATINGS: Final[dict[str, tuple[float, float, Optional[float]]]] = {
    'Lift 2306': (512, 12.2, 8),  # Carbon vtol prop 12x4in 6030 rpm 74 W
    'Lift 2807': (397, 16.9, 12),  # Carbon vtol prop 15x5in 4669 rpm 104 W
    'Lift 3110': (301, 18.1, 20),  # Carbon vtol prop 16x5.4in 5221 rpm 203 W
    'Lift 4010': (299, 22.6, 30),  # Carbon vtol prop 18x6in 5059 rpm 325 W
    'Lift 4114': (249, 33.8, 45),  # Carbon vtol prop 22x7in 4222 rpm 482 W
    'Lift 5010': (242, 41.5, 60),  # Carbon vtol prop 24x8in 4013 rpm 673 W
    'Lift 6215': (111, 46.8, 120),  # Carbon vtol prop 30x10in 3619 rpm 1497 W
    'Lift 7215': (96, 72.4, 180),  # Carbon vtol prop 36x12in 3075 rpm 2277 W
    'Lift 8318': (113, 113.4, 240),  # Carbon vtol prop 36x12in 3532 rpm 3490 W
    'Lift 10025': (69, 153.1, 400),  # Carbon vtol prop 48x16in 2566 rpm 5601 W
    'Lift 12030': (81, 247.6, 549),  # Carbon vtol prop 48x16in 2986 rpm 8986 W
    'Lift 13530': (38, 189.1, 779),  # Carbon vtol prop 56x18in 2647 rpm 12978 W
    'Lift 15040': (36, 274.9, 1074),  # Carbon vtol prop 62x20in 2523 rpm 18987 W
    'Lift 18050': (16, 200.8, 1501),  # Carbon vtol prop 72x24in 2181 rpm 26982 W
    'Lift 20060': (9, 137.6, 1814),  # Carbon vtol prop 72x24in 2386 rpm 35982 W
    'Lift 24070': (10, 187.4, 2154),  # Carbon vtol prop 72x24in 2596 rpm 47993 W
    'Outrunner 1806': (2168, 29.5, None),  # Carbon cruise prop 8x6in 11186 rpm 150 W
    'Outrunner 2212': (1288, 39.4, None),  # Carbon cruise prop 10x7in 9968 rpm 300 W
    'Outrunner 2814': (1009, 53.0, None),  # Carbon cruise prop 11x7in 10643 rpm 550 W
    'Outrunner 3520': (502, 70.7, None),  # Carbon cruise prop 15x10in 7935 rpm 1100 W
    'Outrunner 4120': (557, 96.3, None),  # Carbon cruise prop 15x10in 8819 rpm 1500 W
    'Outrunner 4325': (574, 141.3, None),  # Carbon cruise prop 16x10in 9082 rpm 2200 W
    'Outrunner 5025': (271, 96.4, None),  # Carbon cruise prop 18x10in 8570 rpm 3000 W
    'Outrunner 5330': (243, 128.0, None),  # Carbon cruise prop 20x12in 7717 rpm 4000 W
    'Outrunner 6335': (257, 176.7, None),  # Carbon cruise prop 22x10in 8127 rpm 5500 W
    'Outrunner 7050': (202, 216.1, None),  # Carbon cruise prop 24x12in 7608 rpm 8000 W
    'Outrunner 8020': (175, 299.1, None),  # Carbon cruise prop 28x14in 6542 rpm 11000 W
    'Axial-flux AX-15': (82, 215.5, None),  # Carbon cruise prop 32x16in 5797 rpm 15000 W
    'Axial-flux AX-25': (34, 182.5, None),  # Carbon cruise prop 40x20in 4729 rpm 25000 W
    'Axial-flux AX-40': (19, 199.3, None),  # Carbon cruise prop 50x24in 3873 rpm 40000 W
    'Axial-flux AX-60': (12, 227.6, None),  # Carbon cruise prop 60x30in 3212 rpm 60000 W
    # 'Axial-flux AX-90': listed values kept (no cruise propeller is rated for and absorbs 90000 W)
}


def frame_key(name: str) -> str:
    """A frame's name without its KV suffix ("Lift 12030-60KV" -> "Lift 12030")."""
    return name.rsplit("-", 1)[0] if name.endswith("KV") else name


def _rated_cruise_motors(raw: list[tuple]) -> list[tuple]:
    """Listed cruise frames with their derived KV (in the name when it carries one) and max current."""
    out: list[tuple] = []
    for name, mass, max_w, cont_w, kv, volts, dia, length in raw:
        rating = DERIVED_MOTOR_RATINGS.get(frame_key(name))
        if rating is None:
            out.append((name, mass, max_w, cont_w, kv, volts, dia, length))
            continue
        new_kv, imax, _ = rating
        new_name: str = f"{frame_key(name)}-{new_kv:g}KV" if name.endswith("KV") else name
        out.append((new_name, mass, max_w, cont_w, new_kv, volts, dia, length, imax))
    return out


def _rated_vtol_motors(raw: list[tuple]) -> list[tuple]:
    """Listed lift frames with their derived KV (in the name), max current and max thrust."""
    out: list[tuple] = []
    for name, mass, thrust, max_w, volts, dia, length in raw:
        new_kv, imax, rated = DERIVED_MOTOR_RATINGS[frame_key(name)]
        out.append((f"{frame_key(name)}-{new_kv:g}KV", mass, rated, max_w, volts, dia, length, imax))
    return out


# (name, mass_kg, continuous_power_w, bsfc_kg_kwh, fuel_type, length_m, width_m, height_m); the regulated
# output-voltage variants are appended by catalog_variants.generator_rows.
GENERATOR_FIELDS: Final[list[str]] = ["name", "mass_kg", "continuous_power_w", "bsfc_kg_kwh", "fuel_type",
                                      "length_m", "width_m", "height_m"]
ICE_GENERATORS: Final[list[tuple]] = [
    ("2-stroke genset 0.35 kW",   1.6,   350, 0.95, "gasoline", 0.20, 0.14, 0.14),
    ("2-stroke genset 0.6 kW",    2.4,   600, 0.85, "gasoline", 0.25, 0.18, 0.18),
    ("2-stroke genset 1.0 kW",    3.0,  1000, 0.78, "gasoline", 0.27, 0.19, 0.19),
    ("2-stroke genset 1.5 kW",    3.8,  1500, 0.72, "gasoline", 0.30, 0.20, 0.20),
    ("EFI genset 2 kW",           4.5,  2000, 0.65, "gasoline", 0.32, 0.22, 0.21),
    ("EFI genset 3 kW",           5.8,  3000, 0.60, "gasoline", 0.35, 0.24, 0.22),
    ("EFI genset 4 kW",           7.0,  4000, 0.58, "gasoline", 0.38, 0.25, 0.24),
    ("EFI genset 6 kW",           9.5,  6000, 0.55, "gasoline", 0.42, 0.28, 0.26),
    ("Heavy-fuel genset 8 kW",   13.0,  8000, 0.52, "jp8",      0.46, 0.30, 0.28),
    ("Heavy-fuel genset 10 kW",  15.5, 10000, 0.50, "jp8",      0.50, 0.32, 0.30),
    ("Rotary genset 15 kW",      19.0, 15000, 0.48, "gasoline", 0.48, 0.34, 0.32),
    ("Heavy-fuel genset 20 kW",  26.0, 20000, 0.45, "jp8",      0.60, 0.38, 0.35),
    ("Rotary genset 30 kW",      33.0, 30000, 0.42, "gasoline", 0.62, 0.42, 0.38),
    ("Heavy-fuel genset 40 kW",  45.0, 40000, 0.40, "jp8",      0.75, 0.48, 0.42),
    ("Heavy-fuel genset 55 kW",  58.0, 55000, 0.38, "jp8",      0.85, 0.52, 0.46),
    ("Heavy-fuel genset 75 kW",  75.0, 75000, 0.36, "jp8",      0.95, 0.58, 0.50),
]

# Cell classes packs are built from (`packs.build_pack`). ESTIMATES representative of current generic cells
# (energy / power / high-drain 18650 and 21700, 4680, large-format pouches; LiPo pouches with the high discharge
# currents a hybrid buffer needs), not datasheet values. Cylindrical cells: length x diameter x diameter.
# (name, mass_kg, chemistry, nominal_voltage_v, max_voltage_v, capacity_wh, max_current_a, length_m, width_m, height_m)
CELLS: Final[list[tuple]] = [
    ("Li-ion 18650 energy 3.5Ah",     0.048, "li_ion", 3.6, 4.2,  12.6,  10, 0.065, 0.0185, 0.0185),
    ("Li-ion 18650 power 3.0Ah",      0.047, "li_ion", 3.6, 4.2,  10.8,  20, 0.065, 0.0185, 0.0185),
    ("Li-ion 18650 high-drain 2.5Ah", 0.045, "li_ion", 3.6, 4.2,   9.0,  30, 0.065, 0.0185, 0.0185),
    ("Li-ion 21700 energy 5.0Ah",     0.069, "li_ion", 3.6, 4.2,  18.0,  10, 0.070, 0.021,  0.021),
    ("Li-ion 21700 balanced 4.5Ah",   0.068, "li_ion", 3.6, 4.2,  16.2,  15, 0.070, 0.021,  0.021),
    ("Li-ion 21700 power 4.2Ah",      0.070, "li_ion", 3.6, 4.2,  15.1,  30, 0.070, 0.021,  0.021),
    ("Li-ion 4680 22Ah",              0.355, "li_ion", 3.6, 4.2,  79.2, 100, 0.080, 0.046,  0.046),
    ("Li-ion pouch 10Ah",             0.150, "li_ion", 3.6, 4.2,  36.0,  20, 0.120, 0.060,  0.007),
    ("Li-ion pouch 50Ah",             0.720, "li_ion", 3.6, 4.2, 180.0,  50, 0.300, 0.100,  0.012),
    ("Li-ion pouch 100Ah",            1.400, "li_ion", 3.6, 4.2, 360.0, 100, 0.340, 0.140,  0.016),
    ("LiPo pouch 1.3Ah 50C",          0.030, "lipo",   3.7, 4.2,   4.81,  65, 0.070, 0.035,  0.007),
    ("LiPo pouch 2.2Ah 25C",          0.045, "lipo",   3.7, 4.2,   8.14,  55, 0.090, 0.035,  0.008),
    ("LiPo pouch 5Ah 25C",            0.105, "lipo",   3.7, 4.2,  18.5, 125, 0.140, 0.045,  0.0075),
    ("LiPo pouch 10Ah 25C",           0.205, "lipo",   3.7, 4.2,  37.0, 250, 0.160, 0.065,  0.009),
    ("LiPo pouch 16Ah 15C",           0.310, "lipo",   3.7, 4.2,  59.2, 240, 0.180, 0.075,  0.011),
    ("LiPo pouch 22Ah 25C",           0.450, "lipo",   3.7, 4.2,  81.4, 550, 0.200, 0.090,  0.012),
    ("LiPo pouch 30Ah 10C",           0.580, "lipo",   3.7, 4.2, 111.0, 300, 0.220, 0.100,  0.013),
]

# (diameter_in, pitch_in, mass_kg, max_power_w) — vendor I/O is in inches; converted to metres
CRUISE_PROPS_IN: Final[list[tuple]] = [
    (8, 6, 0.012, 250), (10, 7, 0.020, 450), (11, 7, 0.025, 600), (13, 8, 0.035, 1000),
    (15, 10, 0.055, 1600), (16, 10, 0.065, 2400), (18, 10, 0.090, 3300), (20, 12, 0.120, 4500),
    (22, 10, 0.160, 6500), (24, 12, 0.200, 9000), (28, 14, 0.300, 13000), (32, 16, 0.450, 19000),
    (40, 20, 0.800, 32000), (50, 24, 1.400, 50000), (60, 30, 2.200, 75000), (66, 32, 2.800, 100000),
]
VTOL_PROPS_IN: Final[list[tuple]] = [
    (9, 4.5, 0.010, 200), (12, 4, 0.018, 300), (15, 5, 0.030, 450), (16, 5.4, 0.035, 600),
    (18, 6, 0.050, 800), (22, 7, 0.080, 1900), (24, 8, 0.100, 2700), (28, 9, 0.150, 3600),
    (30, 10, 0.190, 4500), (32, 11, 0.240, 6000), (36, 12, 0.330, 9500), (40, 13, 0.450, 13500),
    (48, 16, 0.750, 20000), (56, 18, 1.100, 28000), (62, 20, 1.500, 37000), (72, 24, 2.200, 50000),
]

# Payload rows (researched gimbals + generic sensor classes) live in src/schemas/payload_catalog.py.
# fmt: on


def pack_envelope(
    capacity_wh: float, specific_energy_wh_kg: float, energy_density_wh_l: float
) -> tuple[float, float, float, float]:
    """Pack (mass_kg, length_m, width_m, height_m) from capacity, specific energy and energy density.

    The single source of pack geometry: every cell-built pack (`packs.build_pack`) takes its mass and box here.
    """
    volume_m3: float = capacity_wh / (energy_density_wh_l * 1000.0)
    unit: float = (volume_m3 / (PACK_ASPECT[0] * PACK_ASPECT[1] * PACK_ASPECT[2])) ** (1.0 / 3.0)
    return (
        # strictly above the exact mass (next whole gram), so float rounding never lifts a pack past its specific
        # energy or the 500 Wh/kg physical limit
        (math.floor(capacity_wh / specific_energy_wh_kg * MASS_DECIMALS_SCALE) + 1) / MASS_DECIMALS_SCALE,
        round(PACK_ASPECT[0] * unit, 4),
        round(PACK_ASPECT[1] * unit, 4),
        round(PACK_ASPECT[2] * unit, 4),
    )


def base_propellers(application: str) -> list[Propeller]:
    """The base (listed-pitch, non-folding) propellers of one application, in catalogue order."""
    table: list[tuple] = CRUISE_PROPS_IN if application == "cruise" else VTOL_PROPS_IN
    return [p for p in _propeller_rows() if p.application == application][:len(table)]


def _propeller_rows() -> list[Propeller]:
    return propeller_rows({"cruise": CRUISE_PROPS_IN, "vtol": VTOL_PROPS_IN})


def _tuple_rows(model: type[BaseModel], raw: list[tuple], fields: list[str] | None = None) -> list[BaseModel]:
    """Zip raw tuples with a field order (default: model order minus `id`) and validate each row."""
    names: list[str] = fields or [f for f in model.model_fields if f != "id"]
    return [model(**dict(zip(names, row, strict=True))) for row in raw]


def build_catalog() -> dict[str, list[BaseModel]]:
    """Return every catalog row, validated through its Pydantic schema."""
    cruise_motors: list[CruiseMotor] = cruise_motor_rows(_rated_cruise_motors(CRUISE_MOTORS))
    vtol_motors: list[VtolMotor] = vtol_motor_rows(_rated_vtol_motors(VTOL_MOTORS))
    return {
        "cruise_motors": cruise_motors,
        "vtol_motors": vtol_motors,
        "ice_generators": generator_rows(_tuple_rows(IceGenerator, ICE_GENERATORS, GENERATOR_FIELDS)),
        "cells": _tuple_rows(Cell, CELLS),
        "propellers": _propeller_rows(),
        "escs": esc_rows([*cruise_motors, *vtol_motors]),
        "payloads": list(PAYLOADS),
    }


@functools.lru_cache(maxsize=1)
def _cached_catalog() -> dict[str, tuple[BaseModel, ...]]:
    return {table: tuple(rows) for table, rows in build_catalog().items()}


def catalog_rows(table: str) -> list:
    """The validated rows `seed_database` inserts for `table` (built once; rows are frozen)."""
    if table not in CATALOG_TABLES:
        raise ValueError(f"Unknown catalog table {table!r}; expected one of {sorted(CATALOG_TABLES)}")
    return list(_cached_catalog()[table])


def _is_optional(annotation: object) -> bool:
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation)


def _sql_type(annotation: object) -> str:
    # Unwrap Optional[X] -> X; Literal[...] and str map to TEXT.
    if _is_optional(annotation):
        annotation = next(a for a in typing.get_args(annotation) if a is not type(None))
    if annotation is float:
        return "REAL"
    if annotation is int:
        return "INTEGER"
    return "TEXT"


def _create_table_sql(table: str, model: type[BaseModel]) -> str:
    columns: list[str] = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for name, info in model.model_fields.items():
        if name != "id":
            nullable: str = "" if _is_optional(info.annotation) else " NOT NULL"
            columns.append(f"{name} {_sql_type(info.annotation)}{nullable}")
    return f"CREATE TABLE {table} ({', '.join(columns)})"


def seed_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """(Re)create the catalog at `db_path`. Idempotent: tables are dropped and rebuilt."""
    catalog: dict[str, list[BaseModel]] = build_catalog()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # closing() releases the file handle; the inner `with conn` wraps everything in one transaction.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        for table, model in CATALOG_TABLES.items():
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(_create_table_sql(table, model))
            fields: list[str] = [f for f in model.model_fields if f != "id"]
            placeholders: str = ", ".join("?" for _ in fields)
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({placeholders})",
                [tuple(getattr(row, f) for f in fields) for row in catalog[table]],
            )
        conn.execute(f"PRAGMA user_version = {CATALOG_VERSION}")  # an int constant, never user input
    return db_path


def catalog_version(db_path: Path) -> int:
    """The catalog's schema/content version (SQLite user_version); 0 for a catalog seeded before versioning."""
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


if __name__ == "__main__":
    path: Path = seed_database()
    counts: dict[str, int] = {table: len(rows) for table, rows in build_catalog().items()}
    print(f"[Tool] seeded {path}: {counts}")
