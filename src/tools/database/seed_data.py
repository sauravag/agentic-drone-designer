"""Generates data/cots_catalog.db: a representative COTS catalog spanning Group 1 (~2 kg) to
Group 3 (~500 kg) UAVs.

NOTE: names are generic product classes and figures are representative of typical vendor
datasheets, not quotes for a specific SKU. Verify against real datasheets before procurement.
Run directly: `python -m src.tools.database.seed_data`
"""

import sqlite3
import types
from contextlib import closing
import typing
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.components import Battery, CruiseMotor, IceGenerator, Payload, Propeller, VtolMotor
from src.units import INCH_TO_M

DEFAULT_DB_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "cots_catalog.db"

CATALOG_TABLES: Final[dict[str, type[BaseModel]]] = {
    "cruise_motors": CruiseMotor,
    "vtol_motors": VtolMotor,
    "ice_generators": IceGenerator,
    "batteries": Battery,
    "propellers": Propeller,
    "payloads": Payload,
}

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

# (name, mass_kg, continuous_power_w, bsfc_kg_kwh, fuel_type, length_m, width_m, height_m)
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

# (name, chemistry, capacity_wh, max_c_rate, voltage_v) — mass/dims derived from pack constants
BATTERIES: Final[list[tuple]] = [
    ("Li-Ion 21700 3S1P",     "li_ion",    54.0,  3.0,  10.8),
    ("Li-Ion 21700 4S2P",     "li_ion",   144.0,  3.0,  14.4),
    ("Li-Ion 21700 6S2P",     "li_ion",   216.0,  3.0,  21.6),
    ("Li-Ion 21700 6S4P",     "li_ion",   432.0,  3.0,  21.6),
    ("Li-Ion 21700 6S6P",     "li_ion",   648.0,  3.0,  21.6),
    ("Li-Ion 21700 12S4P",    "li_ion",   864.0,  3.0,  43.2),
    ("Li-Ion 21700 12S6P",    "li_ion",  1296.0,  3.0,  43.2),
    ("Li-Ion 21700 12S10P",   "li_ion",  2160.0,  3.0,  43.2),
    ("Li-Ion 21700 14S12P",   "li_ion",  3024.0,  3.0,  50.4),
    ("Li-Ion HV pack 5 kWh",  "li_ion",  5000.0,  3.0, 400.0),
    ("Li-Ion HV pack 15 kWh", "li_ion", 15000.0,  3.0, 400.0),
    ("LiPo 3S 2200mAh",       "lipo",      24.4, 25.0,  11.1),
    ("LiPo 4S 5000mAh",       "lipo",      74.0, 25.0,  14.8),
    ("LiPo 6S 10000mAh",      "lipo",     222.0, 25.0,  22.2),
    ("LiPo 6S 16000mAh",      "lipo",     355.0, 25.0,  22.2),
    ("LiPo 12S 16000mAh",     "lipo",     710.0, 25.0,  44.4),
    ("LiPo 12S 22000mAh",     "lipo",     977.0, 25.0,  44.4),
    ("LiPo HV pack 2 kWh",    "lipo",    2000.0, 25.0, 100.0),
    ("LiPo HV pack 4 kWh",    "lipo",    4000.0, 25.0, 400.0),
    ("LiPo HV pack 6 kWh",    "lipo",    6000.0, 25.0, 400.0),
    ("LiPo HV pack 8 kWh",    "lipo",    8000.0, 25.0, 400.0),
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

# (name, sensor_type, mass_kg, power_w, length_m, width_m, height_m)
PAYLOADS: Final[list[tuple]] = [
    ("Micro EO 2-axis gimbal",          "eo",            0.10,   5, 0.06, 0.05, 0.07),
    ("Multispectral 5-band camera",     "multispectral", 0.17,   7, 0.09, 0.06, 0.05),
    ("Mini EO/IR gimbal",               "eo_ir",         0.35,  10, 0.08, 0.08, 0.10),
    ("Compact EO/IR 3-axis zoom",       "eo_ir",         0.80,  18, 0.11, 0.11, 0.14),
    ("LiDAR mapping unit",              "lidar",         1.20,  25, 0.15, 0.10, 0.10),
    ("EO/IR 30x zoom + LRF gimbal",     "eo_ir",         1.40,  25, 0.13, 0.13, 0.17),
    ("Comms relay radio",               "comms_relay",   1.50,  35, 0.18, 0.12, 0.06),
    ("Mini SAR",                        "sar",           1.80,  60, 0.25, 0.15, 0.10),
    ("SIGINT receiver",                 "sigint",        2.50,  40, 0.22, 0.16, 0.08),
    ("Hyperspectral imager",            "hyperspectral", 3.00,  45, 0.25, 0.15, 0.15),
    ("EO/IR stabilised gimbal 3.5 kg",  "eo_ir",         3.50,  60, 0.18, 0.18, 0.25),
    ("EO/IR turret 7 kg",               "eo_ir",         7.00, 120, 0.23, 0.23, 0.32),
    ("SAR/GMTI pod",                    "sar",          12.00, 400, 0.70, 0.25, 0.25),
    ("MWIR EO/IR turret 15 kg",         "eo_ir",        15.00, 250, 0.30, 0.30, 0.42),
    ("Long-range MWIR turret 28 kg",    "eo_ir",        28.00, 450, 0.38, 0.38, 0.52),
    ("Strategic EO/IR turret 45 kg",    "eo_ir",        45.00, 800, 0.45, 0.45, 0.62),
]
# fmt: on


def _battery_rows() -> list[Battery]:
    """Derive pack mass and a box envelope from capacity using chemistry constants."""
    rows: list[Battery] = []
    for name, chem, wh, c_rate, volts in BATTERIES:
        volume_m3: float = wh / (PACK_ENERGY_DENSITY_WH_L[chem] * 1000.0)
        unit: float = (volume_m3 / (PACK_ASPECT[0] * PACK_ASPECT[1] * PACK_ASPECT[2])) ** (1.0 / 3.0)
        rows.append(
            Battery(
                name=name, chemistry=chem, capacity_wh=wh, max_c_rate=c_rate, voltage_v=volts,
                mass_kg=round(wh / PACK_SPECIFIC_ENERGY_WH_KG[chem], 3),
                length_m=round(PACK_ASPECT[0] * unit, 4),
                width_m=round(PACK_ASPECT[1] * unit, 4),
                height_m=round(PACK_ASPECT[2] * unit, 4),
            )
        )
    return rows


def _propeller_rows() -> list[Propeller]:
    rows: list[Propeller] = []
    for application, table in (("cruise", CRUISE_PROPS_IN), ("vtol", VTOL_PROPS_IN)):
        for dia_in, pitch_in, mass, power in table:
            rows.append(
                Propeller(
                    name=f"Carbon {application} prop {dia_in}x{pitch_in}in",
                    diameter_m=round(dia_in * INCH_TO_M, 4), pitch_m=round(pitch_in * INCH_TO_M, 4),
                    mass_kg=mass, max_power_w=power, application=application,
                )
            )
    return rows


PAYLOAD_FIELDS: Final[list[str]] = ["name", "sensor_type", "mass_kg", "power_w", "length_m", "width_m", "height_m"]


def _tuple_rows(model: type[BaseModel], raw: list[tuple], fields: list[str] | None = None) -> list[BaseModel]:
    """Zip raw tuples with a field order (default: model order minus `id`) and validate each row."""
    names: list[str] = fields or [f for f in model.model_fields if f != "id"]
    return [model(**dict(zip(names, row, strict=True))) for row in raw]


def build_catalog() -> dict[str, list[BaseModel]]:
    """Return every catalog row, validated through its Pydantic schema."""
    return {
        "cruise_motors": _tuple_rows(CruiseMotor, CRUISE_MOTORS),
        "vtol_motors": _tuple_rows(VtolMotor, VTOL_MOTORS),
        "ice_generators": _tuple_rows(IceGenerator, ICE_GENERATORS),
        "batteries": _battery_rows(),
        "propellers": _propeller_rows(),
        "payloads": _tuple_rows(Payload, PAYLOADS, PAYLOAD_FIELDS),
    }


def _sql_type(annotation: object) -> str:
    # Unwrap Optional[X] -> X; Literal[...] and str map to TEXT.
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
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
            columns.append(f"{name} {_sql_type(info.annotation)} NOT NULL")
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
    return db_path


if __name__ == "__main__":
    path: Path = seed_database()
    counts: dict[str, int] = {table: len(rows) for table, rows in build_catalog().items()}
    print(f"[Tool] seeded {path}: {counts}")
