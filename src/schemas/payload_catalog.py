"""The payload catalogue: researched EO / EO-IR gimbals (NextVision, SIYI) plus the generic sensor rows.

This is the single source of payload rows. `seed_data` seeds the `payloads` table from `PAYLOADS`. The UI prefill,
the description parser and requirement changes look models up here without touching the database.
Catalogue names are append-only: a renamed model keeps its old name in `ALIASES`.
"""

import re
from typing import Final, Optional

from src.schemas.components import Payload
from src.schemas.requirements import CUSTOM_PAYLOAD

__all__ = ["CUSTOM_PAYLOAD", "PAYLOADS", "PAYLOAD_RESEARCH_DATE", "payload_choices", "payload_label", "payload_spec",
           "resolve_payload_name"]

PAYLOAD_RESEARCH_DATE: Final[str] = "2026-11-18"
MM_TO_M: Final[float] = 0.001
QUOTE_ONLY: Final[str] = "quote only"
NOT_PUBLIC: Final[str] = "not public"
CUSTOM_LABEL: Final[str] = "Custom payload"
SENSOR_LABELS: Final[dict[str, str]] = {"eo": "EO", "eo_ir": "EO/IR"}

# Datasheet values, researched 2026-11-18 from:
#   - nextvision-sys.com/compare/ and the individual camera pages;
#   - siyi.biz spec pages;
#   - Reebot's MT11 page.
# Power is typical/average (continuous). NextVision doesn't say what its masses include (probably the bare camera).
# SIYI masses include: ZT6 its plate, ZR30/ZT30 their mounts, MT11 its board.
# fmt: off
# (vendor, model, sensor_type, mass_kg, power_w, length_mm, width_mm, height_mm, reference_price_usd)
GIMBALS: Final[tuple[tuple, ...]] = (
    ("NextVision", "DragonEye2",     "eo_ir", 0.125,  3.5,  40.0,  40.0,  65.0, QUOTE_ONLY),
    ("NextVision", "X80",            "eo",    0.145,  4.5,  53.0,  58.0,  75.0, QUOTE_ONLY),
    ("NextVision", "Starling",       "eo_ir", 0.180,  6.0,  84.0,  58.0,  82.0, QUOTE_ONLY),
    ("NextVision", "Starling-LRF",   "eo_ir", 0.230,  6.0,  78.0,  66.0,  91.0, QUOTE_ONLY),
    ("NextVision", "Stinger",        "eo_ir", 0.230,  7.0,  61.0,  60.0,  91.0, QUOTE_ONLY),
    ("NextVision", "NightHawk2",     "eo_ir", 0.250,  8.0,  64.0,  64.0,  94.0, QUOTE_ONLY),
    ("NextVision", "Swift",          "eo_ir", 0.290,  6.0,  64.0,  67.0,  97.0, QUOTE_ONLY),
    ("NextVision", "NightHawk2-V",   "eo_ir", 0.320,  5.0,  64.0,  64.0,  95.0, QUOTE_ONLY),
    ("NextVision", "NightHawk2-UZ",  "eo_ir", 0.320,  6.0,  64.0,  64.0,  94.0, QUOTE_ONLY),
    ("NextVision", "Raptor",         "eo_ir", 0.640,  7.0,  85.0,  85.0, 129.0, QUOTE_ONLY),
    ("NextVision", "Condor",         "eo_ir", 1.500, 12.0, 180.0, 183.0, 211.0, QUOTE_ONLY),
    ("NextVision", "Raven 2.5",      "eo_ir", 2.150, 20.0, 154.0, 174.0, 238.0, QUOTE_ONLY),
    ("SIYI",       "A8 mini",        "eo",    0.095,  5.0,  55.0,  55.0,  70.0, "≈ $251"),
    ("SIYI",       "ZT6",            "eo_ir", 0.197,  6.5,  73.5,  75.0, 131.5, "≈ $2,940–3,700"),
    ("SIYI",       "ZR10",           "eo",    0.381,  4.0, 121.0, 101.0,  78.0, "≈ $532"),
    ("SIYI",       "UniPod MT11",    "eo_ir", 0.5335, 11.4, 141.5, 141.5, 169.0, NOT_PUBLIC),
    ("SIYI",       "UniPod NV10",    "eo",    0.628, 15.0, 141.5, 141.5, 183.0, NOT_PUBLIC),
    ("SIYI",       "ZR30",           "eo",    0.668,  5.0, 132.0, 100.0, 180.0, "≈ $1,181"),
    ("SIYI",       "ZT30",           "eo_ir", 0.854,  9.0, 134.5, 140.0, 201.0, "≈ $7,000"),
)

# Generic sensor classes (no vendor): what Custom payloads are matched against for the parts list.
# (name, sensor_type, mass_kg, power_w, length_m, width_m, height_m)
GENERIC: Final[tuple[tuple, ...]] = (
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
)
# fmt: on
ALIASES: Final[dict[str, str]] = {}  # old catalogue name -> current name (names are append-only)


def _gimbal(vendor: str, model: str, sensor: str, mass: float, power: float, length_mm: float, width_mm: float,
            height_mm: float, price: str) -> Payload:
    return Payload(name=f"{vendor} {model}", vendor=vendor, sensor_type=sensor, mass_kg=mass, power_w=power,
                   length_m=length_mm * MM_TO_M, width_m=width_mm * MM_TO_M, height_m=height_mm * MM_TO_M,
                   reference_price_usd=price)


def _generic(name: str, sensor: str, mass: float, power: float, length: float, width: float, height: float) -> Payload:
    return Payload(name=name, sensor_type=sensor, mass_kg=mass, power_w=power, length_m=length, width_m=width,
                   height_m=height)


PAYLOADS: Final[tuple[Payload, ...]] = (
    tuple(_gimbal(*row) for row in GIMBALS) + tuple(_generic(*row) for row in GENERIC)
)
_BY_NAME: Final[dict[str, Payload]] = {p.name: p for p in PAYLOADS}


def payload_spec(name: str) -> Optional[Payload]:
    """The catalogue row for an exact name, or for an alias of a renamed one. None for Custom or an unknown name."""
    return _BY_NAME.get(ALIASES.get(name, name))


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_payload_name(text: str) -> Optional[str]:
    """A catalogue gimbal named loosely: any case or punctuation, vendor optional (e.g. 'Siyi ZT-30', 'raptor')."""
    wanted: str = _key(text)
    for p in PAYLOADS:
        if p.vendor is not None and wanted in (_key(p.name), _key(p.name.removeprefix(f"{p.vendor} "))):
            return p.name
    return None


def payload_choices() -> tuple[str, ...]:
    """Dropdown order: gimbals by vendor then mass, then generic rows by mass, then Custom."""
    gimbals = sorted((p for p in PAYLOADS if p.vendor is not None), key=lambda p: (p.vendor, p.mass_kg))
    generic = sorted((p for p in PAYLOADS if p.vendor is None), key=lambda p: p.mass_kg)
    return tuple(p.name for p in (*gimbals, *generic)) + (CUSTOM_PAYLOAD,)


def payload_label(name: str) -> str:
    """Dropdown text, e.g. 'SIYI ZT30 — EO/IR, 0.854 kg, 9 W' or 'Generic · Mini SAR — 1.80 kg, 60 W'."""
    part: Optional[Payload] = payload_spec(name)
    if part is None:
        return CUSTOM_LABEL
    if part.vendor is None:
        return f"Generic · {part.name} — {part.mass_kg:.2f} kg, {part.power_w:g} W"
    sensor: str = SENSOR_LABELS.get(part.sensor_type, part.sensor_type)
    return f"{part.name} — {sensor}, {part.mass_kg:g} kg, {part.power_w:g} W"
