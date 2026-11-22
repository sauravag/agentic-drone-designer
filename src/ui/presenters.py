"""Pure helpers for the web form: form -> validated requirements, and results -> display tables.

Kept free of Streamlit so they are unit-testable; app.py only lays out widgets and calls these.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any, Final, Optional

from pydantic import ValidationError

from src.schemas.assumptions import DesignAssumptions
from src.schemas.components import SelectedBOM
from src.schemas.fabrication import FabricationSettings
from src.schemas.kit_files import KIT_DOWNLOADS, KIT_SUBDIR, SHEET_GLOB, SHEETS_DIR, stock_slug
from src.schemas.payload_catalog import payload_display_name
from src.schemas.requirements import ARK_JETSON_PAB, CUBE_ORANGE_PLUS, OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, PlantCaseResult, SizingOutput
from src.tools.flight_control.runtime import MAX_OVERSHOOT, MAX_SETTLING_S, MAX_TRANSITION_ALT_LOSS_M
from src.units import WH_TO_J

FIELD_LABELS: Final[dict[str, str]] = {
    "mission_name": "Mission name",
    "group_constraint": "DoD group",
    "propulsion_type": "Propulsion",
    "airframe_topology": "Airframe",
    "payload_model": "Payload",
    "payload_mass_kg": "Payload mass (kg)",
    "payload_power_w": "Payload power (W)",
    "endurance_hours": "Endurance (h)",
    "combat_radius_km": "Combat radius (km)",
    "cruise_speed_ms": "Cruise speed (m/s)",
    "service_ceiling_m": "Service ceiling (m)",
    "max_stall_speed_ms": "Max stall speed at take-off (m/s)",
    "design_wind_ms": "Design headwind (m/s)",
    "takeoff_altitude_m": "Take-off altitude (m)",
    "isa_temperature_offset_c": "ISA temperature offset (°C)",
    "runway_available": "Runway available",
    "flight_controller": "Flight controller",
    "companion_computer": "Jetson companion computer",
}
FLIGHT_CONTROLLER_NAMES: Final[dict[str, str]] = {
    CUBE_ORANGE_PLUS: "Cube Orange+", ARK_JETSON_PAB: "ARK Jetson PAB (Orin NX 16 GB)",
}
# Hardware choices are the designer's, not the mission's: presets never overwrite them.
HARDWARE_KEYS: Final[tuple[str, ...]] = ("flight_controller", "companion_computer")
NEW_DESIGN_HARDWARE: Final[dict[str, Any]] = {"flight_controller": CUBE_ORANGE_PLUS, "companion_computer": True}
ARK_BUILT_IN: Final[str] = "built into the ARK stack"
AUTO: Final[str] = "auto"
# Form sections -> (field, min, max, step, format). Bounds sit inside OperationalRequirements'
# constraints; tests/test_ui.py pins that, and that every preset fits.
NUMBER_FIELDS: Final[dict[str, tuple[tuple[str, float, float, float, str], ...]]] = {
    # Payload formats keep every catalogue value exact (0.5335 kg, 3.5 W): the box rounds what it holds.
    "Payload": (("payload_mass_kg", 0.01, 200.0, 0.1, "%.4f"), ("payload_power_w", 0.0, 5000.0, 5.0, "%.1f")),
    "Mission": (("endurance_hours", 0.2, 48.0, 0.5, "%.1f"), ("combat_radius_km", 1.5, 1000.0, 5.0, "%.0f"),
                ("cruise_speed_ms", 8.5, 120.0, 1.0, "%.1f"), ("design_wind_ms", 0.0, 40.0, 1.0, "%.1f")),
    "Envelope": (("service_ceiling_m", 150.0, 10000.0, 100.0, "%.0f"), ("max_stall_speed_ms", 5.5, 45.0, 0.5, "%.1f"),
                 ("takeoff_altitude_m", 0.0, 5000.0, 100.0, "%.0f"),
                 ("isa_temperature_offset_c", -40.0, 40.0, 1.0, "%.0f")),
}
DOWNLOAD_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".step", ".vsp3", ".param", ".md", ".pdf", ".xml"})
SHEET_STEM_NUMBER: Final[re.Pattern[str]] = re.compile(r"-\d+$")
SECONDS_PER_MINUTE: Final[float] = 60.0
WORST_PLANT_LABEL: Final[str] = ", worst plant"  # suffix for step rows that hold the worst over the plants
IN_AVIONICS_ALLOWANCE: Final[str] = "in avionics allowance"
PAYLOAD_CAPTION: Final[str] = ("Masses and powers are datasheet values (typical power). NextVision masses are "
                                "probably the bare camera, so add your mount if it's separate.")
NO_PRICE: Final[str] = "—"  # generic and Custom payload rows have no reference price


def parse_form(values: dict[str, Any]) -> tuple[Optional[OperationalRequirements], list[str]]:
    """Validate raw widget values against the requirements schema; errors use form labels."""
    cleaned: dict[str, Any] = dict(values)
    if cleaned.get("group_constraint") == AUTO:
        cleaned["group_constraint"] = None
    try:
        return OperationalRequirements(**cleaned), []
    except ValidationError as exc:
        errors = [
            f"{FIELD_LABELS.get(str(err['loc'][0]), str(err['loc'][0])) if err['loc'] else 'Form'}: {err['msg']}"
            for err in exc.errors()
        ]
        return None, errors


def form_defaults(req: OperationalRequirements) -> dict[str, Any]:
    """Widget values for a preset (None group -> 'auto')."""
    values: dict[str, Any] = req.model_dump()
    values["group_constraint"] = AUTO if req.group_constraint is None else req.group_constraint
    return values


def avionics_values(req: OperationalRequirements) -> dict[str, str]:
    """Display values for the two avionics requirement rows (names, and whether a Jetson is fitted)."""
    is_ark: bool = req.flight_controller == ARK_JETSON_PAB
    return {"flight_controller": FLIGHT_CONTROLLER_NAMES[req.flight_controller],
            "companion_computer": ARK_BUILT_IN if is_ark else ("yes" if req.companion_computer else "no")}


def payload_display(req: OperationalRequirements) -> str:
    """The payload's name: a catalogue model (marked when its mass or power was edited) or Custom."""
    return payload_display_name(req.payload_model, req.payload_mass_kg, req.payload_power_w)


def payload_summary(req: OperationalRequirements) -> str:
    """One line for the results headline, e.g. 'SIYI ZT30 (0.854 kg, 9 W)'."""
    return f"{payload_display(req)} ({req.payload_mass_kg:g} kg, {req.payload_power_w:g} W)"


def avionics_summary(req: OperationalRequirements, assumptions: DesignAssumptions) -> str:
    """One line naming the fitted avionics stack, with the Jetson's power and mass when one is fitted."""
    a = assumptions
    if req.flight_controller == ARK_JETSON_PAB:
        return f"ARK Jetson PAB + Orin NX 16 GB ({a.jetson_power_w:g} W, {a.ark_pab_stack_mass_kg:g} kg)"
    if req.companion_computer:
        return f"Cube Orange+ · Jetson Orin NX companion ({a.jetson_power_w:g} W, {a.companion_computer_mass_kg:g} kg)"
    return "Cube Orange+ · no Jetson"


def sizing_summary(s: SizingOutput) -> list[tuple[str, str]]:
    """Headline sizing results as (label, formatted value) pairs."""
    return [
        ("MTOW", f"{s.mtow_kg:.2f} kg"),
        ("DoD group", f"Group {s.dod_group}"),
        ("Airframe", s.airframe_topology.replace("_", " ")),
        ("Propulsion", s.propulsion_type.replace("_", " ")),
        ("Airfoil", s.airfoil),
        ("Wingspan", f"{s.wingspan_m:.2f} m"),
        ("Wing area", f"{s.wing_area_m2:.3f} m²"),
        ("Aspect ratio", f"{s.aspect_ratio:.1f}"),
        ("Fuselage (L × D)", f"{s.fuselage_length_m:.2f} × {s.fuselage_diameter_m:.2f} m"),
        ("Cruise L/D", f"{s.assumed_ld_cruise:.1f}"),
        ("Stall speed (SL)", f"{s.v_stall_ms:.1f} m/s"),
        *([("Stall speed at take-off site", f"{s.v_stall_takeoff_ms:.1f} m/s")]
          if s.v_stall_takeoff_ms is not None else []),
        ("Loiter speed", f"{s.v_loiter_ms:.1f} m/s"),
        ("Empty mass", f"{s.empty_mass_kg:.2f} kg"),
        ("Battery mass", f"{s.battery_mass_kg:.2f} kg"),
        ("Fuel mass", f"{s.fuel_mass_kg:.2f} kg"),
        ("Cruise power", f"{s.cruise_power_w:.0f} W"),
        ("Peak hover power", f"{s.hover_power_w:.0f} W"),
    ]


def phase_rows(s: SizingOutput) -> list[dict[str, Any]]:
    """One row per mission phase (duration, power, energy, fuel)."""
    return [
        {"Phase": p.name, "Duration (min)": round(p.duration_s / SECONDS_PER_MINUTE, 1),
         "Propulsive power (W)": round(p.propulsive_power_w), "Total power (W)": round(p.total_power_w),
         "Energy (Wh)": round(p.energy_j / WH_TO_J, 1), "Fuel burned (kg)": round(p.fuel_burned_kg, 3)}
        for p in s.phases
    ]


def bom_rows(bom: SelectedBOM) -> list[dict[str, Any]]:
    """One row per part (quantity-weighted), then the total hardware mass."""
    parts: list[tuple[str, Any, int]] = [
        ("Cruise motor", bom.cruise_motor, 1), ("Cruise propeller", bom.cruise_propeller, 1),
        ("Battery", bom.battery, 1), ("Payload", bom.payload, 1),
        ("Generator", bom.ice_generator, 1),
        ("Lift motor", bom.vtol_motor, bom.vtol_motor_count), ("Lift propeller", bom.vtol_propeller, bom.vtol_motor_count),
    ]
    rows = [
        {"Part": label, "Item": part.name, "Qty": qty, "Unit mass (kg)": round(part.mass_kg, 3),
         "Mass (kg)": round(qty * part.mass_kg, 3), "Power (W)": None, "Ref. price": _part_price(label, part)}
        for label, part, qty in parts if part is not None
    ]
    rows += [_avionics_row(label, part) for label, part in
             (("Flight controller", bom.flight_controller), ("Companion computer", bom.companion_computer))
             if part is not None]
    rows.append({"Part": "Total", "Item": "", "Qty": None, "Unit mass (kg)": None,
                 "Mass (kg)": round(bom.total_hardware_mass_kg, 3), "Power (W)": None, "Ref. price": ""})
    return rows


def _part_price(label: str, part: Any) -> str:
    """Only the payload carries a reference price among these parts (generic and Custom rows show a dash)."""
    return (part.reference_price_usd or NO_PRICE) if label == "Payload" else ""


def _avionics_row(label: str, part: Any) -> dict[str, Any]:
    # Text throughout the column ("in avionics allowance" or "25"), so Streamlit's Arrow table has one type.
    power: str = IN_AVIONICS_ALLOWANCE if part.power_w is None else f"{part.power_w:g}"
    return {"Part": label, "Item": part.name, "Qty": 1, "Unit mass (kg)": round(part.mass_kg, 3),
            "Mass (kg)": round(part.mass_kg, 3), "Power (W)": power, "Ref. price": part.reference_price_usd}


def aero_rows(aero: AeroStabilityOutput) -> list[dict[str, str]]:
    """VSPAERO stability and control derivatives (body axes, per radian)."""
    entries: list[tuple[str, str, float, str]] = [
        ("Zero-alpha lift", "CL0", aero.cl0, ""), ("Lift-curve slope", "CLα", aero.cl_alpha_per_rad, "/rad"),
        ("Zero-lift drag", "CD0", aero.cd0, ""), ("Cruise lift-to-drag", "L/D", aero.ld_cruise, ""),
        ("Pitch stability", "Cmα", aero.cm_alpha_per_rad, "/rad"),
        ("Weathercock stability", "Cnβ", aero.cn_beta_per_rad, "/rad"),
        ("Dihedral effect", "Clβ", aero.cl_beta_per_rad, "/rad"),
        ("Aileron power", "Clδa", aero.cl_delta_a_per_rad, "/rad"),
        ("Elevator power", "Cmδe", aero.cm_delta_e_per_rad, "/rad"),
        ("Rudder power", "Cnδr", aero.cn_delta_r_per_rad, "/rad"),
        ("Neutral point", "X_np", aero.x_np_m, "m aft of nose"),
    ]
    return [{"Quantity": q, "Symbol": sym, "Value": f"{v:.4g}", "Unit": u} for q, sym, v, u in entries]


def kit_summary(cad: CADArtifacts, settings: Optional[FabricationSettings] = None) -> Optional[str]:
    """E.g. "Laser kit: 1.84 kg, 6 sheets (4 × Birch ply 1.5 mm, 2 × Birch ply 3 mm)"; None without a kit.

    The sheet count comes from the DXF files actually in the kit folder, grouped by stock; `settings` supplies the
    stock names (the file names only carry their slugs)."""
    if cad.kit_dir is None:
        return None
    names = {stock_slug(sheet.name): sheet.name for sheet in settings.sheets} if settings else {}
    per_stock = Counter(SHEET_STEM_NUMBER.sub("", p.stem) for p in (Path(cad.kit_dir) / SHEETS_DIR).glob(SHEET_GLOB))
    total = sum(per_stock.values())
    groups = ", ".join(f"{n} × {names.get(slug, slug.replace('-', ' '))}" for slug, n in sorted(per_stock.items()))
    mass = "" if cad.kit_mass_kg is None else f"{cad.kit_mass_kg:.2f} kg, "
    return f"Laser kit: {mass}{total} sheet{'' if total == 1 else 's'}" + (f" ({groups})" if groups else "")


def cad_rows(cad: CADArtifacts, settings: Optional[FabricationSettings] = None) -> list[tuple[str, str]]:
    """Packing, mass and inertia results as (label, formatted value) pairs; the kit follows the as-built mass."""
    kit = kit_summary(cad, settings)
    return [
        ("Static margin", f"{cad.static_margin:.1%}"),
        ("CG", f"{cad.x_cg_m:.3f} m aft of nose"),
        ("Interferences", str(cad.collision_count)),
        ("As-built mass", f"{cad.as_built_mass_kg:.2f} kg"),
        *([("Laser kit", kit.removeprefix("Laser kit: "))] if kit else []),
        ("Ixx / Iyy / Izz", f"{cad.ixx_kgm2:.3g} / {cad.iyy_kgm2:.3g} / {cad.izz_kgm2:.3g} kg·m²"),
        ("Ixz", f"{cad.ixz_kgm2:.3g} kg·m²"),
    ]


def fcs_rows(fcs: FCSArtifacts) -> list[dict[str, str]]:
    """SITL doublet-test results against the SRS limits; with perturbed plants the step rows are the worst plant."""
    worst: str = WORST_PLANT_LABEL if fcs.perturbation_cases else ""
    entries: list[tuple[str, float, float, str]] = [
        (f"Roll overshoot{worst} (+15° step)", fcs.roll_overshoot, MAX_OVERSHOOT, "%"),
        (f"Roll settling time{worst}", fcs.roll_settling_s, MAX_SETTLING_S, "s"),
        (f"Pitch overshoot{worst} (+10° step)", fcs.pitch_overshoot, MAX_OVERSHOOT, "%"),
        (f"Pitch settling time{worst}", fcs.pitch_settling_s, MAX_SETTLING_S, "s"),
        ("Transition altitude loss", fcs.max_transition_altitude_loss_m, MAX_TRANSITION_ALT_LOSS_M, "m"),
    ]
    rows = [{"Check": name, "Result": _fmt(value, unit), "Limit": f"< {_fmt(limit, unit)}",
             "Pass": "yes" if value < limit else "no"} for name, value, limit, unit in entries]
    rows.append({"Check": "Rate-gain scale / attempts", "Result": f"{fcs.gain_scale:.3f} / {fcs.attempts}",
                 "Limit": "", "Pass": "yes" if fcs.passed else "no"})
    if fcs.perturbation_cases:
        names: str = ", ".join(c.case for c in fcs.perturbation_cases)
        rows.append({"Check": f"Worst case (of {names})", "Result": fcs.governing_case, "Limit": "",
                     "Pass": "yes" if fcs.passed else "no"})
        rows += [_plant_row(c) for c in fcs.perturbation_cases]
    return rows


def _plant_row(case: PlantCaseResult) -> dict[str, str]:
    """One plant's roll/pitch result, gated on the same SRS limits as the nominal rows."""
    if case.error:
        return {"Check": f"Plant: {case.case}", "Result": f"crashed: {case.error}", "Limit": "", "Pass": "no"}
    passed: bool = (max(case.roll_overshoot, case.pitch_overshoot) < MAX_OVERSHOOT
                    and max(case.roll_settling_s, case.pitch_settling_s) < MAX_SETTLING_S)
    result: str = (f"roll {case.roll_overshoot:.0%} / {case.roll_settling_s:.1f} s, "
                   f"pitch {case.pitch_overshoot:.0%} / {case.pitch_settling_s:.1f} s")
    return {"Check": f"Plant: {case.case}", "Result": result, "Limit": "", "Pass": "yes" if passed else "no"}


def _fmt(value: float, unit: str) -> str:
    return f"{value:.0%}" if unit == "%" else f"{value:.2f} {unit}"


def read_report(out_dir: Optional[Path], report_path: Optional[Path]) -> Optional[str]:
    """design_report.md text if it sits directly in the run folder and is still readable, else None."""
    if out_dir is None or report_path is None:
        return None
    path: Path = report_path.resolve()
    if path.parent != out_dir.resolve() or path.is_symlink():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:  # run folder deleted while the result is still on screen
        return None


def _is_plain_file_in(path: Path, folder: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.resolve().parent == folder


def _real_folder(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _kit_files(kit: Path) -> list[Path]:
    """The build sheet and parts list, then the DXF sheets (regular files only, never through a symlink)."""
    if not _real_folder(kit):
        return []
    files = [p for p in (kit / name for name in KIT_DOWNLOADS) if _is_plain_file_in(p, kit.resolve())]
    sheets = kit / SHEETS_DIR
    if _real_folder(sheets):
        files += sorted(p for p in sheets.glob(SHEET_GLOB) if _is_plain_file_in(p, sheets.resolve()))
    return files


def downloadable_files(out_dir: Path) -> list[Path]:
    """Regular artifact files directly inside the run folder (no symlinks, no sub-folders), then the kit's build
    sheet, parts list and DXF sheets from its `kit` folder."""
    root: Path = out_dir.resolve()
    if not root.is_dir():  # run folder deleted while the result is still on screen
        return []
    files = sorted(p for p in root.iterdir() if p.suffix in DOWNLOAD_SUFFIXES and _is_plain_file_in(p, root))
    return [*files, *_kit_files(root / KIT_SUBDIR)]
