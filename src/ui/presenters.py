"""Pure helpers for the web form: form -> validated requirements, and results -> display tables.

Kept free of Streamlit so they are unit-testable; app.py only lays out widgets and calls these.
"""

from pathlib import Path
from typing import Any, Final, Optional

from pydantic import ValidationError

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, SizingOutput
from src.tools.flight_control.runtime import MAX_OVERSHOOT, MAX_SETTLING_S, MAX_TRANSITION_ALT_LOSS_M
from src.units import WH_TO_J

FIELD_LABELS: Final[dict[str, str]] = {
    "mission_name": "Mission name",
    "group_constraint": "DoD group",
    "propulsion_type": "Propulsion",
    "airframe_topology": "Airframe",
    "payload_mass_kg": "Payload mass (kg)",
    "payload_power_w": "Payload power (W)",
    "endurance_hours": "Endurance (h)",
    "combat_radius_km": "Combat radius (km)",
    "cruise_speed_ms": "Cruise speed (m/s)",
    "service_ceiling_m": "Service ceiling (m)",
    "max_stall_speed_ms": "Max stall speed (m/s)",
    "runway_available": "Runway available",
}
AUTO: Final[str] = "auto"
# Form sections -> (field, min, max, step, format). Bounds sit inside OperationalRequirements'
# constraints; tests/test_ui.py pins that, and that every preset fits.
NUMBER_FIELDS: Final[dict[str, tuple[tuple[str, float, float, float, str], ...]]] = {
    "Payload": (("payload_mass_kg", 0.01, 200.0, 0.1, "%.2f"), ("payload_power_w", 0.0, 5000.0, 5.0, "%.0f")),
    "Mission": (("endurance_hours", 0.2, 48.0, 0.5, "%.1f"), ("combat_radius_km", 1.5, 1000.0, 5.0, "%.0f"),
                ("cruise_speed_ms", 8.5, 120.0, 1.0, "%.1f")),
    "Envelope": (("service_ceiling_m", 150.0, 10000.0, 100.0, "%.0f"), ("max_stall_speed_ms", 5.5, 45.0, 0.5, "%.1f")),
}
DOWNLOAD_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".step", ".vsp3", ".param", ".md", ".xml"})
SECONDS_PER_MINUTE: Final[float] = 60.0
IN_AVIONICS_ALLOWANCE: Final[str] = "in avionics allowance"


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
         "Mass (kg)": round(qty * part.mass_kg, 3), "Power (W)": None, "Ref. price": ""}
        for label, part, qty in parts if part is not None
    ]
    rows += [_avionics_row(label, part) for label, part in
             (("Flight controller", bom.flight_controller), ("Companion computer", bom.companion_computer))
             if part is not None]
    rows.append({"Part": "Total", "Item": "", "Qty": None, "Unit mass (kg)": None,
                 "Mass (kg)": round(bom.total_hardware_mass_kg, 3), "Power (W)": None, "Ref. price": ""})
    return rows


def _avionics_row(label: str, part: Any) -> dict[str, Any]:
    power: Any = IN_AVIONICS_ALLOWANCE if part.power_w is None else part.power_w
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


def cad_rows(cad: CADArtifacts) -> list[tuple[str, str]]:
    """Packing, mass and inertia results as (label, formatted value) pairs."""
    return [
        ("Static margin", f"{cad.static_margin:.1%}"),
        ("CG", f"{cad.x_cg_m:.3f} m aft of nose"),
        ("Interferences", str(cad.collision_count)),
        ("As-built mass", f"{cad.as_built_mass_kg:.2f} kg"),
        ("Ixx / Iyy / Izz", f"{cad.ixx_kgm2:.3g} / {cad.iyy_kgm2:.3g} / {cad.izz_kgm2:.3g} kg·m²"),
        ("Ixz", f"{cad.ixz_kgm2:.3g} kg·m²"),
    ]


def fcs_rows(fcs: FCSArtifacts) -> list[dict[str, str]]:
    """SITL doublet-test results against the SRS limits."""
    entries: list[tuple[str, float, float, str]] = [
        ("Roll overshoot (+15° step)", fcs.roll_overshoot, MAX_OVERSHOOT, "%"),
        ("Roll settling time", fcs.roll_settling_s, MAX_SETTLING_S, "s"),
        ("Pitch overshoot (+10° step)", fcs.pitch_overshoot, MAX_OVERSHOOT, "%"),
        ("Pitch settling time", fcs.pitch_settling_s, MAX_SETTLING_S, "s"),
        ("Transition altitude loss", fcs.max_transition_altitude_loss_m, MAX_TRANSITION_ALT_LOSS_M, "m"),
    ]
    rows = [{"Check": name, "Result": _fmt(value, unit), "Limit": f"< {_fmt(limit, unit)}",
             "Pass": "yes" if value < limit else "no"} for name, value, limit, unit in entries]
    rows.append({"Check": "Rate-gain scale / attempts", "Result": f"{fcs.gain_scale:.3f} / {fcs.attempts}",
                 "Limit": "", "Pass": "yes" if fcs.passed else "no"})
    return rows


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


def downloadable_files(out_dir: Path) -> list[Path]:
    """Regular artifact files directly inside the run folder (no symlinks, no sub-folders)."""
    root: Path = out_dir.resolve()
    if not root.is_dir():  # run folder deleted while the result is still on screen
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.is_symlink() and p.suffix in DOWNLOAD_SUFFIXES and p.resolve().parent == root
    )
