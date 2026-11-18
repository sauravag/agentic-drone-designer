"""Design assumptions panel: the designer's technology constants, edited in the web form.

Session state is the store (one `assume_<field>` key per DesignAssumptions field, in the model's own unit, no
hidden scaling); defaults come from DEFAULT_ASSUMPTIONS on first render.
"""

from typing import Final, Optional

import streamlit as st
from pydantic import ValidationError

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions

KEY_PREFIX: Final[str] = "assume_"
RESET_KEY: Final[str] = "reset_assumptions"
FORMAT: Final[str] = "%g"
STEP_FRACTION: Final[float] = 0.01
STEP_WIDE: Final[float] = 1.0

# (field, label with unit). Groups follow what the sizing code actually uses: the electric pack is Li-ion, the
# hybrid's buffer pack is LiPo (src/tools/sizing/mission_profile.py).
GROUPS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "Battery (electric, Li-ion)": (
        ("liion_specific_energy_wh_kg", "Li-ion specific energy (Wh/kg)"),
        ("liion_energy_density_wh_l", "Li-ion energy density (Wh/L)"),
        ("liion_max_c_rate", "Li-ion max discharge rate (C)"),
        ("battery_reserve_fraction", "Battery reserve (fraction)"),
    ),
    "Hybrid buffer pack (LiPo)": (
        ("lipo_specific_energy_wh_kg", "LiPo specific energy (Wh/kg)"),
        ("lipo_energy_density_wh_l", "LiPo energy density (Wh/L)"),
        ("lipo_max_c_rate", "LiPo max discharge rate (C)"),
    ),
    "Hybrid engine": (
        ("bsfc_kg_per_kwh", "BSFC (kg/kWh)"),
        ("fuel_reserve_fraction", "Fuel reserve (fraction)"),
    ),
    "Efficiencies": (
        ("drivetrain_efficiency", "Drivetrain efficiency (fraction)"),
        ("hover_figure_of_merit", "Hover figure of merit (fraction)"),
        ("lift_motor_efficiency", "Lift motor efficiency (fraction)"),
    ),
    "Avionics": (
        ("avionics_base_mass_kg", "Allowance base mass (kg)"),
        ("avionics_mass_per_mtow", "Allowance mass per kg MTOW (kg/kg)"),
        ("avionics_base_power_w", "Allowance base power (W)"),
        ("avionics_power_per_kg_w", "Allowance power per kg MTOW (W/kg)"),
        ("cube_orange_plus_mass_kg", "Cube Orange+ mass (kg)"),
        ("ark_pab_stack_mass_kg", "ARK PAB stack mass (kg)"),
        ("companion_computer_mass_kg", "Jetson companion mass (kg)"),
        ("jetson_power_w", "Jetson power (W)"),
    ),
}
GROUP_HELP: Final[dict[str, str]] = {
    "Battery (electric, Li-ion)": "Used for battery-electric designs (the sizing code sizes the electric pack as Li-ion).",
    "Hybrid buffer pack (LiPo)": "Used for the hybrid's buffer pack (the sizing code sizes it as LiPo).",
    "Avionics": ("Avionics mass = allowance base + per-kg MTOW + the chosen flight-controller stack "
                 "(Cube or ARK) + the Jetson companion (Cube with the Jetson ticked only). Avionics power = "
                 "allowance base + per-kg MTOW (incl. the flight controller's own draw) + Jetson power whenever "
                 "a Jetson is fitted (ARK always). Both are drawn for the whole flight."),
}
_FIELDS: Final[tuple[str, ...]] = tuple(f for fields in GROUPS.values() for f, _ in fields)


def _key(field: str) -> str:
    return f"{KEY_PREFIX}{field}"


def _init_state() -> None:
    for field in _FIELDS:
        st.session_state.setdefault(_key(field), float(getattr(DEFAULT_ASSUMPTIONS, field)))


def _reset() -> None:
    """on_click callback: runs before widgets are instantiated, so the keys may be reassigned."""
    for field in _FIELDS:
        st.session_state[_key(field)] = float(getattr(DEFAULT_ASSUMPTIONS, field))


def _step(field: str) -> float:
    return STEP_FRACTION if getattr(DEFAULT_ASSUMPTIONS, field) < STEP_WIDE else STEP_WIDE


def _number(field: str, label: str) -> None:
    info = DesignAssumptions.model_fields[field]
    bounds = {m.__class__.__name__: m for m in info.metadata}
    st.number_input(label, min_value=float(bounds["Ge"].ge), max_value=float(bounds["Le"].le), step=_step(field),
                    format=FORMAT, key=_key(field), help=info.description)


def current_assumptions() -> tuple[Optional[DesignAssumptions], list[str]]:
    """The panel's values as a validated model, or (None, messages) when any value is out of bounds."""
    _init_state()
    try:
        return DesignAssumptions(**{f: st.session_state[_key(f)] for f in _FIELDS}), []
    except ValidationError as exc:
        return None, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def render_assumptions_panel() -> None:
    _init_state()
    with st.expander("Design assumptions"):
        st.caption("Technology constants the sizing chain uses. Defaults reproduce the standard results; they apply "
                   "to both the description and the requirements forms.")
        for column, (title, fields) in zip(st.columns(len(GROUPS)), GROUPS.items()):
            with column:
                st.subheader(title, help=GROUP_HELP.get(title))
                for field, label in fields:
                    _number(field, label)
        model, errors = current_assumptions()
        st.button("Reset to defaults", key=RESET_KEY, on_click=_reset)
        for message in errors:
            st.error(message)
        if model is not None and (changed := model.non_default()):
            st.caption("Non-default: " + ", ".join(f"{k} = {v[1]:g} (default {v[0]:g})" for k, v in changed.items()))
