"""Laser kit settings panel: the bed, sheet and tube stock, and the structural allowables, edited in the web form.

Session state is the store, in display units (mm for lengths, MPa for stresses; everything else in the model's own
unit): one `fab_<field>` key per scalar FabricationSettings field, plus the sheet and tube tables and the rod list.
The scalar inputs are generated from the model's fields (label from the description, bounds from the constraints),
so a new field appears here without editing this file. A loaded or re-run design is queued with `restore` and
copied into the widgets at the start of the next render, before they are drawn.
"""

import math
import re
from typing import Any, Final, Optional

import streamlit as st
from pydantic import ValidationError

from src.schemas.fabrication import ALL_FAMILIES, DEFAULT_FABRICATION, FabricationSettings

KEY_PREFIX: Final[str] = "fab_"
RESET_KEY: Final[str] = "reset_fabrication"
SHEETS_KEY: Final[str] = "fab_sheets_rows"  # the sheet table as last edited (display units)
TUBES_KEY: Final[str] = "fab_tubes_rows"
RODS_KEY: Final[str] = "fab_rods_mm"
SHEETS_SOURCE_KEY: Final[str] = "fab_sheets_source"  # the rows the editor was seeded with
TUBES_SOURCE_KEY: Final[str] = "fab_tubes_source"
TABLE_VERSION_KEY: Final[str] = "fab_table_version"  # bumped to make the data editors forget their edits
PENDING_KEY: Final[str] = "fab_pending"
ERROR_PREFIX: Final[str] = "Laser kit: "
FORMAT: Final[str] = "%g"
INT_FORMAT: Final[str] = "%d"
DEFAULT_STEP: Final[float] = 0.1
SIGNIFICANT_DIGITS: Final[int] = 12  # rounds away the float noise of mm <-> m and MPa <-> Pa conversions
ADVANCED_COLUMNS: Final[int] = 2
CORE_COLUMNS: Final[int] = 4

TABLE_FIELDS: Final[tuple[str, ...]] = ("sheets", "tubes", "rods_m")
SCALAR_FIELDS: Final[tuple[str, ...]] = tuple(n for n in FabricationSettings.model_fields if n not in TABLE_FIELDS)
TABLE_NOUNS: Final[dict[str, str]] = {"sheets": "Sheet", "tubes": "Tube", "rods_m": "Rod"}
CORE_LABELS: Final[dict[str, str]] = {
    "bed_width_m": "Bed width (mm)",
    "bed_height_m": "Bed height (mm)",
    "margin_m": "Margin (mm)",
    "kerf_m": "Kerf (mm)",
}
# (suffix, display unit, factor from the model's unit): lengths show in mm, stresses in MPa.
_UNITS: Final[tuple[tuple[str, str, float], ...]] = (("_m", "mm", 1e3), ("_pa", "MPa", 1e-6))
_UNIT_TOKENS: Final[dict[str, str]] = {"(m)": "(mm)", "(Pa)": "(MPa)"}
MM: Final[float] = 1e-3

SHEET_HELP: Final[str] = ("One row per sheet stock; Families lists the part families cut from it, comma separated "
                          f"({', '.join(ALL_FAMILIES)}). Each family goes on exactly one sheet.")
TUBE_HELP: Final[str] = "Carbon tube stock the sizing picks from (outer and inner diameter)."
RODS_HELP: Final[str] = "Solid carbon rod diameters (mm), comma separated."
FAMILY_SEPARATORS: Final[str] = r"[,\s]+"


# ------------------------------------------------------------ conversions ---
def _unit(name: str) -> tuple[str, float]:
    return next(((unit, factor) for suffix, unit, factor in _UNITS if name.endswith(suffix)), ("", 1.0))


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _is_int(name: str) -> bool:
    return FabricationSettings.model_fields[name].annotation is int


def _to_display(name: str, value: float) -> float:
    return int(value) if _is_int(name) else _round(value * _unit(name)[1])


def _to_model(name: str, value: float) -> float:
    return int(value) if _is_int(name) else _round(value / _unit(name)[1])


def _label(name: str) -> str:
    if name in CORE_LABELS:
        return CORE_LABELS[name]
    text = (FabricationSettings.model_fields[name].description or name).rstrip(".")
    if _unit(name)[0]:
        for token, shown in _UNIT_TOKENS.items():
            text = text.replace(token, shown, 1)
    return text


def field_label(name: str) -> str:
    """How the panel names a setting: its input's label, or the table it is edited in."""
    if name in TABLE_FIELDS:
        return f"{TABLE_NOUNS[name]}s"
    return _label(name)


def same_as_shown(a: FabricationSettings, b: FabricationSettings) -> bool:
    """The two settings show the same values in the panel (display rounding ignored)."""
    return _as_shown(a) == _as_shown(b)


def _words(name: str) -> str:
    """A field name as plain words (unit suffix dropped), for the cross-field messages the model words itself."""
    return re.sub(r"_(pa|m|kg_m2|kg_m3|g_m2)$", "", name).replace("_", " ")


def display_values(settings: FabricationSettings) -> dict[str, float]:
    """The scalar settings in display units, keyed by field name."""
    return {name: _to_display(name, getattr(settings, name)) for name in SCALAR_FIELDS}


def sheet_rows(settings: FabricationSettings) -> list[dict[str, Any]]:
    return [{"Name": s.name, "Thickness (mm)": _round(s.thickness_m / MM), "Density (kg/m3)": s.density_kg_m3,
             "Families": ", ".join(s.families)} for s in settings.sheets]


def tube_rows(settings: FabricationSettings) -> list[dict[str, Any]]:
    return [{"OD (mm)": _round(t.od_m / MM), "ID (mm)": _round(t.id_m / MM)} for t in settings.tubes]


def rods_text(settings: FabricationSettings) -> str:
    return ", ".join(f"{_round(d / MM):g}" for d in settings.rods_m)


# ------------------------------------------------------------- validation ---
class _CellError(ValueError):
    """A table cell that is blank or not a number; the message names the row in user terms."""


def _cell(row: dict[str, Any], column: str, where: str) -> float:
    try:
        number = float(row.get(column))  # type: ignore[arg-type]  # None and text raise below
    except (TypeError, ValueError):
        raise _CellError(f"{where}: {column} must be a number") from None
    if math.isnan(number):
        raise _CellError(f"{where}: {column} must be a number")
    return number


def _is_blank(row: dict[str, Any]) -> bool:
    return all(v is None or v == "" or (isinstance(v, float) and math.isnan(v)) for v in row.values())


def _sheet(row: dict[str, Any], where: str) -> dict[str, Any]:
    name = str(row.get("Name") or "").strip()
    if not name:
        raise _CellError(f"{where}: Name is required")
    families = tuple(f for f in re.split(FAMILY_SEPARATORS, str(row.get("Families") or "")) if f)
    unknown = [f for f in families if f not in ALL_FAMILIES]
    if unknown:
        raise _CellError(f"{where}: unknown part family {unknown[0]!r} (use {', '.join(ALL_FAMILIES)})")
    return {"name": name, "thickness_m": _round(_cell(row, "Thickness (mm)", where) * MM),
            "density_kg_m3": _cell(row, "Density (kg/m3)", where), "families": families}


def _tube(row: dict[str, Any], where: str) -> dict[str, float]:
    return {"od_m": _round(_cell(row, "OD (mm)", where) * MM), "id_m": _round(_cell(row, "ID (mm)", where) * MM)}


def _rods(text: str) -> list[float]:
    try:
        return [_round(float(part) * MM) for part in re.split(FAMILY_SEPARATORS, text.strip()) if part]
    except ValueError:
        raise _CellError("Rod diameters: use numbers in mm separated by commas") from None


def _table_data(sheets: list[dict[str, Any]], tubes: list[dict[str, Any]],
                rods: str) -> tuple[dict[str, Any], list[str]]:
    data: dict[str, Any] = {}
    errors: list[str] = []
    for key, rows, parse, noun in (("sheets", sheets, _sheet, "Sheet"), ("tubes", tubes, _tube, "Tube")):
        parsed: list[Any] = []
        for number, row in enumerate((r for r in rows if not _is_blank(r)), start=1):
            try:
                parsed.append(parse(row, f"{noun} row {number}"))
            except _CellError as exc:
                errors.append(f"{ERROR_PREFIX}{exc}")
        data[key] = parsed
    try:
        data["rods_m"] = _rods(rods)
    except _CellError as exc:
        errors.append(f"{ERROR_PREFIX}{exc}")
    return data, errors


def _describe(error: Any) -> str:
    """One pydantic error in user terms: the field's label for field errors, raw field names replaced otherwise."""
    loc, msg = error["loc"], error["msg"].removeprefix("Value error, ")
    if "id_m" in msg and "od_m" in msg:  # the tube rule, worded by the model in metres
        msg = "inner diameter must be below outer diameter"
    if error["type"] == "too_short" and len(loc) == 1 and loc[0] in TABLE_FIELDS:
        return f"{ERROR_PREFIX}{TABLE_NOUNS[str(loc[0])]} stock needs at least one entry"
    if loc and loc[0] in SCALAR_FIELDS:
        return f"{ERROR_PREFIX}{_label(str(loc[0]))}: {msg}"
    if loc and loc[0] in TABLE_FIELDS:
        where = TABLE_NOUNS[str(loc[0])]
        row = f" row {int(loc[1]) + 1}" if len(loc) > 1 and isinstance(loc[1], int) else ""
        return f"{ERROR_PREFIX}{where}{row}: {msg}"
    for name in SCALAR_FIELDS:
        msg = re.sub(rf"\b{name}\b", _words(name), msg)
    return f"{ERROR_PREFIX}{msg}"


def _as_shown(settings: FabricationSettings) -> tuple[Any, ...]:
    """Everything the panel shows of `settings`; two settings that look alike in the panel compare equal."""
    return (display_values(settings), sheet_rows(settings), tube_rows(settings), rods_text(settings))


def build_settings(values: dict[str, float], sheets: list[dict[str, Any]], tubes: list[dict[str, Any]],
                   rods: str) -> tuple[Optional[FabricationSettings], list[str]]:
    """Validated settings from display-unit panel values, or (None, messages in user terms)."""
    tables, errors = _table_data(sheets, tubes, rods)
    errors = [*(f"{ERROR_PREFIX}{_label(n)}: enter a number" for n in SCALAR_FIELDS if values.get(n) is None),
              *errors]
    if errors:
        return None, errors
    data = {name: _to_model(name, values[name]) for name in SCALAR_FIELDS} | tables
    try:
        return FabricationSettings.model_validate(data), []
    except ValidationError as exc:
        return None, [_describe(e) for e in exc.errors()]


# ------------------------------------------------------------------ state ---
def _key(name: str) -> str:
    return f"{KEY_PREFIX}{name}"


def _load(settings: FabricationSettings) -> None:
    """Copy `settings` into the widgets' keys; the data editors restart from the new rows."""
    for name, value in display_values(settings).items():
        st.session_state[_key(name)] = value
    sheets, tubes = sheet_rows(settings), tube_rows(settings)
    st.session_state.update({SHEETS_SOURCE_KEY: sheets, SHEETS_KEY: sheets, TUBES_SOURCE_KEY: tubes,
                             TUBES_KEY: tubes, RODS_KEY: rods_text(settings),
                             TABLE_VERSION_KEY: st.session_state.get(TABLE_VERSION_KEY, 0) + 1})


def _init_state() -> None:
    if _key(SCALAR_FIELDS[0]) not in st.session_state:
        _load(DEFAULT_FABRICATION)


def _reset() -> None:
    """on_click callback: runs before widgets are instantiated, so the keys may be reassigned."""
    st.session_state.pop(PENDING_KEY, None)  # a queued design must not win over an explicit reset
    _load(DEFAULT_FABRICATION)


def current_fabrication() -> tuple[Optional[FabricationSettings], list[str]]:
    """The panel's values as validated settings, or (None, messages) when anything is invalid."""
    _init_state()
    values = {name: st.session_state[_key(name)] for name in SCALAR_FIELDS}
    return build_settings(values, st.session_state[SHEETS_KEY], st.session_state[TUBES_KEY],
                          st.session_state[RODS_KEY])


def restore(settings: FabricationSettings) -> bool:
    """Show a loaded or re-run design's own settings in the panel (applied at the next render).

    True when they were queued, i.e. the panel holds something else; the caller reruns when it can."""
    current = current_fabrication()[0]
    if current is not None and _as_shown(current) == _as_shown(settings):
        return False
    st.session_state[PENDING_KEY] = settings
    return True


def _apply_pending() -> None:
    pending: Optional[FabricationSettings] = st.session_state.pop(PENDING_KEY, None)
    if pending is not None:
        _load(pending)


# ---------------------------------------------------------------- widgets ---
def _step(name: str) -> float:
    default = _to_display(name, getattr(DEFAULT_FABRICATION, name))
    if _is_int(name):
        return 1
    return 10.0 ** (math.floor(math.log10(default)) - 1) if default > 0 else DEFAULT_STEP


def _bounds(name: str) -> tuple[Optional[float], Optional[float]]:
    """(min, max) in display units. A strict lower bound (> 0) is offered as inclusive, so a zero can be typed
    and is then reported by the model's own check instead of being silently refused by the box."""
    marks = {type(m).__name__: m for m in FabricationSettings.model_fields[name].metadata}
    low = next((getattr(marks[k], k.lower()) for k in ("Ge", "Gt") if k in marks), None)
    high = marks["Le"].le if "Le" in marks else None
    return (None if low is None else _to_display(name, low)), (None if high is None else _to_display(name, high))


def _number(name: str) -> None:
    low, high = _bounds(name)
    st.number_input(_label(name), min_value=low, max_value=high, step=_step(name),
                    format=INT_FORMAT if _is_int(name) else FORMAT, key=_key(name),
                    help=FabricationSettings.model_fields[name].description)


def _bed_inputs() -> None:
    for column, name in zip(st.columns(CORE_COLUMNS), CORE_LABELS):
        with column:
            _number(name)


def _table(title: str, help_text: str, source_key: str, value_key: str, editor_key: str) -> None:
    st.markdown(f"**{title}**", help=help_text)
    edited = st.data_editor(st.session_state[source_key], key=editor_key, num_rows="dynamic", hide_index=True,
                            width="stretch")
    # a list in gives a list out; a DataFrame (older Streamlit, or a frame in session state) gives a frame
    st.session_state[value_key] = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)


def _stock_tables() -> None:
    version = st.session_state[TABLE_VERSION_KEY]
    sheets_col, tubes_col = st.columns(2)
    with sheets_col:
        _table("Sheet stock", SHEET_HELP, SHEETS_SOURCE_KEY, SHEETS_KEY, f"fab_sheets_editor_{version}")
    with tubes_col:
        _table("Carbon tube stock", TUBE_HELP, TUBES_SOURCE_KEY, TUBES_KEY, f"fab_tubes_editor_{version}")
    st.text_input("Carbon rod diameters (mm)", key=RODS_KEY, help=RODS_HELP)


def _advanced_inputs() -> None:
    st.caption("Stock sizes, material properties and structural allowables the kit sizing uses. Defaults are "
               "typical values; change them to match your material test data.")
    names = [n for n in SCALAR_FIELDS if n not in CORE_LABELS]
    for offset, column in enumerate(st.columns(ADVANCED_COLUMNS)):
        with column:
            for name in names[offset::ADVANCED_COLUMNS]:
                _number(name)


def render_fabrication_panel() -> None:
    _init_state()
    _apply_pending()
    with st.expander("Laser kit settings"):
        st.caption("The laser bed, sheet and tube stock, and structural allowables for the cut kit of Group 1-2 "
                   "designs. Defaults describe a 600 x 400 mm bed and birch ply; they apply to both forms.")
        _bed_inputs()
        _stock_tables()
        with st.popover("Structural allowables (advanced)"):  # secondary: a popover keeps it out of the way
            _advanced_inputs()
        st.button("Reset to defaults", key=RESET_KEY, on_click=_reset)
        for message in current_fabrication()[1]:
            st.error(message)
