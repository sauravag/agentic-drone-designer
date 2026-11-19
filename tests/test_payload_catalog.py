"""Selectable payloads: the researched gimbal catalogue, the payload model on the requirements, its parts-list
line, requirement changes, the explorer and the display helpers."""

import sqlite3
from pathlib import Path
from typing import Optional

import pytest
from pydantic import ValidationError

from src import pipeline
from src.agent import explorer
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes
from src.schemas.components import ComponentNeeds
from src.schemas.requirements import OperationalRequirements
from src.schemas.payload_catalog import (
    CUSTOM_PAYLOAD,
    PAYLOADS,
    payload_choices,
    payload_label,
    payload_spec,
    resolve_payload_name,
)
from src.tools.database.matcher import load_catalog, match_payload, payload_part, select_bom
from src.tools.database.seed_data import CATALOG_VERSION, catalog_version, seed_database
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from src.ui.examples import EXAMPLE_MISSIONS

MM = 0.001
QUOTE = "quote only"
NOT_PUBLIC = "not public"

# (name, sensor_type, mass_kg, power_w, length_mm, width_mm, height_mm, price) from the 2026-11-18 research.
GIMBALS = [
    ("NextVision DragonEye2", "eo_ir", 0.125, 3.5, 40, 40, 65, QUOTE),
    ("NextVision X80", "eo", 0.145, 4.5, 53, 58, 75, QUOTE),
    ("NextVision Starling", "eo_ir", 0.180, 6, 84, 58, 82, QUOTE),
    ("NextVision Starling-LRF", "eo_ir", 0.230, 6, 78, 66, 91, QUOTE),
    ("NextVision Stinger", "eo_ir", 0.230, 7, 61, 60, 91, QUOTE),
    ("NextVision NightHawk2", "eo_ir", 0.250, 8, 64, 64, 94, QUOTE),
    ("NextVision Swift", "eo_ir", 0.290, 6, 64, 67, 97, QUOTE),
    ("NextVision NightHawk2-V", "eo_ir", 0.320, 5, 64, 64, 95, QUOTE),
    ("NextVision NightHawk2-UZ", "eo_ir", 0.320, 6, 64, 64, 94, QUOTE),
    ("NextVision Raptor", "eo_ir", 0.640, 7, 85, 85, 129, QUOTE),
    ("NextVision Condor", "eo_ir", 1.500, 12, 180, 183, 211, QUOTE),
    ("NextVision Raven 2.5", "eo_ir", 2.150, 20, 154, 174, 238, QUOTE),
    ("SIYI A8 mini", "eo", 0.095, 5, 55, 55, 70, "≈ $251"),
    ("SIYI ZT6", "eo_ir", 0.197, 6.5, 73.5, 75, 131.5, "≈ $2,940–3,700"),
    ("SIYI ZR10", "eo", 0.381, 4, 121, 101, 78, "≈ $532"),
    ("SIYI UniPod MT11", "eo_ir", 0.5335, 11.4, 141.5, 141.5, 169, NOT_PUBLIC),
    ("SIYI UniPod NV10", "eo", 0.628, 15, 141.5, 141.5, 183, NOT_PUBLIC),
    ("SIYI ZR30", "eo", 0.668, 5, 132, 100, 180, "≈ $1,181"),
    ("SIYI ZT30", "eo_ir", 0.854, 9, 134.5, 140, 201, "≈ $7,000"),
]

# The generic rows exactly as seeded before the gimbals were added (pin: Custom parts lists must not change).
GENERIC = [
    ("Micro EO 2-axis gimbal", "eo", 0.10, 5, 0.06, 0.05, 0.07),
    ("Multispectral 5-band camera", "multispectral", 0.17, 7, 0.09, 0.06, 0.05),
    ("Mini EO/IR gimbal", "eo_ir", 0.35, 10, 0.08, 0.08, 0.10),
    ("Compact EO/IR 3-axis zoom", "eo_ir", 0.80, 18, 0.11, 0.11, 0.14),
    ("LiDAR mapping unit", "lidar", 1.20, 25, 0.15, 0.10, 0.10),
    ("EO/IR 30x zoom + LRF gimbal", "eo_ir", 1.40, 25, 0.13, 0.13, 0.17),
    ("Comms relay radio", "comms_relay", 1.50, 35, 0.18, 0.12, 0.06),
    ("Mini SAR", "sar", 1.80, 60, 0.25, 0.15, 0.10),
    ("SIGINT receiver", "sigint", 2.50, 40, 0.22, 0.16, 0.08),
    ("Hyperspectral imager", "hyperspectral", 3.00, 45, 0.25, 0.15, 0.15),
    ("EO/IR stabilised gimbal 3.5 kg", "eo_ir", 3.50, 60, 0.18, 0.18, 0.25),
    ("EO/IR turret 7 kg", "eo_ir", 7.00, 120, 0.23, 0.23, 0.32),
    ("SAR/GMTI pod", "sar", 12.00, 400, 0.70, 0.25, 0.25),
    ("MWIR EO/IR turret 15 kg", "eo_ir", 15.00, 250, 0.30, 0.30, 0.42),
    ("Long-range MWIR turret 28 kg", "eo_ir", 28.00, 450, 0.38, 0.38, 0.52),
    ("Strategic EO/IR turret 45 kg", "eo_ir", 45.00, 800, 0.45, 0.45, 0.62),
]


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    return seed_database(tmp_path / "catalog.db")


# ------------------------------------------------------------------ catalogue ---
@pytest.mark.parametrize("name, sensor, mass, power, length, width, height, price", GIMBALS)
def test_gimbal_rows_match_research(name: str, sensor: str, mass: float, power: float, length: float,
                                    width: float, height: float, price: str) -> None:
    part = payload_spec(name)
    assert part is not None
    assert (part.sensor_type, part.mass_kg, part.power_w) == (sensor, mass, power)
    assert (part.length_m, part.width_m, part.height_m) == pytest.approx((length * MM, width * MM, height * MM))
    assert part.vendor == name.split()[0]
    assert part.reference_price_usd == price


def test_generic_rows_unchanged() -> None:
    generic = [p for p in PAYLOADS if p.vendor is None]
    assert [(p.name, p.sensor_type, p.mass_kg, p.power_w, p.length_m, p.width_m, p.height_m) for p in generic] == GENERIC
    assert all(p.reference_price_usd is None for p in generic)


def test_choices_order() -> None:
    choices = payload_choices()
    nextvision, siyi, generic = choices[:12], choices[12:19], choices[19:35]
    assert all(c.startswith("NextVision ") for c in nextvision) and all(c.startswith("SIYI ") for c in siyi)
    for block in (nextvision, siyi, generic):
        masses = [payload_spec(c).mass_kg for c in block]
        assert masses == sorted(masses)
    assert [payload_spec(c).vendor for c in generic] == [None] * 16
    assert choices[-1] == CUSTOM_PAYLOAD and len(choices) == 36


@pytest.mark.parametrize("text, expected", [
    ("zt30", "SIYI ZT30"), ("Siyi ZT-30", "SIYI ZT30"), ("SIYI ZT30", "SIYI ZT30"),
    ("NextVision Raptor", "NextVision Raptor"), ("raptor", "NextVision Raptor"),
    ("Foo 9000", None), ("custom", None),
])
def test_resolve_payload_name(text: str, expected: object) -> None:
    assert resolve_payload_name(text) == expected


def test_payload_label() -> None:
    assert payload_label("SIYI ZT30") == "SIYI ZT30 — EO/IR, 0.854 kg, 9 W"
    assert payload_label("Mini SAR") == "Generic · Mini SAR — 1.80 kg, 60 W"
    assert payload_label(CUSTOM_PAYLOAD) == "Custom payload"


def test_unknown_and_custom_have_no_spec() -> None:
    assert payload_spec("Foo 9000") is None and payload_spec(CUSTOM_PAYLOAD) is None


def test_seeded_db_has_gimbals_with_vendor(catalog: Path) -> None:
    rows = load_catalog(catalog, "payloads")
    assert len(rows) == 35
    assert next(p for p in rows if p.name == "SIYI ZT30").vendor == "SIYI"


def test_custom_match_never_picks_a_gimbal(catalog: Path) -> None:
    assert match_payload(catalog, 0.1).name == "Micro EO 2-axis gimbal"
    assert match_payload(catalog, 0.5, "eo_ir").name == "Compact EO/IR 3-axis zoom"


def test_stale_catalog_is_reseeded(catalog: Path) -> None:
    with sqlite3.connect(catalog) as conn:
        conn.execute("PRAGMA user_version = 0")
    pipeline.ensure_catalog(catalog)
    assert catalog_version(catalog) == CATALOG_VERSION
    assert any(p.vendor == "SIYI" for p in load_catalog(catalog, "payloads"))


# ---------------------------------------------------------- requirement + parts list ---
G1 = next(iter(EXAMPLE_MISSIONS.values()))
ZT30 = "SIYI ZT30"


def _named(mass: float = 0.854, power: float = 9.0, model: str = ZT30) -> OperationalRequirements:
    return G1.model_copy(update={"payload_model": model, "payload_mass_kg": mass, "payload_power_w": power})


def _payload_needs(model: str = ZT30, mass: float = 0.854, power: Optional[float] = None) -> ComponentNeeds:
    return ComponentNeeds(cruise_power_w=100.0, battery_energy_j=1e6, battery_peak_power_w=500.0,
                          payload_mass_kg=mass, payload_power_w=power, payload_model=model)


def test_requirements_default_custom_and_old_json_loads() -> None:
    old = G1.model_dump(mode="json")
    old.pop("payload_model")
    assert OperationalRequirements.model_validate(old).payload_model == CUSTOM_PAYLOAD


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError, match="Foo 9000"):
        OperationalRequirements.model_validate({**G1.model_dump(), "payload_model": "Foo 9000"})


def test_payload_edited() -> None:
    assert _named().payload_edited is False
    assert _named(mass=0.9).payload_edited is True
    assert _named(power=12.0).payload_edited is True
    assert G1.payload_edited is False


def test_named_model_bom_line(catalog: Path) -> None:
    part = payload_part(catalog, _payload_needs(power=9.0))
    spec = payload_spec(ZT30)
    assert (part.name, part.mass_kg, part.power_w, part.vendor) == (ZT30, 0.854, 9.0, "SIYI")
    assert (part.length_m, part.width_m, part.height_m) == (spec.length_m, spec.width_m, spec.height_m)
    edited = payload_part(catalog, _payload_needs(mass=0.9, power=12.0))
    assert (edited.name, edited.mass_kg, edited.power_w) == ("SIYI ZT30 (edited)", 0.9, 12.0)


def test_custom_bom_line_matches_today(catalog: Path) -> None:
    part = payload_part(catalog, _payload_needs(model=CUSTOM_PAYLOAD, mass=1.0))
    assert part == match_payload(catalog, 1.0)  # Custom: today's lightest-generic-at-or-above match
    assert (part.name, part.mass_kg) == ("LiDAR mapping unit", 1.20)


def test_component_needs_copies_payload_model() -> None:
    req = _named()
    needs = component_needs(size_aircraft(req), req)
    assert (needs.payload_model, needs.payload_mass_kg, needs.payload_power_w) == (ZT30, 0.854, 9.0)
    custom = component_needs(size_aircraft(G1), G1)
    assert (custom.payload_model, custom.payload_power_w) == (CUSTOM_PAYLOAD, None)


def test_named_model_reaches_cad_payload_box(catalog: Path) -> None:
    pytest.importorskip("build123d")
    from src.tools.aero_cad import components as cad_components
    from tests.test_aero_cad import _fitted_layout

    req = _named()
    sizing = size_aircraft(req)
    bom = select_bom(catalog, component_needs(sizing, req))
    assert bom.payload.name == ZT30
    assert cad_components.payload(bom, _fitted_layout(sizing, bom)).mass_kg == 0.854


# ----------------------------------------------------- changes, explorer, review panel ---
def _payload_change(**requirements: object) -> ChangeSet:
    return ChangeSet(label="payload", requirements=RequirementChanges(**requirements),
                     assumptions=AssumptionChanges(), knobs=KnobChanges())


def test_model_change_fills_mass_and_power() -> None:
    change = _payload_change(payload_model=ZT30)
    new_req, _, _ = apply_changes(G1, DEFAULT_ASSUMPTIONS, change)
    assert (new_req.payload_model, new_req.payload_mass_kg, new_req.payload_power_w) == (ZT30, 0.854, 9.0)
    labels = [row[0] for row in change.describe(G1, DEFAULT_ASSUMPTIONS)]
    assert labels == ["Payload mass (kg)", "Payload power (W)", "Payload"]


def test_model_change_keeps_stated_mass() -> None:
    new_req, _, _ = apply_changes(G1, DEFAULT_ASSUMPTIONS, _payload_change(payload_model=ZT30, payload_mass_kg=1.0))
    assert (new_req.payload_mass_kg, new_req.payload_power_w, new_req.payload_edited) == (1.0, 9.0, True)


def test_switch_to_custom_keeps_numbers() -> None:
    new_req, _, _ = apply_changes(_named(), DEFAULT_ASSUMPTIONS, _payload_change(payload_model=CUSTOM_PAYLOAD))
    assert (new_req.payload_model, new_req.payload_mass_kg, new_req.payload_power_w) == (CUSTOM_PAYLOAD, 0.854, 9.0)


def test_unknown_model_change_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Foo 9000"):
        apply_changes(G1, DEFAULT_ASSUMPTIONS, _payload_change(payload_model="Foo 9000"))


def test_explorer_payload_lever_switches_named_model_to_custom() -> None:
    lever = explorer.LEVERS["payload_mass"]
    named = explorer._lever_change(lever, 0.5, _named()).requirements
    assert (named.payload_mass_kg, named.payload_model) == (0.5, CUSTOM_PAYLOAD)
    custom = explorer._lever_change(lever, 0.5, G1).requirements
    assert (custom.payload_mass_kg, custom.payload_model) == (0.5, None)
    endurance = explorer._lever_change(explorer.LEVERS["endurance"], 1.5, _named()).requirements
    assert endurance.payload_model is None


def test_review_panel_payload_model_choices() -> None:
    from src.ui import design_loop

    assert design_loop._field_choices("requirements", "payload_model") == payload_choices()
    assert design_loop._field_choices("requirements", "propulsion_type") == ("electric", "series_hybrid", "auto")
    assert design_loop._field_choices("requirements", "payload_mass_kg") is None


# ------------------------------------------------------------------------ display ---
def test_bom_rows_show_payload_vendor_price(catalog: Path) -> None:
    from src.ui.presenters import bom_rows

    req = _named()
    named = {r["Part"]: r for r in bom_rows(select_bom(catalog, component_needs(size_aircraft(req), req)))}
    assert (named["Payload"]["Item"], named["Payload"]["Ref. price"]) == (ZT30, "≈ $7,000")
    generic = {r["Part"]: r for r in bom_rows(select_bom(catalog, component_needs(size_aircraft(G1), G1)))}
    assert generic["Payload"]["Ref. price"] == "—"


def test_report_rows_show_payload_model() -> None:
    from src.agent.report import requirement_rows

    assert ("Payload", "SIYI ZT30 (edited)") in requirement_rows(_named(mass=0.9))
    assert ("Payload", "SIYI ZT30") in requirement_rows(_named())
    assert ("Payload", "Custom payload") in requirement_rows(G1)


def test_payload_summary() -> None:
    from src.ui.presenters import payload_summary

    assert payload_summary(_named()) == "SIYI ZT30 (0.854 kg, 9 W)"
    assert payload_summary(_named(mass=0.9)) == "SIYI ZT30 (edited) (0.9 kg, 9 W)"
    assert payload_summary(G1) == "Custom payload (1 kg, 15 W)"


# ------------------------------------------------------------- final-review fixes ---
def _form_format(field: str) -> str:
    from src.ui.presenters import NUMBER_FIELDS

    return next(fmt for name, *_, fmt in NUMBER_FIELDS["Payload"] if name == field)


@pytest.mark.parametrize("part", [p for p in PAYLOADS if p.vendor is not None], ids=lambda p: p.name)
def test_form_shows_catalogue_values_exactly(part) -> None:
    """The number boxes round to their format; a datasheet value must survive it or the unit reads as edited."""
    assert float(_form_format("payload_mass_kg") % part.mass_kg) == part.mass_kg
    assert float(_form_format("payload_power_w") % part.power_w) == part.power_w


@pytest.mark.parametrize("text, expected", [
    ("SIYI MT11", "SIYI UniPod MT11"), ("MT11", "SIYI UniPod MT11"), ("NV10", "SIYI UniPod NV10"),
    ("ZT30 gimbal", ZT30), ("siyi zt30 eo/ir", ZT30), ("NextVision Raptor camera", "NextVision Raptor"),
    ("gimbal", None),
])
def test_resolve_common_phrasings(text: str, expected: object) -> None:
    assert resolve_payload_name(text) == expected


def test_chat_offers_catalogue_names_and_resolves_loosely() -> None:
    from src.agent.agent_schemas import ProposedChange
    from src.agent.design_chat import _allowed_fields, to_change_set

    assert _allowed_fields(False)["requirements"]["payload_model"]["enum"] == list(payload_choices())
    proposed = ProposedChange.model_validate({"label": "lighter gimbal",
                                             "changes": [{"field": "payload_model", "value": "zt30"}]})
    assert to_change_set(proposed).requirements.payload_model == ZT30


def test_restating_the_current_model_keeps_edited_values() -> None:
    edited = _named(mass=1.0)
    new_req, _, _ = apply_changes(edited, DEFAULT_ASSUMPTIONS,
                                  _payload_change(payload_model=ZT30, cruise_speed_ms=25.0))
    assert (new_req.payload_mass_kg, new_req.payload_power_w) == (1.0, 9.0)


# -------------------------------------------------------------- display and review panel ---
def test_describe_names_custom_payload() -> None:
    to_custom = _payload_change(payload_model=CUSTOM_PAYLOAD).describe(_named(), DEFAULT_ASSUMPTIONS)
    assert ("Payload", ZT30, "Custom payload") in to_custom
    to_model = _payload_change(payload_model=ZT30).describe(G1, DEFAULT_ASSUMPTIONS)
    assert ("Payload", "Custom payload", ZT30) in to_model


def test_payload_display_name_is_the_one_edited_rule() -> None:
    from src.schemas.payload_catalog import EDITED_SUFFIX, payload_display_name
    from src.tools.database import matcher

    assert payload_display_name(ZT30, 0.854, 9.0) == ZT30
    assert payload_display_name(ZT30, 0.9, 9.0) == ZT30 + EDITED_SUFFIX
    assert payload_display_name(ZT30, 0.854, 12.0) == ZT30 + EDITED_SUFFIX
    assert payload_display_name(CUSTOM_PAYLOAD, 1.0, 5.0) == "Custom payload"
    assert not hasattr(matcher, "EDITED_SUFFIX")  # the catalogue owns the payload naming rule


def _review_state(monkeypatch: pytest.MonkeyPatch, **fields: object) -> dict[str, object]:
    from src.ui import design_loop

    state = {design_loop._review_key("requirements", name): value for name, value in fields.items()}
    monkeypatch.setattr(design_loop.st, "session_state", state)
    design_loop._fill_review_payload()
    return {name: state.get(design_loop._review_key("requirements", name)) for name in
            ("payload_model", "payload_mass_kg", "payload_power_w")}


def test_review_panel_model_pick_refills_mass_and_power(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _review_state(monkeypatch, payload_model="SIYI ZR10", payload_mass_kg=0.854, payload_power_w=9.0)
    assert state == {"payload_model": "SIYI ZR10", "payload_mass_kg": 0.381, "payload_power_w": 4.0}


def test_review_panel_refill_leaves_custom_and_absent_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    custom = _review_state(monkeypatch, payload_model=CUSTOM_PAYLOAD, payload_mass_kg=0.854)
    assert custom == {"payload_model": CUSTOM_PAYLOAD, "payload_mass_kg": 0.854, "payload_power_w": None}
    model_only = _review_state(monkeypatch, payload_model="SIYI ZR10")  # mass/power follow the model on apply
    assert model_only == {"payload_model": "SIYI ZR10", "payload_mass_kg": None, "payload_power_w": None}
