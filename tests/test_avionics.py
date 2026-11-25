"""Avionics stack choice (Cube Orange+ / ARK Jetson PAB), the optional Jetson, and the avionics assumptions."""

import json
from pathlib import Path
from typing import Final

import pytest

from src.tools.sizing.mission_solver import size_aircraft
from src.ui.examples import G1_ELECTRIC_FW, G2_ELECTRIC_VTOL, G3_HYBRID_VTOL

BASELINE_PATH: Final[Path] = Path(__file__).parent / "data" / "avionics_baseline.json"
BASELINE_FIELDS: Final[tuple[str, ...]] = (
    "mtow_kg", "empty_mass_kg", "battery_mass_kg", "fuel_mass_kg", "cruise_power_w", "hover_power_w",
)
EXACT: Final[float] = 1e-9
BASELINE_MISSIONS: Final[tuple] = (G1_ELECTRIC_FW, G2_ELECTRIC_VTOL, G3_HYBRID_VTOL)


# Recorded from the code before the avionics change (re-recorded at the 2026-11-23 boom re-baseline and the
# 2026-11-25 matched-propulsion re-baseline, scripts/regenerate_avionics_baseline.py): Cube Orange+ without a
# Jetson at default assumptions must size every reference design exactly as before.
# (The baseline holds the designs that existed then; later examples have no "before".)
@pytest.mark.parametrize("req", BASELINE_MISSIONS, ids=lambda r: r.mission_name)
def test_reference_designs_unchanged_without_jetson(req) -> None:
    expected: dict[str, float] = json.loads(BASELINE_PATH.read_text())[req.mission_name]
    sizing = size_aircraft(req)
    for field in BASELINE_FIELDS:
        assert getattr(sizing, field) == pytest.approx(expected[field], rel=EXACT, abs=EXACT), field


# ------------------------------------------------------------------ schemas ---
from pydantic import ValidationError  # noqa: E402

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions  # noqa: E402
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes  # noqa: E402
from src.schemas.requirements import OperationalRequirements  # noqa: E402
from src.ui.examples import G1_ELECTRIC_FW  # noqa: E402

LEGACY_REQUIREMENTS: Final[Path] = Path(__file__).parent / "data" / "flight_control" / "g1" / "requirements.json"
MTOW_KG: Final[float] = 10.0


def _req(**update: object) -> OperationalRequirements:
    """Build through the constructor so validators run (model_copy skips them)."""
    return OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), **update})


def test_old_requirements_json_loads_as_cube_without_jetson() -> None:
    req = OperationalRequirements.model_validate(json.loads(LEGACY_REQUIREMENTS.read_text()))
    assert req.flight_controller == "cube_orange_plus"
    assert req.companion_computer is False and req.jetson_fitted is False


def test_ark_normalises_companion_and_fits_jetson() -> None:
    req = _req(flight_controller="ark_jetson_pab", companion_computer=True)
    assert req.companion_computer is False and req.jetson_fitted is True


def test_cube_with_companion_fits_jetson() -> None:
    assert _req(companion_computer=True).jetson_fitted is True


def test_apply_changes_switching_to_ark_normalises() -> None:
    change = ChangeSet(label="ark", requirements=RequirementChanges(flight_controller="ark_jetson_pab"),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    req, _, _ = apply_changes(_req(companion_computer=True), DEFAULT_ASSUMPTIONS, change)
    assert req.flight_controller == "ark_jetson_pab" and req.companion_computer is False


@pytest.mark.parametrize(("flight_controller", "companion", "expected"), [
    ("cube_orange_plus", False, 0.3 + 0.1),
    ("cube_orange_plus", True, 0.3 + 0.1 + 0.2),
    ("ark_jetson_pab", False, 0.227 + 0.1 + 0.166),
])
def test_avionics_mass_formula(flight_controller: str, companion: bool, expected: float) -> None:
    assert DEFAULT_ASSUMPTIONS.avionics_mass_kg(MTOW_KG, flight_controller, companion) == pytest.approx(expected)


@pytest.mark.parametrize(("fitted", "expected"), [(False, 10.0 + 4.0), (True, 10.0 + 4.0 + 25.0)])
def test_avionics_power_formula(fitted: bool, expected: float) -> None:
    assert DEFAULT_ASSUMPTIONS.avionics_power_w(MTOW_KG, fitted) == pytest.approx(expected)


@pytest.mark.parametrize(("field", "below", "above"), [
    ("avionics_base_mass_kg", -0.01, 5.01), ("avionics_mass_per_mtow", -0.01, 0.051),
    ("avionics_base_power_w", 0.99, 200.1), ("avionics_power_per_kg_w", -0.01, 5.01),
    ("cube_orange_plus_mass_kg", 0.029, 0.51), ("ark_pab_stack_mass_kg", 0.079, 1.01),
    ("companion_computer_mass_kg", 0.019, 3.01), ("jetson_power_w", 0.99, 100.1),
])
def test_avionics_assumption_bounds(field: str, below: float, above: float) -> None:
    for value in (below, above):
        with pytest.raises(ValidationError):
            DesignAssumptions(**{field: value})


def test_change_sets_carry_avionics_fields() -> None:
    from src.schemas.changes import FIELD_LABELS
    for name in ("flight_controller", "companion_computer", "jetson_power_w", "cube_orange_plus_mass_kg"):
        assert name in FIELD_LABELS
    assert set(AssumptionChanges.model_fields) >= {"jetson_power_w", "avionics_base_mass_kg"}


# ------------------------------------------------------------------- sizing ---
from src.tools.sizing.layouts import layout_spec  # noqa: E402
from src.tools.sizing.mass_model import PropulsionLoads, empty_mass_breakdown, size_geometry, sizing_gear  # noqa: E402
from src.tools.sizing.mission_profile import hotel_power_w  # noqa: E402

JETSON_W: Final[float] = 25.0


def test_hotel_power_adds_jetson_watts() -> None:
    cube, cube_jetson = _req(), _req(companion_computer=True)
    extra = hotel_power_w(cube_jetson, MTOW_KG, DEFAULT_ASSUMPTIONS) - hotel_power_w(cube, MTOW_KG, DEFAULT_ASSUMPTIONS)
    assert extra == pytest.approx(JETSON_W)


def test_breakdown_avionics_entry_is_the_argument() -> None:
    geom = size_geometry(MTOW_KG, 1.0, 10.0, 0.5, 0.04, 1.0, layout_spec("pod_boom_tractor"))
    gear = sizing_gear(geom, layout_spec("pod_boom_tractor"), "runway")
    parts = empty_mass_breakdown(MTOW_KG, geom, "fixed_wing", PropulsionLoads(500.0), avionics_mass_kg=0.5,
                                 booms=(), gear=gear)
    assert parts["avionics"] == 0.5


def test_sizing_records_avionics_and_grows_with_jetson() -> None:
    cube = size_aircraft(_req())
    cube_jetson = size_aircraft(_req(companion_computer=True))
    ark = size_aircraft(_req(flight_controller="ark_jetson_pab"))
    assert cube_jetson.mtow_kg > ark.mtow_kg > cube.mtow_kg
    for sizing, req in ((cube, _req()), (cube_jetson, _req(companion_computer=True))):
        expected = DEFAULT_ASSUMPTIONS.avionics_mass_kg(sizing.mtow_kg, req.flight_controller, req.companion_computer)
        assert sizing.avionics_mass_kg == pytest.approx(expected)
        assert sizing.avionics_power_w == pytest.approx(DEFAULT_ASSUMPTIONS.avionics_power_w(sizing.mtow_kg,
                                                                                            req.jetson_fitted))
    assert ark.avionics_power_w - DEFAULT_ASSUMPTIONS.avionics_power_w(ark.mtow_kg, False) == pytest.approx(JETSON_W)


# ---------------------------------------------------------------- parts list ---
from src.schemas.components import CompanionComputer, ComponentNeeds, FlightController  # noqa: E402
from src.tools.database.matcher import avionics_parts  # noqa: E402
from src.tools.sizing.mission_solver import component_needs  # noqa: E402
from src.ui.presenters import bom_rows  # noqa: E402

CUBE_KG, ARK_KG, COMPANION_KG = 0.073, 0.166, 0.2


def _needs(flight_controller: str = "cube_orange_plus", companion: bool = False) -> ComponentNeeds:
    return ComponentNeeds(cruise_power_w=100.0, battery_energy_j=1e6, battery_peak_power_w=500.0, payload_mass_kg=1.0,
                          flight_controller=flight_controller, companion_computer=companion)


def test_cube_bom_has_flight_controller_only() -> None:
    fc, companion = avionics_parts(_needs(), DEFAULT_ASSUMPTIONS)
    assert "Cube Orange+" in fc.name and fc.mass_kg == CUBE_KG and fc.power_w is None and companion is None


def test_cube_with_jetson_adds_companion() -> None:
    _, companion = avionics_parts(_needs(companion=True), DEFAULT_ASSUMPTIONS)
    assert companion is not None and companion.mass_kg == COMPANION_KG and companion.power_w == JETSON_W


def test_ark_bom_has_stack_and_no_companion() -> None:
    fc, companion = avionics_parts(_needs("ark_jetson_pab", companion=True), DEFAULT_ASSUMPTIONS)
    assert "ARK" in fc.name and fc.mass_kg == ARK_KG and fc.power_w == JETSON_W and companion is None


def test_avionics_lines_follow_edited_assumptions() -> None:
    edited = DEFAULT_ASSUMPTIONS.model_copy(update={"jetson_power_w": 15.0, "cube_orange_plus_mass_kg": 0.08})
    fc, companion = avionics_parts(_needs(companion=True), edited)
    assert fc.mass_kg == 0.08 and companion is not None and companion.power_w == 15.0


def _bom(db_path: Path, req: OperationalRequirements):
    from src.tools.database.matcher import select_bom
    return select_bom(db_path, component_needs(size_aircraft(req), req), DEFAULT_ASSUMPTIONS)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.tools.database.seed_data import seed_database
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    seed_database(path)
    return path


def test_total_hardware_mass_counts_avionics_lines(catalog: Path) -> None:
    bom = _bom(catalog, _req(companion_computer=True))
    without = bom.model_copy(update={"flight_controller": None, "companion_computer": None})
    assert bom.total_hardware_mass_kg == pytest.approx(without.total_hardware_mass_kg + CUBE_KG + COMPANION_KG)


def test_bom_rows_show_avionics_rows(catalog: Path) -> None:
    rows = {r["Part"]: r for r in bom_rows(_bom(catalog, _req(companion_computer=True)))}
    assert rows["Flight controller"]["Power (W)"] == "in avionics allowance"
    assert rows["Companion computer"]["Power (W)"] == f"{JETSON_W:g}"  # one column type: no Arrow mixed-type warning
    assert all(r["Power (W)"] is None or isinstance(r["Power (W)"], str) for r in rows.values())
    assert rows["Flight controller"]["Ref. price"].startswith("≈ $")
    assert rows["Total"]["Mass (kg)"] == pytest.approx(round(_bom(catalog, _req(companion_computer=True))
                                                            .total_hardware_mass_kg, 3))


def test_component_needs_copies_avionics_choice() -> None:
    req = _req(flight_controller="ark_jetson_pab")
    needs = component_needs(size_aircraft(req), req)
    assert needs.flight_controller == "ark_jetson_pab" and needs.companion_computer is False
    assert FlightController and CompanionComputer  # exported for the BOM schema


# ------------------------------------------------- final-review fixes (RED first) ---
def _change(**requirements: object) -> ChangeSet:
    return ChangeSet(label="avionics", requirements=RequirementChanges(**requirements),
                     assumptions=AssumptionChanges(), knobs=KnobChanges())


def test_switching_ark_to_cube_keeps_the_jetson() -> None:
    ark = _req(flight_controller="ark_jetson_pab")
    req, _, _ = apply_changes(ark, DEFAULT_ASSUMPTIONS, _change(flight_controller="cube_orange_plus"))
    assert req.flight_controller == "cube_orange_plus" and req.companion_computer is True
    rows = _change(flight_controller="cube_orange_plus").describe(ark, DEFAULT_ASSUMPTIONS)
    assert ("Jetson companion computer", "no", "yes") in rows  # the implied change is shown, not hidden


def test_switching_ark_to_cube_can_drop_the_jetson_explicitly() -> None:
    ark = _req(flight_controller="ark_jetson_pab")
    change = _change(flight_controller="cube_orange_plus", companion_computer=False)
    req, _, _ = apply_changes(ark, DEFAULT_ASSUMPTIONS, change)
    assert req.companion_computer is False


def test_companion_on_ark_is_shown_as_no_change() -> None:
    ark = _req(flight_controller="ark_jetson_pab")
    assert _change(companion_computer=True).describe(ark, DEFAULT_ASSUMPTIONS) == []


def test_report_rows_say_ark_has_a_jetson() -> None:
    from src.agent import report
    rows = dict(report.requirement_rows(_req(flight_controller="ark_jetson_pab")))
    assert rows["Flight controller"] == "ARK Jetson PAB (Orin NX 16 GB)"
    assert rows["Jetson companion computer"] == "built into the ARK stack"
    cube_rows = dict(report.requirement_rows(_req(companion_computer=True)))
    assert cube_rows["Flight controller"] == "Cube Orange+" and cube_rows["Jetson companion computer"] == "yes"


def test_report_facts_and_chat_context_state_jetson_fitted() -> None:
    from src.agent import report
    from src.agent.design_chat import _design_data
    from types import SimpleNamespace
    ark = _req(flight_controller="ark_jetson_pab")
    facts = json.loads(report.facts(report.ReportInputs(run_id="r", out_dir=Path("."), requirements=ark, sizing=None,
                                                        bom=None, aero=None, cad=None, fcs=None, converged=True,
                                                        iterations=0)))
    assert facts["requirements"]["jetson_fitted"] is True
    assert facts["avionics"] == "ARK Jetson PAB + Orin NX 16 GB (25 W, 0.166 kg)"
    ctx = SimpleNamespace(requirements=ark, assumptions=DEFAULT_ASSUMPTIONS, failed_gates=(), critic_summary=(),
                          options=(), allow_knobs=False)
    assert _design_data(ctx)["requirements"]["jetson_fitted"] is True
