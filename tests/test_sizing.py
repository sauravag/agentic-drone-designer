"""Milestone 2 acceptance tests: DoD group/topology rules and the time-stepped mission solver."""

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import SizingOutput
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import seed_database
from src.tools.sizing import mission_solver
from src.tools.sizing.aero_model import air_properties, airfoil_polar, reynolds_number, select_airfoil
from src.units import G0
from src.tools.sizing.mission_solver import (
    SizingDivergenceError,
    component_needs,
    reserve_fraction,
    size_aircraft,
)
from src.tools.sizing.topology import classify_group, select_airframe, select_propulsion
from tests.reference_designs import CASES, G1_ELECTRIC_FW, G2_ELECTRIC_VTOL, G3_HYBRID_VTOL, REQ_FAILED

GROUP1_MAX_KG: float = 9.07
GROUP2_MAX_KG: float = 24.95
GROUP3_MAX_KG: float = 599.0



@pytest.fixture(scope="module")
def g1() -> SizingOutput:
    return size_aircraft(G1_ELECTRIC_FW)


@pytest.fixture(scope="module")
def g2() -> SizingOutput:
    return size_aircraft(G2_ELECTRIC_VTOL)


@pytest.fixture(scope="module")
def g3() -> SizingOutput:
    return size_aircraft(G3_HYBRID_VTOL)


# -------------------------------------------------------------- topology ---


@pytest.mark.parametrize(
    ("mtow", "group"),
    [(2.0, 1), (9.069, 1), (9.07, 2), (24.94, 2), (24.95, 3), (598.9, 3)],
)
def test_classify_group_boundaries(mtow: float, group: int) -> None:
    assert classify_group(mtow) == group


@pytest.mark.parametrize("mtow", [599.0, 0.0, -1.0])
def test_classify_group_rejects_out_of_range(mtow: float) -> None:
    with pytest.raises(ValueError):
        classify_group(mtow)


@pytest.mark.parametrize(("runway", "expected"), [(False, "quadplane_vtol"), (True, "fixed_wing")])
def test_auto_airframe_follows_runway(runway: bool, expected: str) -> None:
    req = G1_ELECTRIC_FW.model_copy(update={"runway_available": runway})
    assert select_airframe(req) == expected


def test_explicit_airframe_is_respected() -> None:
    req = G1_ELECTRIC_FW.model_copy(update={"airframe_topology": "quadplane_vtol"})
    assert select_airframe(req) == "quadplane_vtol"


@pytest.mark.parametrize(
    ("mtow", "hours", "expected"),
    [(10.0, 3.0, "electric"), (15.0, 2.0, "series_hybrid"), (10.0, 3.1, "series_hybrid")],
)
def test_auto_propulsion_rule(mtow: float, hours: float, expected: str) -> None:
    req = G1_ELECTRIC_FW.model_copy(update={"endurance_hours": hours})
    assert select_propulsion(req, mtow) == expected


@pytest.mark.parametrize(
    ("reynolds", "airfoil"),
    [(1.5e5, "sd7037"), (3e5, "mh32"), (1e6, "mh32"), (1.01e6, "naca4415")],
)
def test_reynolds_matched_airfoil(reynolds: float, airfoil: str) -> None:
    assert select_airfoil(reynolds) == airfoil


# ------------------------------------------------- reference cases (M2) ---


def test_group1_electric_fixed_wing(g1: SizingOutput) -> None:
    assert g1.mtow_kg < GROUP1_MAX_KG
    assert (g1.dod_group, g1.propulsion_type, g1.airframe_topology) == (1, "electric", "fixed_wing")
    assert g1.fuel_mass_kg == 0.0 and g1.hover_power_w == 0.0


def test_group2_electric_quadplane(g2: SizingOutput) -> None:
    assert GROUP1_MAX_KG <= g2.mtow_kg < GROUP2_MAX_KG
    assert (g2.dod_group, g2.propulsion_type, g2.airframe_topology) == (2, "electric", "quadplane_vtol")
    assert g2.hover_power_w > g2.cruise_power_w  # hover is the peak electrical load


def test_group3_hybrid_quadplane(g3: SizingOutput) -> None:
    assert GROUP2_MAX_KG <= g3.mtow_kg < GROUP3_MAX_KG
    assert (g3.dod_group, g3.propulsion_type, g3.airframe_topology) == (3, "series_hybrid", "quadplane_vtol")
    assert g3.fuel_mass_kg > 0.0 and g3.battery_mass_kg > 0.0 and g3.generator_power_w > 0.0


# ----------------------------------------------------- physics invariants ---


@pytest.mark.parametrize("case", list(CASES))
def test_stall_constraint_and_geometry(case: str, request: pytest.FixtureRequest) -> None:
    s: SizingOutput = request.getfixturevalue(case)
    assert s.v_stall_ms <= CASES[case].max_stall_speed_ms + 1e-6
    assert s.wingspan_m == pytest.approx((s.wing_area_m2 * s.aspect_ratio) ** 0.5, rel=1e-6)
    assert s.mean_chord_m == pytest.approx(s.wing_area_m2 / s.wingspan_m, rel=1e-6)
    assert s.v_loiter_ms > s.v_stall_ms
    assert 8.0 < s.assumed_ld_cruise < 30.0  # plausible small-UAV L/D band


@pytest.mark.parametrize("case", list(CASES))
def test_stall_speed_holds_with_independent_low_re_polar(case: str, request: pytest.FixtureRequest) -> None:
    """Re-derive CLmax at the sea-level stall Reynolds number (not cruise Re) and check Vstall."""
    s: SizingOutput = request.getfixturevalue(case)
    req = CASES[case]
    rho, mu = air_properties(0.0)
    re_stall: float = reynolds_number(rho, req.max_stall_speed_ms, s.mean_chord_m, mu)
    cl_max: float = airfoil_polar(s.airfoil, float(f"{re_stall:.2g}")).cl_max_3d
    v_stall: float = math.sqrt(2.0 * s.mtow_kg * G0 / (rho * s.wing_area_m2 * cl_max))
    assert v_stall <= req.max_stall_speed_ms * 1.02  # 2%: Re rounding to 2 significant figures


def test_wing_grows_when_cruise_at_altitude_would_stall() -> None:
    """Slow cruise at a high ceiling: wing must keep cruise >= 1.2 x local stall speed."""
    req = G1_ELECTRIC_FW.model_copy(update={"service_ceiling_m": 6000.0, "cruise_speed_ms": 15.0})
    s = size_aircraft(req)
    rho_c, _ = air_properties(req.service_ceiling_m)
    cl_cruise: float = 2.0 * s.mtow_kg * G0 / (rho_c * req.cruise_speed_ms**2 * s.wing_area_m2)
    assert cl_cruise <= s.cl_max / 1.2**2 * 1.05  # cruise polar CLmax >= stall polar CLmax
    assert s.v_stall_ms < req.max_stall_speed_ms  # the cruise constraint, not stall, sized the wing


def test_airfoil_flip_flop_falls_back_to_conservative_polar(monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    flip = iter(["mh32", "sd7037"] * 10)
    monkeypatch.setattr(mission_solver, "select_airfoil", lambda _re: next(flip))
    with caplog.at_level(logging.WARNING, logger=mission_solver.__name__):
        s = size_aircraft(G1_ELECTRIC_FW)
    assert "did not settle" in caplog.text
    lowest = min(airfoil_polar(n, 2e5).cl_max_3d for n in ("mh32", "sd7037"))
    assert s.cl_max <= lowest * 1.1


def test_electric_fallback_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    req = G2_ELECTRIC_VTOL.model_copy(update={"propulsion_type": "auto", "endurance_hours": 30.0})
    with caplog.at_level(logging.INFO, logger=mission_solver.__name__):
        try:
            size_aircraft(req)
        except SizingDivergenceError as exc:
            assert isinstance(exc.__cause__, SizingDivergenceError)  # electric failure is chained
    assert "electric sizing failed" in caplog.text


@pytest.mark.parametrize("case", list(CASES))
def test_mass_closure(case: str, request: pytest.FixtureRequest) -> None:
    s: SizingOutput = request.getfixturevalue(case)
    total: float = s.empty_mass_kg + s.payload_mass_kg + s.battery_mass_kg + s.fuel_mass_kg
    assert total == pytest.approx(s.mtow_kg, rel=1e-3)


def test_electric_reserve_is_twenty_percent(g2: SizingOutput) -> None:
    used: float = sum(p.energy_j for p in g2.phases)
    assert g2.energy_reserve_fraction == pytest.approx(0.20)
    assert g2.battery_energy_j * (1 - 0.20) >= used * 0.999


def test_quadplane_has_six_phases_fixed_wing_four(g1: SizingOutput, g2: SizingOutput) -> None:
    assert [p.name for p in g2.phases] == ["VTOL_Takeoff", "Climb", "Ingress", "Loiter", "Egress", "VTOL_Land"]
    assert [p.name for p in g1.phases] == ["Climb", "Ingress", "Loiter", "Egress"]


def test_hybrid_fuel_burn_lightens_aircraft(g3: SizingOutput) -> None:
    ingress, egress = g3.phases[2], g3.phases[4]
    assert ingress.fuel_burned_kg > 0
    assert egress.propulsive_power_w < ingress.propulsive_power_w  # same speed, lighter aircraft


def test_hybrid_generator_covers_cruise_with_margin(g3: SizingOutput) -> None:
    # Generator is sized on the first (heaviest) cruise step of Ingress.
    assert g3.generator_power_w >= 1.15 * g3.phases[2].total_power_w
    assert g3.generator_power_w == pytest.approx(1.15 * g3.phases[2].total_power_w, rel=0.03)


def test_auto_propulsion_switches_to_hybrid_for_long_endurance() -> None:
    req = G2_ELECTRIC_VTOL.model_copy(update={"propulsion_type": "auto", "endurance_hours": 8.0})
    assert size_aircraft(req).propulsion_type == "series_hybrid"


def test_solver_is_deterministic() -> None:
    assert size_aircraft(G1_ELECTRIC_FW) == size_aircraft(G1_ELECTRIC_FW)


def test_higher_aspect_ratio_override_improves_ld(g1: SizingOutput) -> None:
    hi_ar = size_aircraft(G1_ELECTRIC_FW, {"aspect_ratio": 16.0})
    assert hi_ar.aspect_ratio == 16.0
    assert hi_ar.assumed_ld_cruise > g1.assumed_ld_cruise


def test_unknown_override_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sizing override"):
        size_aircraft(G1_ELECTRIC_FW, {"wing_magic": 2.0})


@pytest.mark.parametrize("bad", [0.0, -3.0, float("nan")])
def test_non_physical_override_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="aspect_ratio"):
        size_aircraft(G1_ELECTRIC_FW, {"aspect_ratio": bad})


def test_infeasible_mission_raises_divergence() -> None:
    req = G1_ELECTRIC_FW.model_copy(
        update={"payload_mass_kg": 150.0, "endurance_hours": 40.0, "propulsion_type": "electric"}
    )
    with pytest.raises(SizingDivergenceError):
        size_aircraft(req)


# ------------------------------------------------ hand-off to the matcher ---


@pytest.mark.parametrize("case", list(CASES))
def test_catalog_can_supply_every_reference_design(case: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    s: SizingOutput = request.getfixturevalue(case)
    db: Path = seed_database(tmp_path / "catalog.db")
    bom = select_bom(db, component_needs(s, CASES[case]))
    assert (bom.vtol_motor is not None) == (s.airframe_topology == "quadplane_vtol")
    assert (bom.ice_generator is not None) == (s.propulsion_type == "series_hybrid")


def test_reserve_fraction_override_resizes_the_energy_store() -> None:
    """Critic knob (M5): a larger reserve means more fuel (hybrid) or battery (electric), and is reported."""
    from tests.reference_designs import G1_ELECTRIC_FW, G3_HYBRID_VTOL

    for req in (G1_ELECTRIC_FW, G3_HYBRID_VTOL):
        base, tight = size_aircraft(req), size_aircraft(req, {"reserve_fraction": 0.30})
        assert tight.energy_reserve_fraction == 0.30 and tight.mtow_kg > base.mtow_kg
        store = (lambda s: s.fuel_mass_kg) if base.propulsion_type == "series_hybrid" else (lambda s: s.battery_mass_kg)
        assert store(tight) > store(base)
    with pytest.raises(ValueError, match="reserve_fraction"):
        size_aircraft(G1_ELECTRIC_FW, {"reserve_fraction": 0.9})


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_assumptions_reproduce_the_reference_designs(name: str) -> None:
    req = CASES[name]
    assert size_aircraft(req).mtow_kg == size_aircraft(req, None, DEFAULT_ASSUMPTIONS).mtow_kg


def test_c_rate_drives_electric_vtol_battery_mass() -> None:
    req = REQ_FAILED.model_copy(update={"endurance_hours": 1.0, "combat_radius_km": 20.0, "cruise_speed_ms": 22.0})
    base = size_aircraft(req)
    fast = size_aircraft(req, None, DesignAssumptions(liion_max_c_rate=5))
    assert base.mtow_kg == pytest.approx(46.7, rel=0.02)
    assert fast.mtow_kg == pytest.approx(15.9, rel=0.05)


def test_specific_energy_and_efficiency_change_the_design() -> None:
    req = REQ_FAILED.model_copy(update={"endurance_hours": 3.0, "combat_radius_km": 50.0, "cruise_speed_ms": 22.0})
    better = size_aircraft(req, None, DesignAssumptions(liion_specific_energy_wh_kg=300))
    assert better.mtow_kg == pytest.approx(23.1, rel=0.03)
    with pytest.raises(SizingDivergenceError):
        size_aircraft(req)  # 200 Wh/kg spirals


def test_reserve_comes_from_assumptions_unless_overridden() -> None:
    a = DesignAssumptions(battery_reserve_fraction=0.30)
    assert reserve_fraction({}, "electric", a) == 0.30
    assert reserve_fraction({"reserve_fraction": 0.25}, "electric", a) == 0.25
    assert reserve_fraction({}, "series_hybrid", a) == 0.10


def test_slow_or_oscillating_fixed_point_now_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    req = REQ_FAILED.model_copy(update={"endurance_hours": 1.0, "combat_radius_km": 20.0})  # 30 m/s: oscillated
    calls: list[tuple[float, float]] = []
    real_evaluate = mission_solver._evaluate

    def spy(mtow: float, design: object) -> object:
        ev = real_evaluate(mtow, design)  # type: ignore[arg-type]
        calls.append((mtow, ev.mtow_out))
        return ev

    monkeypatch.setattr(mission_solver, "_evaluate", spy)
    sizing = size_aircraft(req)
    assert 10.0 < sizing.mtow_kg < 150.0
    mtow_in, mtow_out = calls[-1]  # the returned evaluation is the last one computed
    assert mtow_out == pytest.approx(sizing.mtow_kg)
    assert abs(mtow_out - mtow_in) <= 1e-6 * mtow_in


def test_genuine_spirals_still_raise() -> None:
    with pytest.raises(SizingDivergenceError, match="mass spiral"):
        size_aircraft(REQ_FAILED)


# --- synthetic maps exercising the damping and root-find fallback in isolation ---

def _fake_design() -> SimpleNamespace:
    return SimpleNamespace(req=SimpleNamespace(payload_mass_kg=1.0), propulsion="electric", airframe="fixed_wing")


def _patch_map(monkeypatch: pytest.MonkeyPatch, f: Callable[[float], float]) -> None:
    monkeypatch.setattr(mission_solver, "_evaluate",
                        lambda mtow, design: SimpleNamespace(mtow_in=mtow, mtow_out=f(mtow)))


def test_is_oscillating_requires_a_full_alternating_window() -> None:
    window = mission_solver.OSCILLATION_WINDOW
    alternating = [1 if i % 2 == 0 else -1 for i in range(window)]
    assert mission_solver._is_oscillating(alternating)
    assert not mission_solver._is_oscillating(alternating[:-1])  # short window
    assert not mission_solver._is_oscillating([*alternating[:-1], alternating[-2]])  # one repeat
    assert not mission_solver._is_oscillating([1] * window)


def test_overshooting_map_halves_damping_then_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    # f' = -4: the damping-0.5 iteration has slope -1.5 (alternating, diverging); 0.25 gives slope -0.25.
    _patch_map(monkeypatch, lambda m: 50.0 - 4.0 * m)
    real = mission_solver._is_oscillating
    detections: list[bool] = []
    monkeypatch.setattr(mission_solver, "_is_oscillating", lambda s: detections.append(real(s)) or detections[-1])
    monkeypatch.setattr(mission_solver, "_root_find", lambda *a: pytest.fail("fallback must not run"))
    ev = mission_solver._converge(_fake_design())  # type: ignore[arg-type]
    assert any(detections)
    assert ev.mtow_out == pytest.approx(10.0, rel=1e-6)


def _spy_root_find(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    ran: list[bool] = []
    real = mission_solver._root_find
    monkeypatch.setattr(mission_solver, "_root_find", lambda *a: ran.append(True) or real(*a))
    return ran


def test_slow_contraction_uses_upward_bracket_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_map(monkeypatch, lambda m: m + 0.001 * (10.0 - m))  # fixed point 10 above the start, f' = 0.999
    ran = _spy_root_find(monkeypatch)
    ev = mission_solver._converge(_fake_design())  # type: ignore[arg-type]
    assert ran
    assert ev.mtow_out == pytest.approx(10.0, rel=1e-6)


def test_slow_contraction_uses_downward_bracket_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_map(monkeypatch, lambda m: m - 0.001 * (m - 3.0))  # fixed point 3 below the start
    ran = _spy_root_find(monkeypatch)
    ev = mission_solver._converge(_fake_design())  # type: ignore[arg-type]
    assert ran
    assert ev.mtow_out == pytest.approx(3.0, rel=1e-6)


def test_rootless_growing_map_raises_mass_spiral(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_map(monkeypatch, lambda m: m + 1.0)
    with pytest.raises(SizingDivergenceError, match="mass spiral"):
        mission_solver._converge(_fake_design())  # type: ignore[arg-type]


def test_rootless_shrinking_map_raises_mass_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_map(monkeypatch, lambda m: m * 0.999)
    with pytest.raises(SizingDivergenceError, match="mass collapse"):
        mission_solver._converge(_fake_design())  # type: ignore[arg-type]


def test_sizing_output_carries_the_requested_cruise_speed() -> None:
    assert size_aircraft(CASES["g1"]).v_cruise_ms == CASES["g1"].cruise_speed_ms


def test_vne_ratio_is_shared_from_the_aero_model() -> None:
    from src.tools.flight_control import flight_model
    from src.tools.sizing.aero_model import VNE_TO_CRUISE

    assert VNE_TO_CRUISE == 1.4 and flight_model.VNE_TO_CRUISE is VNE_TO_CRUISE
