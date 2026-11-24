"""Propulsion matching search (spec 6.3-6.4): pick S, motor winding, propeller and ESC per propulsor."""

import math
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Final, Optional

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.components import Cell, CruiseMotor, Esc, Propeller, VtolMotor
from src.schemas.propulsion import OperatingPoint, PropulsionMatch
from src.tools.database.packs import charged_voltage_v
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion import matching
from src.tools.propulsion.matching import (
    ESC_CURRENT_MARGIN, MAX_MATCH_THROTTLE, PROP_TIP_MACH_MAX, FlightPoint, MatchContext, PropulsionMatchError,
    match_propulsion,
)
from src.tools.propulsion.prop_tables import propeller_table
from src.tools.sizing.errors import SizingDivergenceError

RHO, MU, SOUND = 1.225, 1.81e-5, 340.3
G1_MAX_CRUISE_D_M: Final[float] = 0.26  # 8 and 10 inch props only: keeps the table builds few
G1_LIFT_MIN_D_M: Final[float] = 0.36  # -> 15 inch lift rotors
FORWARD_PHASES: Final[tuple[str, ...]] = ("Climb", "Ingress", "Loiter", "Egress")
G1_MATCH_SECONDS_MAX: Final[float] = 5.0


def _forward(name: str, airspeed_ms: float, thrust_n: float, duration_s: float, sagged: bool = False) -> FlightPoint:
    return FlightPoint(name=name, is_hover=False, airspeed_ms=airspeed_ms, rho=RHO, mu=MU, speed_of_sound_ms=SOUND,
                       thrust_n=thrust_n, duration_s=duration_s, check_sagged=sagged)


def _hover(name: str, thrust_n: float, duration_s: float) -> FlightPoint:
    return FlightPoint(name=name, is_hover=True, airspeed_ms=0.0, rho=RHO, mu=MU, speed_of_sound_ms=SOUND,
                       thrust_n=thrust_n, duration_s=duration_s, check_sagged=True)


# G1-like fixed wing (~4 kg MTOW, W = 39.2 N, sea level): drag from L/D ~10-12, plus W sin(gamma) in the 2.5 m/s climb.
G1_POINTS: Final[tuple[FlightPoint, ...]] = (
    _forward("Climb", 16.0, 3.9 + 39.2 * 2.5 / 16.0, 180.0, sagged=True),
    _forward("Ingress", 18.0, 3.6, 1200.0),
    _forward("Loiter", 14.0, 3.3, 3600.0),
    _forward("Egress", 18.0, 3.6, 1200.0),
)
# G1 quadplane (~5 kg, 4 lift rotors): hover at weight, hover peak at the VTOL thrust-to-weight.
QUAD_HOVER: Final[tuple[FlightPoint, ...]] = (_hover("Hover", 49.0 / 4, 90.0), _hover("Hover_Peak", 1.3 * 49.0 / 4, 10.0))


def _context(points: tuple[FlightPoint, ...] = G1_POINTS, **changes: object) -> MatchContext:
    fields: dict[str, object] = dict(
        chemistry=None, is_hybrid=False, lift_rotor_count=0, min_lift_rotor_diameter_m=0.0,
        max_cruise_prop_diameter_m=G1_MAX_CRUISE_D_M, require_folding_prop=False, pusher_factor=1.0, hotel_w=15.0,
        assumptions=DEFAULT_ASSUMPTIONS)
    return MatchContext(points=points, **{**fields, **changes})  # type: ignore[arg-type]


def _quad_context(**changes: object) -> MatchContext:
    quad: dict[str, object] = dict(lift_rotor_count=4, min_lift_rotor_diameter_m=G1_LIFT_MIN_D_M)
    return _context(points=G1_POINTS + QUAD_HOVER, **{**quad, **changes})


@pytest.fixture(scope="module", autouse=True)
def shared_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One temp table cache for the module: each propeller table is built once across the tests."""
    cache: Path = tmp_path_factory.mktemp("prop_tables")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROP_TABLE_CACHE_DIR", str(cache))
        propeller_table.cache_clear()
        yield cache
    propeller_table.cache_clear()


@pytest.fixture(scope="module")
def g1_match() -> PropulsionMatch:
    return match_propulsion(_context())


@pytest.fixture(scope="module")
def quad_match() -> PropulsionMatch:
    return match_propulsion(_quad_context())


def _row(table: str, name: Optional[str]) -> object:
    return next(r for r in catalog_rows(table) if r.name == name)


def _cell(match: PropulsionMatch) -> Cell:
    return _row("cells", match.cell_name)  # type: ignore[return-value]


def test_error_types_keep_their_old_import_path() -> None:
    from src.tools.sizing.mission_solver import SizingDivergenceError as legacy

    assert legacy is SizingDivergenceError
    assert issubclass(PropulsionMatchError, SizingDivergenceError)


def test_constants_carry_the_global_constraint_values() -> None:
    assert (PROP_TIP_MACH_MAX, MAX_MATCH_THROTTLE, ESC_CURRENT_MARGIN) == (0.70, 0.85, 1.2)


def test_g1_cruise_point_picks_consistent_voltage(g1_match: PropulsionMatch) -> None:
    motor: CruiseMotor = _row("cruise_motors", g1_match.cruise_motor)  # type: ignore[assignment]
    esc: Esc = _row("escs", g1_match.cruise_esc)  # type: ignore[assignment]
    charged: float = charged_voltage_v(_cell(g1_match), g1_match.series)
    assert motor.voltage_v >= charged - 1e-9
    assert esc.max_voltage_v >= charged - 1e-9


def _assert_meets_every_rule(match: PropulsionMatch, context: MatchContext) -> None:
    ops: tuple[OperatingPoint, ...] = match.operating_points
    for op in ops:
        assert op.tip_mach <= PROP_TIP_MACH_MAX
        assert op.throttle <= (1.0 if op.sagged else MAX_MATCH_THROTTLE)
    sagged_names = {op.flight_point for op in ops if op.sagged}
    assert sagged_names == {p.name for p in context.points if p.check_sagged}
    roles = (("cruise", match.cruise_motor, match.cruise_esc, "cruise_motors"),
             ("lift", match.lift_motor, match.lift_esc, "vtol_motors"))
    for role, motor_name, esc_name, table in roles:
        role_ops = [op for op in ops if op.propulsor == role]
        if not role_ops:
            continue
        motor: CruiseMotor | VtolMotor = _row(table, motor_name)  # type: ignore[assignment]
        esc: Esc = _row("escs", esc_name)  # type: ignore[assignment]
        peak_a: float = max(op.current_a for op in role_ops)
        assert peak_a <= motor.max_current_a
        assert esc.continuous_current_a >= ESC_CURRENT_MARGIN * peak_a
    prop: Propeller = _row("propellers", match.cruise_propeller)  # type: ignore[assignment]
    assert prop.diameter_m <= context.max_cruise_prop_diameter_m


def test_chosen_set_meets_every_rule(g1_match: PropulsionMatch, quad_match: PropulsionMatch) -> None:
    _assert_meets_every_rule(g1_match, _context())
    _assert_meets_every_rule(quad_match, _quad_context())
    climb = [op for op in g1_match.operating_points if op.flight_point == "Climb"]
    assert sorted(op.sagged for op in climb) == [False, True]


def test_more_thrust_never_picks_a_lighter_set(g1_match: PropulsionMatch) -> None:
    heavier = tuple(replace(p, thrust_n=1.4 * p.thrust_n) for p in G1_POINTS)
    assert match_propulsion(_context(points=heavier)).propulsion_mass_kg >= g1_match.propulsion_mass_kg


def test_thrust_beyond_catalogue_names_the_constraint() -> None:
    huge = tuple(replace(p, thrust_n=2_000.0) for p in G1_POINTS)
    with pytest.raises(PropulsionMatchError, match="current|power|tip Mach|throttle|diameter"):
        match_propulsion(_context(points=huge))


def test_failure_names_the_flight_point_and_the_diameter_scope() -> None:
    heavy = tuple(replace(p, thrust_n=4.0 * p.thrust_n) for p in G1_POINTS)
    with pytest.raises(PropulsionMatchError) as failure:
        match_propulsion(_context(points=heavy))
    message: str = str(failure.value)
    assert any(f"at {name}" for name in FORWARD_PHASES if f"at {name}" in message)
    assert f"{G1_MAX_CRUISE_D_M:.3f} m" in message


def test_lift_rotor_beyond_catalogue_names_diameter() -> None:
    with pytest.raises(PropulsionMatchError, match="diameter"):
        match_propulsion(_quad_context(min_lift_rotor_diameter_m=5.0))


def test_cruise_diameter_below_catalogue_names_diameter() -> None:
    with pytest.raises(PropulsionMatchError, match="diameter"):
        match_propulsion(_context(max_cruise_prop_diameter_m=0.1))


def test_quadplane_lift_rotor_not_smaller_than_disk_loading_diameter(quad_match: PropulsionMatch) -> None:
    rotor: Propeller = _row("propellers", quad_match.lift_rotor)  # type: ignore[assignment]
    lift_diameters = sorted({p.diameter_m for p in catalog_rows("propellers") if p.application == "vtol"})
    assert rotor.application == "vtol"
    assert rotor.diameter_m >= G1_LIFT_MIN_D_M
    assert rotor.diameter_m == min(d for d in lift_diameters if d >= G1_LIFT_MIN_D_M)
    assert quad_match.lift_motor is not None and quad_match.lift_esc is not None


def test_hand_launch_requires_folding_prop() -> None:
    match = match_propulsion(_context(require_folding_prop=True))
    prop: Propeller = _row("propellers", match.cruise_propeller)  # type: ignore[assignment]
    assert prop.is_folding


def test_hybrid_without_generator_in_bus_window_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    real = matching.catalog_rows

    def no_variants(table: str) -> list:
        rows = real(table)
        return [g for g in rows if g.output_voltage_v is None] if table == "ice_generators" else rows

    monkeypatch.setattr(matching, "catalog_rows", no_variants)
    with pytest.raises(PropulsionMatchError, match="bus window"):
        match_propulsion(_context(is_hybrid=True, chemistry="lipo"))


def test_hybrid_series_count_has_a_generator_in_its_bus_window() -> None:
    match = match_propulsion(_context(is_hybrid=True, chemistry="lipo"))
    cell: Cell = _cell(match)
    low, high = 3.5 * match.series, charged_voltage_v(cell, match.series)
    assert any(g.output_voltage_v is not None and low <= g.output_voltage_v <= high
               for g in catalog_rows("ice_generators"))


def test_efficiencies_cover_every_forward_phase_and_hover(g1_match: PropulsionMatch,
                                                          quad_match: PropulsionMatch) -> None:
    assert set(g1_match.efficiencies) == set(FORWARD_PHASES)
    assert set(quad_match.efficiencies) == set(FORWARD_PHASES) | {"Hover", "Hover_Peak"}
    assert all(0.0 < eta < 1.0 for eta in quad_match.efficiencies.values())


def test_forward_efficiency_is_thrust_power_over_electrical_power(g1_match: PropulsionMatch) -> None:
    op = next(o for o in g1_match.operating_points if o.flight_point == "Loiter")
    point = next(p for p in G1_POINTS if p.name == "Loiter")
    assert g1_match.efficiencies["Loiter"] == pytest.approx(point.thrust_n * point.airspeed_ms / op.electrical_power_w)


def test_set_key_is_the_spec_tuple(quad_match: PropulsionMatch) -> None:
    assert quad_match.set_key == (quad_match.series, quad_match.cruise_motor, quad_match.cruise_propeller,
                                  quad_match.lift_motor, quad_match.lift_rotor)


def test_pusher_installation_loss_costs_power(g1_match: PropulsionMatch) -> None:
    pusher = match_propulsion(_context(pusher_factor=0.9))
    assert pusher.efficiencies["Loiter"] < g1_match.efficiencies["Loiter"]


def test_context_rejects_hover_points_without_lift_rotors() -> None:
    with pytest.raises(ValueError, match="lift"):
        _context(points=G1_POINTS + QUAD_HOVER)


def test_flight_point_rejects_non_positive_thrust() -> None:
    with pytest.raises(ValueError, match="thrust_n"):
        _forward("Cruise", 18.0, 0.0, 60.0)


def test_g1_match_is_fast_with_a_warm_cache(g1_match: PropulsionMatch) -> None:
    start: float = time.perf_counter()
    match_propulsion(_context())
    assert time.perf_counter() - start < G1_MATCH_SECONDS_MAX
    assert math.isfinite(g1_match.propulsion_mass_kg)
