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
from src.tools.database.matcher import generator_fits
from src.tools.database.packs import bus_window_v, charged_voltage_v
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion import match_search, matching
from src.tools.propulsion.matching import (
    ESC_CURRENT_MARGIN, MAX_MATCH_THROTTLE, PROP_TIP_MACH_MAX, FlightPoint, MatchContext, PropulsionMatchError,
    match_propulsion,
)
from src.tools.propulsion.match_search import ESC_BURST_DURATION_S
from src.tools.propulsion.prop_tables import propeller_table
from src.tools.sizing.errors import SizingDivergenceError
from src.tools.sizing.mission_profile import GENERATOR_MARGIN

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
QUAD_HOVER: Final[tuple[FlightPoint, ...]] = (_hover("Hover", 49.0 / 4, 90.0),
                                              _hover("Hover_Peak", 1.3 * 49.0 / 4, 10.0))


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
        durations = {p.name: p.duration_s for p in context.points}
        sustained = [op.current_a for op in role_ops if durations[op.flight_point] > ESC_BURST_DURATION_S]
        transient = [op.current_a for op in role_ops if durations[op.flight_point] <= ESC_BURST_DURATION_S]
        assert esc.continuous_current_a >= ESC_CURRENT_MARGIN * max(sustained, default=0.0)
        assert esc.burst_current_a >= max(transient, default=0.0)
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
    # the limiting rule after the scope's colon (the scope text itself always mentions "diameter")
    limit = r"diameter\): (current|power|tip Mach|throttle|sagged throttle)\b[^;|]* at (Climb|Ingress|Loiter|Egress) \("
    with pytest.raises(PropulsionMatchError, match=limit):
        match_propulsion(_context(points=huge))


def test_failure_names_the_flight_point_and_the_diameter_scope() -> None:
    heavy = tuple(replace(p, thrust_n=4.0 * p.thrust_n) for p in G1_POINTS)
    with pytest.raises(PropulsionMatchError) as failure:
        match_propulsion(_context(points=heavy))
    message: str = str(failure.value)
    assert any(f"at {name}" in message for name in FORWARD_PHASES)
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
    # ruling L2: any catalogue size from the disk-loading diameter up to 1.25 x it (the objective chooses)
    assert rotor.diameter_m <= max(1.25 * G1_LIFT_MIN_D_M, min(d for d in lift_diameters if d >= G1_LIFT_MIN_D_M))
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


def _patched_rows(monkeypatch: pytest.MonkeyPatch, table: str, edit) -> None:  # type: ignore[no-untyped-def]
    real = matching.catalog_rows
    monkeypatch.setattr(matching, "catalog_rows", lambda name: edit(real(name)) if name == table else real(name))


def test_hybrid_generator_power_rule_limits_the_cruise_propulsor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_rows(monkeypatch, "ice_generators",
                  lambda rows: [g.model_copy(update={"continuous_power_w": 20.0}) for g in rows])
    context = _context(is_hybrid=True, chemistry="lipo")
    buses = matching._buses(context, matching._voltage_classes(context))
    cruise = matching._role(context, False, buses)
    assert not cruise.best  # 20 W generators cannot carry ~100 W of cruise plus the 15 W hotel load
    assert any(r.rule == "generator power" and r.point in FORWARD_PHASES for r in cruise.rejections)
    with pytest.raises(PropulsionMatchError):
        match_propulsion(context)


def test_no_common_bus_voltage_is_named() -> None:
    bus = ("lipo", 3.7, 4.2)
    cruise = matching._RoleResult(best={(*bus, 4): object()}, rejections=())  # type: ignore[dict-item]
    lift = matching._RoleResult(best={(*bus, 12): object()}, rejections=())  # type: ignore[dict-item]
    error = matching._failure(_quad_context(), cruise, lift)
    assert "no common bus voltage" in str(error) and "S=[4]" in str(error) and "S=[12]" in str(error)


def test_unbuildable_propeller_is_skipped_loudly(monkeypatch: pytest.MonkeyPatch, g1_match: PropulsionMatch) -> None:
    chosen: Propeller = _row("propellers", g1_match.cruise_propeller)  # type: ignore[assignment]
    monkeypatch.setattr(match_search, "_table_failure",
                        lambda d, p: "boom" if (d, p) == (chosen.diameter_m, chosen.pitch_m) else None)
    match = match_propulsion(_context())
    assert match.cruise_propeller != chosen.name
    assert any("skipped" in note and chosen.name in note for note in match.notes)


def test_every_propeller_unbuildable_names_the_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(match_search, "_table_failure", lambda d, p: "boom")
    with pytest.raises(PropulsionMatchError, match="propeller model failed .no table. for Carbon cruise prop"):
        match_propulsion(_context())


def test_match_has_no_skip_note_when_every_table_builds(g1_match: PropulsionMatch) -> None:
    assert not any("skipped" in note for note in g1_match.notes)


def test_thrust_far_below_the_catalogue_names_the_oversize_prune() -> None:
    tiny = tuple(replace(p, thrust_n=0.01) for p in G1_POINTS)
    with pytest.raises(PropulsionMatchError, match=r"6x ideal power \(oversize prune\)"):
        match_propulsion(_context(points=tiny))


def test_folding_props_excluded_when_not_required(g1_match: PropulsionMatch) -> None:
    candidates = matching._cruise_propellers(_context())
    assert candidates and not any(p.is_folding for p in candidates)
    assert not _row("propellers", g1_match.cruise_propeller).is_folding  # type: ignore[attr-defined]


def test_efficiencies_are_read_only_and_round_trip(g1_match: PropulsionMatch) -> None:
    with pytest.raises(TypeError):
        g1_match.efficiencies["Loiter"] = 1.0  # type: ignore[index]
    again = PropulsionMatch.model_validate_json(g1_match.model_dump_json())
    assert again == g1_match and dict(again.efficiencies) == dict(g1_match.efficiencies)


def test_missing_cells_message_without_a_chemistry(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_rows(monkeypatch, "cells", lambda rows: [])
    with pytest.raises(PropulsionMatchError, match="no cells in the catalogue"):
        match_propulsion(_context())


def test_table_failures_are_one_limit_entry_and_never_crowd_out_physical_limits() -> None:
    from src.tools.propulsion.match_search import RULE_MODEL, Rejection

    model = tuple(Rejection(f"{RULE_MODEL}: Prop {i}", "") for i in range(10))
    physical = (Rejection("current", "Climb"),) * 3 + (Rejection("tip Mach", "Climb"),) * 2 + (
        Rejection("throttle", "Loiter"),)
    text: str = matching._limits(model + physical)
    assert all(rule in text for rule in ("current at Climb", "tip Mach at Climb", "throttle at Loiter"))
    assert text.count(RULE_MODEL) == 1 and "Prop 0" in text and "Prop 9" in text


def test_lift_rotor_candidates_span_the_disk_loading_minimum_to_the_oversize_limit() -> None:
    from src.tools.propulsion import matching as mt

    assert mt.LIFT_ROTOR_MAX_OVERSIZE == 1.25
    context = _quad_context(min_lift_rotor_diameter_m=1.0)
    rotors = mt._lift_rotors(context)
    diameters = {p.diameter_m for p in rotors}
    assert diameters and all(1.0 - 1e-9 <= d <= 1.25 + 1e-9 for d in diameters)
    catalogue = {p.diameter_m for p in catalog_rows("propellers") if p.application == "vtol"}
    assert diameters == {d for d in catalogue if 1.0 - 1e-9 <= d <= 1.25 + 1e-9}
    pitches = {p.pitch_m for p in rotors if p.diameter_m == min(diameters)}
    assert len(pitches) >= 2  # every pitch of each diameter


def test_lift_rotor_candidates_keep_the_next_size_up_across_a_catalogue_gap() -> None:
    from src.tools.propulsion import matching as mt

    sizes = sorted({p.diameter_m for p in catalog_rows("propellers") if p.application == "vtol"})
    gap_low, gap_high = max(zip(sizes, sizes[1:]), key=lambda pair: pair[1] / pair[0])
    minimum = gap_low * 1.0001  # just past a size: the next one up is more than 1.25 x away only across a gap
    rotors = mt._lift_rotors(_quad_context(min_lift_rotor_diameter_m=minimum))
    assert {p.diameter_m for p in rotors} == ({gap_high} if gap_high > 1.25 * minimum else
                                              {d for d in sizes if minimum <= d <= 1.25 * minimum})


# ------------------------------------------------------------------ fix round 2: hybrid energy pricing, ESC burst


def test_energy_above_the_generator_share_is_priced_at_the_buffer_battery() -> None:
    from src.tools.propulsion.match_search import priced_storage_kg

    fuel, battery = 1e-7, 2e-6  # kg/J: battery ~20x fuel, as for LiPo vs BSFC x fuel system
    cruise_only = priced_storage_kg((800.0,), (1000.0,), 1000.0, fuel, battery)
    assert cruise_only == pytest.approx(800.0 * 1000.0 * fuel)
    climb = priced_storage_kg((1500.0,), (100.0,), 1000.0, fuel, battery)
    assert climb == pytest.approx(1000.0 * 100.0 * fuel + 500.0 * 100.0 * battery)
    assert priced_storage_kg((1500.0,), (100.0,), math.inf, fuel, battery) == pytest.approx(1500.0 * 100.0 * fuel)


def test_a_low_efficiency_climb_costs_buffer_battery_in_the_hybrid_objective() -> None:
    from src.tools.propulsion.match_search import priced_storage_kg

    fuel, battery, share = 1e-7, 2e-6, 1000.0
    efficient = priced_storage_kg((1100.0,), (300.0,), share, fuel, battery)
    wasteful = priced_storage_kg((1400.0,), (300.0,), share, fuel, battery)  # same thrust power at lower efficiency
    fuel_priced_gap = (1400.0 - 1100.0) * 300.0 * fuel
    assert wasteful - efficient == pytest.approx((1400.0 - 1100.0) * 300.0 * battery)
    assert wasteful - efficient > 10 * fuel_priced_gap


def test_hybrid_match_uses_the_sized_generator_rating() -> None:
    hybrid = dict(is_hybrid=True, chemistry="lipo")
    roomy = match_propulsion(_context(**hybrid, generator_w=5_000.0))
    tight = match_propulsion(_context(**hybrid, generator_w=200.0))  # climb now runs on the buffer
    roomy_kg = float(roomy.notes[0].split("estimated ")[1].split(" kg")[0].split()[-1])
    tight_kg = float(tight.notes[0].split("estimated ")[1].split(" kg")[0].split()[-1])
    assert tight_kg > roomy_kg


def test_esc_burst_rating_covers_a_transient_peak() -> None:
    from src.tools.propulsion.match_search import lightest_esc

    escs = tuple(sorted(catalog_rows("escs"), key=lambda e: (e.mass_kg, e.name)))
    esc = lightest_esc(escs, 10.0, 30.0, 16.8)
    assert esc is not None and esc.continuous_current_a >= ESC_CURRENT_MARGIN * 10.0
    assert esc.burst_current_a >= 30.0
    strict = lightest_esc(escs, 30.0, 0.0, 16.8)  # the same 30 A sustained needs the continuous margin
    assert strict is not None and strict.continuous_current_a >= ESC_CURRENT_MARGIN * 30.0
    assert esc.mass_kg <= strict.mass_kg


def test_only_short_points_count_as_transient() -> None:
    from src.tools.propulsion.match_search import ESC_BURST_DURATION_S, is_transient

    assert is_transient(_hover("Hover_Peak", 10.0, 0.0))
    assert is_transient(_hover("Hover_Peak", 10.0, ESC_BURST_DURATION_S))
    assert not is_transient(_hover("Hover", 10.0, 120.0))


def test_a_light_quadplane_still_gets_its_disk_loading_rotor_despite_a_high_rating() -> None:
    # G2 8 h hybrid regression: ~9.3 kg, 22.8 N hover / 33 N peak per rotor, rotors at least 0.471 m. The only lift
    # diameters allowed (22 in, rated 1900 W) are rated above 6x the ideal hover power; the disk-loading rule, not
    # the rating, sets their size, so they must not be pruned as oversized.
    hover = (_hover("Hover", 22.8, 120.0), _hover("Hover_Peak", 33.1, 0.0))
    context = _context(points=G1_POINTS + hover, lift_rotor_count=4, min_lift_rotor_diameter_m=0.471)
    match = match_propulsion(context)
    rotor: Propeller = _row("propellers", match.lift_rotor)  # type: ignore[assignment]
    assert rotor.diameter_m >= 0.471


def test_air_off_the_density_axis_is_its_own_rule_not_tip_mach() -> None:
    thin = tuple(replace(p, rho=0.25) for p in G1_POINTS)  # reference mu and a: equivalent density = rho
    with pytest.raises(PropulsionMatchError) as failure:
        match_propulsion(_context(points=thin))
    message: str = str(failure.value)
    assert "air outside the propeller tables (equivalent density 0.25 kg/m³ not in [" in message
    assert match_search.RULE_TABLE not in message


def _bom_generator_need_w(match: PropulsionMatch, hotel_w: float) -> float:
    """What sizing hands the BOM: GENERATOR_MARGIN x (the highest sustained forward electrical power + hotel); the
    sustained points are the ones the generator carries (not the check_sagged climb, which the buffer helps)."""
    sustained_names = {p.name for p in G1_POINTS if not p.check_sagged}
    sustained = [p.electrical_power_w for p in match.operating_points
                 if p.propulsor == "cruise" and p.flight_point in sustained_names]
    return GENERATOR_MARGIN * (max(sustained) + hotel_w)


def test_a_matched_hybrid_bus_always_has_a_genset_the_bom_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The matcher's generator cap uses the BOM's margins: a set that passes matching cannot fail select_bom."""
    context = _context(is_hybrid=True, chemistry="lipo")
    free: PropulsionMatch = match_propulsion(context)
    # gensets just above the free match's cruise + hotel: enough without the margins, too small with them
    rating_w: float = 1.1 * (_bom_generator_need_w(free, context.hotel_w) / GENERATOR_MARGIN)
    _patched_rows(monkeypatch, "ice_generators",
                  lambda rows: [g.model_copy(update={"continuous_power_w": rating_w}) for g in rows])
    try:
        tight: PropulsionMatch = match_propulsion(context)
    except PropulsionMatchError:
        return  # no set fits these gensets with the margins: refused at matching, not at select_bom
    window = bus_window_v(_cell(tight), tight.series)
    need_w: float = _bom_generator_need_w(tight, context.hotel_w)
    assert any(generator_fits(g, need_w, window) for g in matching.catalog_rows("ice_generators"))
