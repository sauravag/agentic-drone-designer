"""Task 6: matched propulsion efficiencies, masses and diameters drive the sizing loop (spec 6.5)."""

import math
from pathlib import Path
from typing import Optional

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.propulsion import OperatingPoint, PropulsionMatch
from src.schemas.state import PhaseResult, SizingOutput
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import catalog_rows, seed_database
from src.tools.propulsion.match_types import MatchContext, PropulsionMatchError
from src.tools.sizing import propulsion_loop as loop
from src.tools.sizing.aero_model import air_properties, speed_of_sound_ms
from src.tools.sizing.layouts import layout_spec
from src.tools.sizing.mass_model import Geometry, MatchedHardware, lift_unit_mass_kg, sizing_gear
from src.tools.sizing.mission_profile import (FORWARD_PHASES, SEED_FORWARD_EFFICIENCY, VTOL_THRUST_TO_WEIGHT,
                                              AeroState, PhaseSpec, PropulsionEfficiencies, seed_efficiencies,
                                              simulate)
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from src.units import G0
from tests.reference_designs import G1_ELECTRIC_FW

TRACTOR = layout_spec("pod_boom_tractor")
PUSHER = layout_spec("single_boom_vtail_pusher")


# ------------------------------------------------------------------ helpers


def _geometry() -> Geometry:
    span: float = math.sqrt(6.0)
    return Geometry(wing_area_m2=0.6, aspect_ratio=10.0, wingspan_m=span, mean_chord_m=0.6 / span,
                    fuselage_length_m=0.9, fuselage_diameter_m=0.13, horizontal_tail_area_m2=0.08,
                    vertical_tail_area_m2=0.04, tail_arm_m=0.7)


def _aero(disk_area_m2: float = 0.0) -> AeroState:
    return AeroState(_geometry(), cd0=0.03, k=0.04, rho_takeoff=1.225, rho_cruise=1.1, disk_area_m2=disk_area_m2)


def _plan(with_hover: bool = False) -> tuple[PhaseSpec, ...]:
    forward = (PhaseSpec("Climb", 200.0, False, 14.0, 1.15, 2.5, 250.0),
               PhaseSpec("Ingress", 1000.0, False, 18.0, 1.1, altitude_m=500.0),
               PhaseSpec("Loiter", 3600.0, False, 15.0, 1.1, altitude_m=500.0),
               PhaseSpec("Egress", 1000.0, False, 18.0, 1.1, altitude_m=500.0))
    if not with_hover:
        return forward
    return (PhaseSpec("VTOL_Takeoff", 60.0, True, 0.0, 1.225), *forward,
            PhaseSpec("VTOL_Land", 60.0, True, 0.0, 1.225))


def _efficiencies(**forward: float) -> PropulsionEfficiencies:
    return PropulsionEfficiencies({**{name: 0.6 for name in FORWARD_PHASES}, **forward}, 0.6, 0.62)


def _match(tag: str, mass_kg: float, series: int = 4) -> PropulsionMatch:
    op = OperatingPoint(flight_point="Loiter", propulsor="cruise", rpm=5000.0, current_a=8.0, throttle=0.6,
                        prop_efficiency=0.7, motor_efficiency=0.85, tip_mach=0.3, electrical_power_w=120.0,
                        sagged=False)
    return PropulsionMatch(cell_name="cell", series=series, cruise_motor=f"motor {tag}",
                           cruise_propeller=f"prop {tag}", cruise_esc="esc", operating_points=(op,),
                           efficiencies={name: 0.55 for name in FORWARD_PHASES}, propulsion_mass_kg=mass_kg)


class _FakeSizing:
    """Records every sizing pass; an 'evaluation' is the pass count so far."""

    def __init__(self) -> None:
        self.calls: list[Optional[PropulsionMatch]] = []

    def size_pass(self, efficiencies: PropulsionEfficiencies, match: Optional[PropulsionMatch]) -> int:
        self.calls = [*self.calls, match]
        return len(self.calls)


def _sequence_matcher(sequence: list[PropulsionMatch]):
    calls: list[int] = []

    def fake_match(context: object) -> PropulsionMatch:
        calls.append(1)
        return sequence[min(len(calls), len(sequence)) - 1]

    return fake_match, calls


def _phase_results(plan: tuple[PhaseSpec, ...], powers: dict[str, float]) -> tuple[PhaseResult, ...]:
    return tuple(PhaseResult(name=s.name, duration_s=s.duration_s, propulsive_power_w=powers.get(s.name, 300.0),
                             total_power_w=powers.get(s.name, 300.0) + 15.0, energy_j=1.0) for s in plan)


# ------------------------------------------------------------------ efficiencies and the simulator


def test_seed_efficiencies_are_the_former_assumptions() -> None:
    tractor = seed_efficiencies(TRACTOR, 0.95)
    pusher = seed_efficiencies(PUSHER, 0.95)

    assert set(tractor.forward) == set(FORWARD_PHASES)
    assert all(v == SEED_FORWARD_EFFICIENCY == 0.68 for v in tractor.forward.values())
    assert all(v == pytest.approx(0.68 * 0.95) for v in pusher.forward.values())
    assert tractor.hover == tractor.hover_peak == pytest.approx(0.72 * 0.85)


@pytest.mark.parametrize("forward, hover", [({"Climb": 0.0}, 0.6), ({"Loiter": 1.2}, 0.6), ({}, -0.1),
                                            ({"Egress": math.nan}, 0.6)])
def test_efficiencies_outside_zero_to_one_are_rejected(forward: dict, hover: float) -> None:
    with pytest.raises(ValueError):
        PropulsionEfficiencies({**{name: 0.6 for name in FORWARD_PHASES}, **forward}, hover, 0.6)


def test_efficiencies_need_every_forward_phase_and_are_read_only() -> None:
    with pytest.raises(ValueError, match="Egress"):
        PropulsionEfficiencies({"Climb": 0.6, "Ingress": 0.6, "Loiter": 0.6}, 0.6, 0.6)
    eff = _efficiencies()
    with pytest.raises(TypeError):
        eff.forward["Loiter"] = 0.9  # type: ignore[index]


def test_simulate_uses_per_phase_efficiency() -> None:
    base = simulate(_plan(), 4.0, _aero(), 15.0, 0.0, 0.0, _efficiencies(Loiter=0.6))
    halved = simulate(_plan(), 4.0, _aero(), 15.0, 0.0, 0.0, _efficiencies(Loiter=0.3))

    by_name = {p.name: p for p in base.phases}
    halved_by_name = {p.name: p for p in halved.phases}
    assert halved_by_name["Loiter"].propulsive_power_w == pytest.approx(2.0 * by_name["Loiter"].propulsive_power_w)
    for name in ("Climb", "Ingress", "Egress"):
        assert halved_by_name[name].propulsive_power_w == pytest.approx(by_name[name].propulsive_power_w)


def test_simulate_hover_power_is_ideal_power_over_hover_efficiency() -> None:
    disk: float = 4 * math.pi * 0.36**2 / 4
    trace = simulate(_plan(with_hover=True), 6.0, _aero(disk), 0.0, 0.0, 0.0, _efficiencies())

    weight: float = 6.0 * G0
    ideal: float = weight**1.5 / math.sqrt(2.0 * 1.225 * disk)
    assert trace.phases[0].propulsive_power_w == pytest.approx(ideal / 0.6)
    assert trace.peak_hover_w == pytest.approx(VTOL_THRUST_TO_WEIGHT**1.5 * ideal / 0.62)


def test_speed_of_sound_follows_the_isa_offset() -> None:
    assert speed_of_sound_ms(0.0) == pytest.approx(340.3, abs=0.1)
    assert speed_of_sound_ms(0.0, 20.0) == pytest.approx(340.3 * math.sqrt(308.15 / 288.15), rel=1e-3)
    assert speed_of_sound_ms(4000.0) < speed_of_sound_ms(0.0)


# ------------------------------------------------------------------ flight points and the match context


def test_forward_flight_points_carry_thrust_from_power_and_efficiency() -> None:
    plan = _plan()
    eff = _efficiencies(Climb=0.5, Loiter=0.65)
    points = loop.flight_points(plan, _phase_results(plan, {"Climb": 280.0, "Loiter": 90.0}), eff, 4.0, 0, 10.0)

    by_name = {p.name: p for p in points}
    assert tuple(by_name) == FORWARD_PHASES
    assert by_name["Climb"].thrust_n == pytest.approx(280.0 * 0.5 / 14.0)
    assert by_name["Loiter"].thrust_n == pytest.approx(90.0 * 0.65 / 15.0)
    assert by_name["Climb"].check_sagged and not by_name["Loiter"].check_sagged
    assert by_name["Loiter"].duration_s == 3600.0 and by_name["Loiter"].rho == 1.1
    assert by_name["Loiter"].mu == pytest.approx(air_properties(500.0, 10.0)[1])
    assert by_name["Loiter"].speed_of_sound_ms == pytest.approx(speed_of_sound_ms(500.0, 10.0))
    assert by_name["Climb"].speed_of_sound_ms == pytest.approx(speed_of_sound_ms(250.0, 10.0))


def test_hover_flight_points_are_per_rotor_at_one_and_peak_thrust_to_weight() -> None:
    plan = _plan(with_hover=True)
    points = loop.flight_points(plan, _phase_results(plan, {}), _efficiencies(), 6.0, 4, 0.0)

    by_name = {p.name: p for p in points}
    hover, peak = by_name[loop.HOVER_POINT], by_name[loop.HOVER_PEAK_POINT]
    assert hover.thrust_n == pytest.approx(6.0 * G0 / 4)
    assert peak.thrust_n == pytest.approx(VTOL_THRUST_TO_WEIGHT * 6.0 * G0 / 4)
    assert hover.is_hover and hover.airspeed_ms == 0.0 and hover.check_sagged and peak.check_sagged
    assert hover.duration_s == 120.0 and peak.duration_s == 0.0
    assert hover.rho == 1.225


def test_context_builder_maps_the_design_rulings() -> None:
    points = loop.flight_points(_plan(), _phase_results(_plan(), {}), _efficiencies(), 4.0, 0, 0.0)
    context = loop.match_context(points, is_hybrid=True, lift_rotor_count=0, rotor_diameter_m=0.0,
                                 fuselage_diameter_m=0.12, is_hand_launch=True, spec=PUSHER, hotel_w=20.0,
                                 assumptions=DEFAULT_ASSUMPTIONS)
    assert isinstance(context, MatchContext)
    assert context.chemistry == "lipo" and context.is_hybrid and context.require_folding_prop
    assert context.max_cruise_prop_diameter_m == pytest.approx(loop.PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO * 0.12)
    assert loop.PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO == 3.0
    assert context.pusher_factor == pytest.approx(0.95)

    tractor = loop.match_context(points, is_hybrid=False, lift_rotor_count=0, rotor_diameter_m=0.0,
                                 fuselage_diameter_m=0.12, is_hand_launch=False, spec=TRACTOR, hotel_w=20.0,
                                 assumptions=DEFAULT_ASSUMPTIONS)
    assert tractor.chemistry is None and tractor.pusher_factor == 1.0 and not tractor.require_folding_prop


def test_efficiencies_from_a_match_fill_hover_from_the_seed_for_fixed_wing() -> None:
    seed = seed_efficiencies(TRACTOR, 0.95)
    eff = loop.efficiencies_from(_match("A", 0.2), seed)
    assert dict(eff.forward) == {name: 0.55 for name in FORWARD_PHASES}
    assert (eff.hover, eff.hover_peak) == (seed.hover, seed.hover_peak)


# ------------------------------------------------------------------ matched masses and diameters


def test_matched_hardware_reads_part_masses_and_diameters_from_the_catalogue() -> None:
    motor = catalog_rows("cruise_motors")[0]
    prop = next(p for p in catalog_rows("propellers") if p.application == "cruise")
    lift_motor = next(m for m in catalog_rows("vtol_motors") if m.kv is not None)
    rotor = next(p for p in catalog_rows("propellers") if p.application == "vtol")
    esc, lift_esc = catalog_rows("escs")[:2]
    match = _match("A", 1.0).model_copy(update={
        "cruise_motor": motor.name, "cruise_propeller": prop.name, "cruise_esc": esc.name,
        "lift_motor": lift_motor.name, "lift_rotor": rotor.name, "lift_esc": lift_esc.name})

    hardware = loop.matched_hardware(match, 4)

    assert hardware.cruise_kg == pytest.approx(motor.mass_kg + prop.mass_kg + esc.mass_kg)
    assert hardware.vtol_kg == pytest.approx(4 * lift_unit_mass_kg(lift_motor.mass_kg, rotor.mass_kg,
                                                                   lift_esc.mass_kg))
    assert hardware.cruise_propeller_diameter_m == prop.diameter_m
    assert hardware.lift_rotor_diameter_m == rotor.diameter_m


def test_matched_hardware_names_an_unknown_part() -> None:
    with pytest.raises(PropulsionMatchError, match="prop A"):
        loop.matched_hardware(_match("A", 1.0), 0)


def test_lift_unit_mass_keeps_the_mount_allowance_as_a_floor() -> None:
    assert lift_unit_mass_kg(1.0, 0.1, 0.05) == pytest.approx(1.3)  # mount allowance governs
    assert lift_unit_mass_kg(1.0, 0.3, 0.1) == pytest.approx(1.4)  # real parts exceed it


def test_matched_propeller_diameter_sets_the_sizing_gear() -> None:
    geom = _geometry()
    estimated = sizing_gear(geom, TRACTOR, "runway")
    big = sizing_gear(geom, TRACTOR, "runway", propeller_diameter_m=0.6)
    assert big.leg_height_m > estimated.leg_height_m


# ------------------------------------------------------------------ the outer loop (fake matching)


def test_loop_stops_when_the_set_repeats() -> None:
    sizing = _FakeSizing()
    b, a, a_again = _match("B", 0.2), _match("A", 0.3), _match("A", 0.3)
    fake, calls = _sequence_matcher([b, a, a_again])

    result = loop.size_with_matching(sizing.size_pass, lambda ev: ev, seed_efficiencies(TRACTOR, 0.95), fake)

    assert result.match == a and result.evaluation == 3 and result.passes == 2
    assert sizing.calls == [None, b, a] and len(calls) == 3


def test_cycle_takes_heavier_set() -> None:
    sizing = _FakeSizing()
    light, heavy = _match("A", 0.2), _match("B", 0.5)
    fake, _ = _sequence_matcher([light, heavy, light, heavy])

    result = loop.size_with_matching(sizing.size_pass, lambda ev: ev, seed_efficiencies(TRACTOR, 0.95), fake)

    assert result.match.set_key == heavy.set_key
    assert sizing.calls[-1] is not None and sizing.calls[-1].set_key == heavy.set_key  # re-sized once with it
    assert result.evaluation == len(sizing.calls) == 4
    assert any("cycled" in note and "motor B" in note for note in result.match.notes)


def test_no_convergence_raises() -> None:
    sizing = _FakeSizing()
    fake, calls = _sequence_matcher([_match(str(i), 0.2 + i / 100) for i in range(20)])

    with pytest.raises(PropulsionMatchError, match=f"{loop.MAX_PROPULSION_PASSES} passes"):
        loop.size_with_matching(sizing.size_pass, lambda ev: ev, seed_efficiencies(TRACTOR, 0.95), fake)
    assert len(calls) == loop.MAX_PROPULSION_PASSES == 6


def test_matching_failure_propagates_as_a_sizing_failure() -> None:
    def failing(context: object) -> PropulsionMatch:
        raise PropulsionMatchError("no propulsion set meets every matching rule")

    with pytest.raises(PropulsionMatchError, match="no propulsion set"):
        loop.size_with_matching(_FakeSizing().size_pass, lambda ev: ev, seed_efficiencies(TRACTOR, 0.95), failing)


# ------------------------------------------------------------------ the G1 reference through the real loop


@pytest.fixture(scope="module")
def g1() -> SizingOutput:
    return size_aircraft(G1_ELECTRIC_FW)


def test_outer_loop_converges_for_g1_reference(g1: SizingOutput) -> None:
    assert g1.propulsion_match is not None
    again = size_aircraft(G1_ELECTRIC_FW)
    assert again.propulsion_match is not None and again.propulsion_match.set_key == g1.propulsion_match.set_key
    assert again.mtow_kg == pytest.approx(g1.mtow_kg, rel=1e-9)


def test_phase_powers_use_the_matched_efficiencies(g1: SizingOutput) -> None:
    match = g1.propulsion_match
    assert match is not None
    for phase in g1.phases:
        thrust_power_w: float = phase.propulsive_power_w * match.efficiencies[phase.name]
        assert thrust_power_w < phase.propulsive_power_w  # an efficiency below one costs power
        assert 0.3 < match.efficiencies[phase.name] < 0.9


def test_propulsion_mass_comes_from_matched_set(g1: SizingOutput) -> None:
    match = g1.propulsion_match
    assert match is not None
    hardware: MatchedHardware = loop.matched_hardware(match, 0)
    assert g1.empty_mass_breakdown["cruise_propulsion"] == pytest.approx(hardware.cruise_kg)
    assert hardware.cruise_kg == pytest.approx(match.propulsion_mass_kg)
    assert g1.empty_mass_breakdown["vtol_system"] == 0.0


def test_context_limits_the_cruise_diameter_by_the_fuselage(g1: SizingOutput) -> None:
    match = g1.propulsion_match
    assert match is not None
    prop = next(p for p in catalog_rows("propellers") if p.name == match.cruise_propeller)
    assert prop.diameter_m <= loop.PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO * g1.fuselage_diameter_m + 1e-9


def test_component_needs_carry_the_matched_names(g1: SizingOutput) -> None:
    match = g1.propulsion_match
    assert match is not None
    needs = component_needs(g1, G1_ELECTRIC_FW)
    assert (needs.cruise_motor_name, needs.cruise_propeller_name, needs.cruise_esc_name) == (
        match.cruise_motor, match.cruise_propeller, match.cruise_esc)
    assert (needs.cell_name, needs.battery_series) == (match.cell_name, match.series)


def test_select_bom_returns_the_matched_parts(g1: SizingOutput, tmp_path: Path) -> None:
    match = g1.propulsion_match
    assert match is not None
    bom = select_bom(seed_database(tmp_path / "c.db"), component_needs(g1, G1_ELECTRIC_FW))

    assert (bom.cruise_motor.name, bom.cruise_propeller.name) == (match.cruise_motor, match.cruise_propeller)
    assert bom.cruise_esc is not None and bom.cruise_esc.name == match.cruise_esc
    assert (bom.battery.cell_name, bom.battery.series) == (match.cell_name, match.series)
    assert bom.battery.capacity_j >= g1.battery_energy_j
    assert bom.vtol_motor is None and bom.vtol_esc is None
