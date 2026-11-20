"""Task 8b: structural checks — centreline overwrap (R24/R29), spar torsion (R25, q from V_A per R30), tail-boom
strength (R26), socket bearing bosses (R27) and the crutch rail minimum (R28)."""

import dataclasses
import math

import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.cad_packer import _structure_boxes
from src.tools.aero_cad.layout import fuselage_radius_at
from src.tools.fabrication.build_kit import kit_spar_plan, kit_spar_torsion
from src.tools.fabrication.fuselage_parts import former_stations, fuselage_parts, socket_axis_m, wing_box_floor_m
from src.tools.fabrication.kit import KitUnavailable
from src.tools.fabrication.kit_mass import kit_mass_items, kit_mass_kg
from src.tools.fabrication.kit_pod import crutch_rails_fit
from src.tools.fabrication.tubes import (
    MIN_WEB_M,
    SparTorsion,
    aero_torque_nm,
    boom_tube,
    hover_pitch_torque_nm,
    interaction_ok,
    load_cases,
    manoeuvre_dynamic_pressure_pa,
    plan_spar,
    size_overwrap,
    tail_boom_check,
    tail_boom_ratio,
    torsion_stress_pa,
    tube_ratio,
)
from src.tools.sizing.aero_model import air_properties
from tests.test_fabrication_kit import _build

SETTINGS = DEFAULT_FABRICATION
G2_LIKE_MOMENT_NM = 291.7  # g2's centreline moment (flight governs)
SOCKET_32 = TubeStock(od_m=0.032, id_m=0.028)
WIDE_GAP_M = 0.5


@pytest.fixture(scope="module")
def g1_kit(designs, g1_kit_inputs):
    return _build(designs, g1_kit_inputs, "g1")


@pytest.fixture(scope="module")
def g2_kit(designs, g2_kit_inputs):
    return _build(designs, g2_kit_inputs, "g2")


# ------------------------------------------------------------------------------------- formulas ---
def test_torsion_stress_thin_wall_formula():
    tube = TubeStock(od_m=0.025, id_m=0.023)
    polar = math.pi / 32.0 * (0.025**4 - 0.023**4)
    assert torsion_stress_pa(tube, 20.0) == pytest.approx(20.0 * 0.0125 / polar)
    assert torsion_stress_pa(tube, 20.0) == pytest.approx(23.0e6, rel=0.01)


def test_interaction_rejects_combined_load_each_alone_passes():
    sigma = 0.8 * SETTINGS.sigma_allow_pa / SETTINGS.safety_factor
    tau = 0.8 * SETTINGS.carbon_shear_allow_pa / SETTINGS.safety_factor
    assert interaction_ok(sigma, 0.0, SETTINGS)
    assert interaction_ok(0.0, tau, SETTINGS)
    assert not interaction_ok(sigma, tau, SETTINGS)


def test_manoeuvre_dynamic_pressure_uses_v_a():
    rho = air_properties(0.0)[0]
    v_a = 16.0 * math.sqrt(SETTINGS.n_limit)
    assert manoeuvre_dynamic_pressure_pa(16.0, SETTINGS) == pytest.approx(0.5 * rho * v_a**2)


def test_hover_pitch_torque_is_thrust_times_arm():
    assert hover_pitch_torque_nm(68.0, -0.7) == pytest.approx(68.0 * 0.7)


def test_aero_torque_integrates_tip_inboard():
    def chord(_y: float) -> float:
        return 0.3

    q, cm0 = 500.0, -0.1
    torque = aero_torque_nm(10.0, 1.5, chord, q, 0.5, SETTINGS, cm0=cm0, spar_fraction=0.25)
    assert torque == pytest.approx(q * 0.3**2 * 0.1 * 1.0, rel=1e-3)  # spar on the AC: q c^2 |Cm0| (b - y)
    assert aero_torque_nm(10.0, 1.5, chord, q, 1.5, SETTINGS, cm0=cm0) == pytest.approx(0.0)
    offset = aero_torque_nm(10.0, 1.5, chord, q, 0.0, SETTINGS, cm0=0.0, spar_fraction=0.35)
    lift = SETTINGS.n_limit * 10.0 * 9.80665 / 2.0  # the whole half-wing limit lift, 0.1 c off the spar
    assert offset == pytest.approx(lift * 0.1 * 0.3, rel=1e-3)


def test_torsion_steps_the_spar_up_through_the_same_selection_rule():
    bending_only = plan_spar(5.0, 1.4, 0.1, lambda y: 0.05, None, SETTINGS)
    twisted = SparTorsion(flight_torque_at=lambda y: 25.0, hover_torque_nm=0.0, hover_thrust_n=0.0)
    with_torsion = plan_spar(5.0, 1.4, 0.1, lambda y: 0.05, None, SETTINGS, torsion=twisted)
    root = with_torsion.segments[0].tube
    assert root.od_m > bending_only.segments[0].tube.od_m
    cases = load_cases(5.0, 1.4, 0.1, None, twisted, SETTINGS)
    assert tube_ratio(root, cases, SETTINGS) <= 1.0


# ------------------------------------------------------------------------------------- overwrap ---
def test_overwrap_layers_grow_with_moment():
    t = SETTINGS.overwrap_layer_thickness_m
    wraps = [size_overwrap(m, SOCKET_32, WIDE_GAP_M, SETTINGS) for m in (40.0, 150.0, G2_LIKE_MOMENT_NM)]
    layers = [w.layers for w in wraps]
    assert layers == sorted(layers) and layers[0] < layers[-1]
    assert wraps[-1].layers == 9  # R29: g2 needs 9 layers
    for w in wraps:
        n, od = w.layers, SOCKET_32.od_m
        assert w.od_m == pytest.approx(od + 2 * n * t)
        assert w.lap_m >= 2 * od - 1e-12
        expected = math.pi * (od + n * t) * n * t * 2 * w.lap_m * SETTINGS.overwrap_density_kg_m3
        assert w.mass_kg == pytest.approx(expected)


def test_overwrap_lap_grows_with_epoxy_shear_demand():
    weak_glue = SETTINGS.model_copy(update={"epoxy_lap_shear_pa": 1e6})
    wrap = size_overwrap(G2_LIKE_MOMENT_NM, SOCKET_32, 10.0, weak_glue)
    assert wrap.lap_m > 2 * SOCKET_32.od_m


def test_overwrap_too_long_for_plate_gap_falls_back():
    with pytest.raises(KitUnavailable, match="overwrap"):
        size_overwrap(G2_LIKE_MOMENT_NM, SOCKET_32, 0.05, SETTINGS)


def test_overwrap_beyond_the_layer_cap_falls_back():
    with pytest.raises(KitUnavailable, match="overwrap"):
        size_overwrap(5000.0, SOCKET_32, WIDE_GAP_M, SETTINGS)


def test_g2_kit_has_an_overwrap_and_its_mass_counts(g2_kit):
    wrap = g2_kit.overwrap
    assert wrap is not None and wrap.layers == 9
    items = {i.name: i for i in kit_mass_items(g2_kit, SETTINGS)}
    assert items["OW-1"].mass_kg == pytest.approx(wrap.mass_kg)
    assert items["OW-1"].cg_m[1] == pytest.approx(0.0, abs=1e-9)
    plate = next(p for p in g2_kit.parts if p.label == "WM-R")
    gap = 2 * (plate.location.position.Y - SETTINGS.sheet_for("plate").thickness_m)
    assert 2 * wrap.lap_m <= gap


def test_overwrap_is_a_packing_obstacle_at_its_wrapped_od(g2_kit, g2_kit_inputs):
    layout, wrap = g2_kit_inputs[0], g2_kit.overwrap
    centre_z = socket_axis_m(layout, 0.0)[1]
    lowest = min(lo[2] for lo, hi in _structure_boxes(g2_kit, SETTINGS) if lo[1] < 0.0 < hi[1])
    assert lowest <= centre_z - wrap.od_m / 2 + 1e-9


def test_wing_box_floor_sits_under_the_wrap(designs, g2_kit_inputs):
    layout, airfoil = g2_kit_inputs[0], g2_kit_inputs[1]
    spar = kit_spar_plan(designs["g2"][0], layout, airfoil, SETTINGS)
    wrap = size_overwrap(spar.centre_moment_nm, spar.socket, WIDE_GAP_M, SETTINGS)
    assert wing_box_floor_m(layout, spar, SETTINGS) <= socket_axis_m(layout, 0.0)[1] - wrap.od_m / 2 + 1e-9


# -------------------------------------------------------------------------------------- torsion ---
def test_hover_pitch_torque_checked_at_boom_station(designs, g2_kit_inputs):
    sizing = designs["g2"][0]
    layout, airfoil = g2_kit_inputs[0], g2_kit_inputs[1]
    spar = kit_spar_plan(sizing, layout, airfoil, SETTINGS)
    torsion = kit_spar_torsion(sizing, layout, SETTINGS)
    boom_y = max(abs(b.y_m) for b in layout.vtol_booms)
    spar_x = socket_axis_m(layout, boom_y)[0]
    arm = max(abs(x - spar_x) for x, y, _ in layout.rotor_positions if y > 0)
    assert torsion.hover_torque_nm == pytest.approx(sizing.vtol_thrust_per_motor_n * arm)
    mtow, semi = sizing.mtow_kg, layout.wing.semi_span_m
    at_clamp = load_cases(mtow, semi, boom_y, boom_y, torsion, SETTINGS)
    assert max(t for _, t in at_clamp) >= torsion.hover_torque_nm
    outboard = load_cases(mtow, semi, boom_y + 0.01, boom_y, torsion, SETTINGS)
    assert max(t for _, t in outboard) < torsion.hover_torque_nm
    for seg in spar.segments:
        for y in (seg.y_start_m, min(max(boom_y, seg.y_start_m), seg.y_end_m)):
            assert tube_ratio(seg.tube, load_cases(mtow, semi, y, boom_y, torsion, SETTINGS), SETTINGS) <= 1.0


# ------------------------------------------------------------------------------------ tail boom ---
def test_tail_boom_too_weak_falls_back(designs, g1_kit_inputs):
    sizing, layout = designs["g1"][0], g1_kit_inputs[0]
    big_tail = sizing.model_copy(update={"horizontal_tail_area_m2": 2.0, "vertical_tail_area_m2": 2.0})
    tiny = boom_tube(0.0, SETTINGS)
    with pytest.raises(KitUnavailable, match="tail boom"):
        tail_boom_check(tiny, big_tail, layout, SETTINGS)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_reference_tail_booms_pass(case, designs, request):
    layout = request.getfixturevalue(f"{case}_kit_inputs")[0]
    tube = boom_tube(layout.tail_boom.diameter_m, SETTINGS)
    tail_boom_check(tube, designs[case][0], layout, SETTINGS)
    assert 0.0 < tail_boom_ratio(tube, designs[case][0], layout, SETTINGS) <= 1.0


def test_no_tail_boom_needs_no_check(designs, g1_kit_inputs):
    layout = dataclasses.replace(g1_kit_inputs[0], tail_boom=None)
    tail_boom_check(boom_tube(0.0, SETTINGS), designs["g1"][0], layout, SETTINGS)
    assert tail_boom_ratio(boom_tube(0.0, SETTINGS), designs["g1"][0], layout, SETTINGS) == 0.0


# -------------------------------------------------------------------------------------- bearing ---
def test_bearing_adds_socket_boss_doublers_only_when_needed(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    t = settings.sheet_for("plate").thickness_m
    plain_limit = settings.ply_bearing_allow_pa / settings.safety_factor * spar.socket.od_m * t
    low = fuselage_parts(layout, spar, (), settings, wing_shear_n=0.5 * plain_limit)
    assert not [p for p in low if p.label in ("WM-RD", "WM-LD")]
    high = fuselage_parts(layout, spar, (), settings, wing_shear_n=1.5 * plain_limit)
    bosses = {p.label: p for p in high if p.label in ("WM-RD", "WM-LD")}
    assert sorted(bosses) == ["WM-LD", "WM-RD"] and all(p.family == "plate" for p in bosses.values())
    plate = next(p for p in high if p.label == "WM-R")
    boss = bosses["WM-RD"]
    assert boss.location.position.Y == pytest.approx(plate.location.position.Y - t)  # on the plate's inner face
    assert len(boss.outline.inner_wires()) == 1
    with pytest.raises(KitUnavailable, match="bearing"):
        fuselage_parts(layout, spar, (), settings, wing_shear_n=2.5 * plain_limit)


# --------------------------------------------------------------------------------- crutch rails ---
def _between_formers(layout, spar, settings) -> PlacedComponent:
    """A short, flat component in a former bay aft of the wing box (only the crutch has to open for it)."""
    length = 0.03
    probe = PlacedComponent("battery", "box", 0.3, (length, 0.05, 0.02), (0.0, 0.0, -0.02))
    stations = former_stations(layout, spar, settings, ())
    t = settings.sheet_for("former").thickness_m
    for a, b in zip(stations, stations[1:]):
        if a > socket_axis_m(layout, 0.0)[0] and b - (a + t) > length + 2 * COMPONENT_GAP_M:
            return probe.at_x(0.5 * (a + t + b))
    raise AssertionError("no former bay found")


def _with_rail(probe: PlacedComponent, layout, settings, rail_m: float) -> PlacedComponent:
    inner = min(fuselage_radius_at(layout, x) for x in (probe.x_min, probe.x_max)) - settings.fuselage_cover_thickness_m
    return dataclasses.replace(probe, size_m=(probe.size_m[0], 2 * (inner - rail_m) - COMPONENT_GAP_M, 0.02))


def test_crutch_rail_below_minimum_is_unavailable(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    probe = _between_formers(layout, spar, settings)
    narrow = _with_rail(probe, layout, settings, 0.5 * (settings.crutch_rail_min_m + MIN_WEB_M))
    with pytest.raises(KitUnavailable, match="rail"):
        fuselage_parts(layout, spar, (narrow,), settings, wing_shear_n=0.0)
    wide = _with_rail(probe, layout, settings, settings.crutch_rail_min_m + 0.004)
    assert any(p.label == "C-1" for p in fuselage_parts(layout, spar, (wide,), settings, wing_shear_n=0.0))


def test_pod_sizing_checks_the_crutch_rails(g1_kit_inputs):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    mid = 0.5 * layout.fuselage_length_m
    probe = PlacedComponent("battery", "box", 0.3, (0.05, 0.05, 0.02), (mid, 0.0, 0.0))
    assert crutch_rails_fit(_with_rail(probe, layout, settings, settings.crutch_rail_min_m + 0.001), layout, settings)
    assert not crutch_rails_fit(_with_rail(probe, layout, settings, settings.crutch_rail_min_m - 0.001), layout,
                                settings)


# ------------------------------------------------------------------------------------ the kits ---
def test_g1_g2_still_build_kits_with_positive_margin(g1_kit, g2_kit):
    for kit in (g1_kit, g2_kit):
        assert kit.overwrap is not None
        assert kit_mass_kg(kit, SETTINGS) > 0.0
        assert not [p for p in kit.parts if p.label in ("WM-RD", "WM-LD")]  # light designs: plain plates bear it
    assert g1_kit.overwrap.layers == 5 and g2_kit.overwrap.layers == 9
