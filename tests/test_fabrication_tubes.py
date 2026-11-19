import math

import pytest

from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.layout import WING_Z_FRACTION, layout_from_sizing
from src.tools.sizing.mission_solver import size_aircraft
from tests.reference_designs import CASES

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.fabrication.tubes import (
    MIN_WEB_M,
    KitUnavailable,
    bending_stress_pa,
    joint_y_m,
    plan_spar,
    elliptic_moment_nm,
    usable_depth_m,
)


def test_reviewer_numbers_25kg():
    m = elliptic_moment_nm(25.0, 4.0, 2.0, 0.0)
    assert m == pytest.approx(4.0 * 25 * 9.80665 / 2 * 4 * 2.0 / (3 * math.pi), rel=1e-9)  # ~416 N·m
    assert bending_stress_pa(TubeStock(od_m=0.025, id_m=0.023), m) > 900e6


def test_25kg_design_takes_a_32mm_spar_and_36mm_socket():
    plan = plan_spar(25.0, 2.1, 0.1, lambda y: 0.050, None, DEFAULT_FABRICATION)
    assert round(plan.segments[0].tube.od_m * 1e3) == 32  # 28x24 -> 419 MPa fails; 32x28 -> 312 MPa <= 333
    assert (round(plan.socket.od_m * 1e3), round(plan.socket.id_m * 1e3)) == (36, 32)


def test_spar_steps_down_inside_the_previous_tube():
    depth = lambda y: 0.030 if y < 0.8 else 0.015  # 16x14 root no longer fits past 0.8 m
    plan = plan_spar(5.0, 1.4, 0.06, depth, None, DEFAULT_FABRICATION)
    outer, inner = plan.segments[0].tube, plan.segments[1].tube
    assert inner.od_m <= outer.id_m and outer.id_m - inner.od_m <= 0.0002
    assert plan.segments[1].y_start_m == pytest.approx(plan.segments[0].y_end_m)


def test_socket_is_larger_than_the_spar():
    plan = plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, None, DEFAULT_FABRICATION)
    assert plan.socket.id_m >= plan.segments[0].tube.od_m


def test_hover_moment_governs_when_it_is_larger():
    gentle = DEFAULT_FABRICATION.model_copy(update={"n_limit": 1.0})  # flight 23.6 N·m < hover 43.9 N·m
    flight = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, None, gentle)
    hover = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, 1.2, gentle)
    assert hover.segments[0].tube.od_m > flight.segments[0].tube.od_m


def test_too_thin_wing_is_unavailable():
    with pytest.raises(KitUnavailable, match="no stock tube fits the wing at y ="):
        plan_spar(25.0, 2.1, 0.1, lambda y: 0.020, None, DEFAULT_FABRICATION)


def test_joint_is_on_the_skin():
    assert joint_y_m(0.10, 0.06) == pytest.approx(0.08)


def test_insertion_is_a_fraction_of_the_semi_span():
    plan = plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, None, DEFAULT_FABRICATION)
    assert plan.insertion_m == pytest.approx(0.20 * 1.4)
    assert plan.joint_y_m == 0.06


def test_missing_socket_is_unavailable():
    sparse = DEFAULT_FABRICATION.model_copy(update={"tubes": (TubeStock(od_m=0.020, id_m=0.016),)})
    with pytest.raises(KitUnavailable, match="socket"):
        plan_spar(2.0, 1.0, 0.05, lambda y: 0.03, None, sparse)


def test_step_down_with_no_nested_tube_is_unavailable():
    with pytest.raises(KitUnavailable, match="no stock tube fits the wing at y = 0.860 m"):
        plan_spar(5.0, 1.4, 0.06, lambda y: 0.030 if y < 0.8 else 0.005, None, DEFAULT_FABRICATION)


def test_tip_depth_is_checked():
    depth = lambda y: 0.03 if y < 1.4 - 1e-9 else 0.012
    with pytest.raises(KitUnavailable, match="no stock tube fits the wing at y = 1.400"):
        plan_spar(5.0, 1.4, 0.06, depth, None, DEFAULT_FABRICATION)


def test_no_step_plan_is_one_segment_to_the_tip():
    plan = plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, None, DEFAULT_FABRICATION)
    assert len(plan.segments) == 1
    assert plan.segments[0].y_end_m == pytest.approx(1.4)


@pytest.mark.parametrize(
    "depth",
    [
        lambda y: 0.030 if y < 0.8 else 0.015,
        lambda y: 0.030 if y < 1.4 - 1e-9 else 0.015,
    ],
)
def test_every_segment_fits_its_outboard_end(depth):
    plan = plan_spar(5.0, 1.4, 0.06, depth, None, DEFAULT_FABRICATION)
    for seg in plan.segments:
        assert seg.y_end_m > seg.y_start_m
        assert seg.tube.od_m <= depth(seg.y_end_m) + 1e-9
    assert plan.segments[-1].y_end_m == pytest.approx(1.4)


def test_moment_falls_to_zero_at_the_tip():
    assert elliptic_moment_nm(25.0, 4.0, 2.0, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert elliptic_moment_nm(25.0, 4.0, 2.0, 3.0) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < elliptic_moment_nm(25.0, 4.0, 2.0, 1.0) < elliptic_moment_nm(25.0, 4.0, 2.0, 0.0)


def test_a_rod_can_be_the_tip_member():
    settings = DEFAULT_FABRICATION.model_copy(
        update={"tubes": (TubeStock(od_m=0.005, id_m=0.003), TubeStock(od_m=0.007, id_m=0.005)), "rods_m": (0.003,)}
    )
    depth = lambda y: 0.030 if y < 0.3 else 0.0035
    plan = plan_spar(0.3, 0.5, 0.02, depth, None, settings)
    tip = plan.segments[-1].tube
    assert (tip.od_m, tip.id_m) == (0.003, 0.0)
    assert plan.segments[0].y_end_m == pytest.approx(0.3, abs=1e-6)


def _wing_depth(case: str):
    sizing = size_aircraft(CASES[case])
    layout = layout_from_sizing(sizing, 0.30, sizing.fuselage_diameter_m, sizing.fuselage_length_m)
    foil = load_airfoil(layout.airfoil).to_asb()
    ratio = float(foil.local_thickness(0.25))
    diameter = layout.fuselage_diameter_m
    depth = lambda y: usable_depth_m(ratio * layout.wing.chord_at(y), DEFAULT_FABRICATION)
    joint = joint_y_m(diameter / 2.0, WING_Z_FRACTION * diameter)
    boom_y = max(abs(b.y_m) for b in layout.vtol_booms) if layout.vtol_booms else None
    return sizing, layout, depth, joint, boom_y


@pytest.mark.parametrize("case, root_mm", [("g1", (14, 12)), ("g2", (28, 24))])
def test_reference_designs_get_a_spar(case, root_mm):
    sizing, layout, depth, joint, boom_y = _wing_depth(case)
    semi = layout.wing.semi_span_m
    plan = plan_spar(sizing.mtow_kg, semi, joint, depth, boom_y, DEFAULT_FABRICATION)
    root = plan.segments[0].tube
    assert (round(root.od_m * 1e3), round(root.id_m * 1e3)) == root_mm
    for seg in plan.segments:
        assert seg.y_end_m > seg.y_start_m
        assert seg.tube.od_m <= depth(seg.y_end_m) + 1e-9
    if case == "g1":
        assert plan.segments[-1].tube.od_m <= 0.006 + 1e-9


def test_several_steps_fit_between_two_rib_stations():
    def depth(y: float) -> float:
        return 0.050 - 0.028 * min(max((y - 1.45) / 0.10, 0.0), 1.0)

    plan = plan_spar(25.0, 2.1, 0.1, depth, None, DEFAULT_FABRICATION)
    inside = [s for s in plan.segments if 1.45 <= s.y_start_m <= 1.55]
    assert len(inside) >= 2
    for seg in plan.segments:
        assert seg.y_end_m > seg.y_start_m
        assert seg.tube.od_m <= depth(seg.y_end_m) + 1e-9


def test_negative_boom_distance_is_rejected():
    with pytest.raises(ValueError, match="boom_y_m"):
        plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, -0.5, DEFAULT_FABRICATION)


def test_usable_depth_allows_for_webs_and_hole_clearance():
    t = 0.030
    expected = t - 2 * MIN_WEB_M - DEFAULT_FABRICATION.fit_clearance_m
    assert usable_depth_m(t, DEFAULT_FABRICATION) == pytest.approx(expected, abs=1e-12)
