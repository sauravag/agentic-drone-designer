import math

import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.layout import MIN_BOOM_DIAMETER_M, WING_Z_FRACTION, layout_from_sizing
from src.tools.fabrication.tubes import (
    BOOM_DEFLECTION_FRACTION,
    MIN_WEB_M,
    KitUnavailable,
    bending_stress_pa,
    boom_tube,
    elliptic_moment_nm,
    hover_moment_nm,
    joint_y_m,
    lift_boom_tube,
    load_cases,
    plan_spar,
    tube_ratio,
    usable_depth_m,
)
from src.tools.sizing.mission_solver import size_aircraft
from tests.reference_designs import CASES


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
    """D6: the hover moment is the two rotors' MTOW g / 4 each at the boom's arm from the joint, 1.2 - 0.08 m."""
    gentle = DEFAULT_FABRICATION.model_copy(update={"n_limit": 1.0})  # flight 23.6 N·m < hover 43.9 N·m
    flight = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, None, gentle)
    hover = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, 1.2, gentle)
    assert hover.segments[0].tube.od_m > flight.segments[0].tube.od_m
    (_, _), (moment, _) = load_cases(8.0, 1.5, 0.08, 1.2, None, gentle)
    assert moment == pytest.approx(hover_moment_nm(8.0, 1.2 - 0.08)) == pytest.approx(2 * 8.0 * 9.80665 / 4 * 1.12)
    root = hover.segments[0].tube
    assert tube_ratio(root, ((moment, 0.0),), gentle) <= 1.0
    lighter = [t for t in gentle.tubes if (t.od_m, t.id_m) < (root.od_m, root.id_m)]
    assert all(tube_ratio(t, ((moment, 0.0),), gentle) > 1.0 for t in lighter)


def test_socket_is_sized_by_the_centreline_moment():
    """D6: a wide fuselage puts the joint far out, so the centreline moment, not the joint's, sizes the socket."""
    plan = plan_spar(10.0, 1.2, 0.3, lambda y: 0.05, None, DEFAULT_FABRICATION)
    joint_moment = elliptic_moment_nm(10.0, DEFAULT_FABRICATION.n_limit, 1.2, 0.3)
    assert plan.centre_moment_nm == pytest.approx(elliptic_moment_nm(10.0, DEFAULT_FABRICATION.n_limit, 1.2, 0.0))
    assert plan.centre_moment_nm > joint_moment
    root, socket = plan.segments[0].tube, plan.socket
    sleeves = [t for t in DEFAULT_FABRICATION.tubes if root.od_m - 1e-9 <= t.id_m <= root.od_m + 0.0002 + 1e-9]
    assert tube_ratio(socket, ((plan.centre_moment_nm, 0.0),), DEFAULT_FABRICATION) <= 1.0
    assert tube_ratio(sleeves[0], ((joint_moment, 0.0),), DEFAULT_FABRICATION) <= 1.0  # enough at the joint...
    assert sleeves[0] != socket  # ...but not at the centreline


def test_a_socket_too_weak_for_the_centreline_is_unavailable():
    """D6: the only stock that sleeves the root fails the centreline moment."""
    stock = DEFAULT_FABRICATION.model_copy(update={"tubes": (TubeStock(od_m=0.012, id_m=0.010),
                                                             TubeStock(od_m=0.014, id_m=0.012))})
    with pytest.raises(KitUnavailable, match="sleeve the 12 mm spar tube .* centreline moment"):
        plan_spar(6.0, 1.0, 0.25, lambda y: 0.0125, None, stock)  # joint 25 N·m, centre 50 N·m


@pytest.mark.parametrize("semi_span_m", [0.06, 0.05])
def test_a_wing_that_ends_inside_the_fuselage_is_unavailable(semi_span_m):
    """D8: no exposed span, no spar (the last segment would have no length)."""
    with pytest.raises(KitUnavailable, match="does not reach past the fuselage joint"):
        plan_spar(5.0, semi_span_m, 0.06, lambda y: 0.03, None, DEFAULT_FABRICATION)


def test_too_thin_wing_is_unavailable():
    with pytest.raises(KitUnavailable, match="no stock tube fits the wing at y ="):
        plan_spar(25.0, 2.1, 0.1, lambda y: 0.020, None, DEFAULT_FABRICATION)


def test_joint_is_on_the_skin():
    assert joint_y_m(0.10, 0.06) == pytest.approx(0.08)


def test_insertion_runs_from_the_skin_to_the_centreline():
    plan = plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, None, DEFAULT_FABRICATION)
    assert plan.insertion_m == pytest.approx(0.06)
    assert plan.joint_y_m == 0.06


def test_g1_insertion_clears_three_spar_diameters():
    sizing, layout, depth, joint, boom_y = _wing_depth("g1")
    plan = plan_spar(sizing.mtow_kg, layout.wing.semi_span_m, joint, depth, boom_y, DEFAULT_FABRICATION)
    assert plan.insertion_m == pytest.approx(joint)
    # 56 mm / 14 mm since the 2026-11-25 matched-propulsion re-baseline (3.88 = 54 mm / 14 mm before)
    assert plan.insertion_m / plan.segments[0].tube.od_m == pytest.approx(4.00, abs=0.05)


def test_insertion_shorter_than_three_diameters_is_unavailable():
    # 25 kg needs a 32 mm root tube: 3 x 32 = 96 mm of insertion, but the skin is only 50 mm from the centreline.
    with pytest.raises(KitUnavailable, match=r"insertion 50 mm .* 96 mm"):
        plan_spar(25.0, 2.1, 0.05, lambda y: 0.050, None, DEFAULT_FABRICATION)


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


# G2's root tube 28/24 -> 32/28 mm at the 2026-11-25 matched-propulsion re-baseline (19.9 -> 24.8 kg)
@pytest.mark.parametrize("case, root_mm", [("g1", (14, 12)), ("g2", (32, 28))])
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


@pytest.mark.parametrize("diameter_m, od_mm, id_mm", [(0.0385, 36, 32), (0.062, 40, 36)])
def test_boom_tube_is_the_largest_stock_within_the_layout_diameter(diameter_m, od_mm, id_mm):
    tube = boom_tube(diameter_m, DEFAULT_FABRICATION)
    assert (round(tube.od_m * 1e3), round(tube.id_m * 1e3)) == (od_mm, id_mm)


def test_boom_tube_below_all_stock_takes_the_minimum_boom_floor():
    assert boom_tube(0.010, DEFAULT_FABRICATION).od_m == pytest.approx(0.020)


# ------------------------------------------------------------------------------- R21 lift booms ---


def _passes_r21(tube, moment_nm, settings) -> bool:
    return bending_stress_pa(tube, moment_nm) <= settings.sigma_allow_pa / settings.safety_factor


@pytest.mark.parametrize("thrust_n, arm_m", [(60.0, 0.4), (120.0, 0.55), (20.0, 0.2)])
def test_lift_boom_is_the_smallest_stock_tube_that_carries_the_hub_moment(thrust_n, arm_m):
    settings, cap = DEFAULT_FABRICATION, 0.040
    tube = lift_boom_tube(thrust_n, arm_m, cap, settings)
    moment = thrust_n * arm_m
    assert _passes_r21(tube, moment, settings)
    assert MIN_BOOM_DIAMETER_M - 1e-9 <= tube.od_m <= cap + 1e-9
    smaller = [t for t in settings.tubes if (t.od_m, t.area_m2) < (tube.od_m, tube.area_m2)
               and t.od_m >= MIN_BOOM_DIAMETER_M - 1e-9]
    stiff = [t for t in smaller if _deflection_m(t, thrust_n, arm_m, settings) <= 0.01 * arm_m]  # R22 also applies
    assert not any(_passes_r21(t, moment, settings) for t in stiff)


def test_lift_boom_wider_than_the_layout_boom_is_unavailable():
    with pytest.raises(KitUnavailable, match="lift boom"):
        lift_boom_tube(400.0, 0.6, 0.020, DEFAULT_FABRICATION)


# ------------------------------------------------------------------------- R22 lift-boom stiffness ---


def _deflection_m(tube, thrust_n: float, arm_m: float, settings) -> float:
    return thrust_n * arm_m**3 / (3.0 * settings.carbon_modulus_pa * tube.second_moment_m4)


def test_carbon_modulus_and_deflection_limit_are_the_ruled_values():
    assert DEFAULT_FABRICATION.carbon_modulus_pa == pytest.approx(70e9)
    assert BOOM_DEFLECTION_FRACTION == pytest.approx(0.01)


@pytest.mark.parametrize("thrust_n, arm_m", [(68.8, 0.691), (30.0, 0.45), (120.0, 0.55)])
def test_lift_boom_tip_deflects_at_most_one_percent_of_its_arm(thrust_n, arm_m):
    settings, cap = DEFAULT_FABRICATION, 0.040
    tube = lift_boom_tube(thrust_n, arm_m, cap, settings)
    assert _deflection_m(tube, thrust_n, arm_m, settings) <= BOOM_DEFLECTION_FRACTION * arm_m
    smaller = [t for t in settings.tubes if (t.od_m, t.area_m2) < (tube.od_m, tube.area_m2)
               and t.od_m >= MIN_BOOM_DIAMETER_M - 1e-9]
    for t in smaller:
        ok = _passes_r21(t, thrust_n * arm_m, settings) and \
            _deflection_m(t, thrust_n, arm_m, settings) <= BOOM_DEFLECTION_FRACTION * arm_m
        assert not ok, t


def test_lift_boom_too_flexible_within_the_layout_diameter_is_unavailable():
    """D18: the message names the check that failed: deflection here (the boom is strong enough)."""
    with pytest.raises(KitUnavailable, match="too flexible: .* deflects"):
        lift_boom_tube(68.8, 0.691, 0.025, DEFAULT_FABRICATION)  # strong enough, far too flexible


def test_lift_boom_too_weak_names_the_stress():
    with pytest.raises(KitUnavailable, match="too weak: .* MPa"):
        lift_boom_tube(400.0, 0.6, 0.020, DEFAULT_FABRICATION)
