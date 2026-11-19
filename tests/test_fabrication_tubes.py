import math

import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.fabrication.tubes import (
    KitUnavailable,
    bending_stress_pa,
    joint_y_m,
    plan_spar,
    wing_root_moment_nm,
)


def test_reviewer_numbers_25kg():
    m = wing_root_moment_nm(25.0, 2.0, 4.0)
    assert m == pytest.approx(4.0 * 25 * 9.80665 / 2 * 4 * 2.0 / (3 * math.pi), rel=1e-9)  # ~416 N·m
    assert bending_stress_pa(TubeStock(od_m=0.025, id_m=0.023), m) > 900e6


def test_25kg_design_takes_a_32mm_spar_and_36mm_socket():
    plan = plan_spar(25.0, 2.1, 0.1, lambda y: 0.050, None, DEFAULT_FABRICATION)
    assert round(plan.segments[0].tube.od_m * 1e3) == 32  # 28x24 -> 419 MPa fails; 32x28 -> 312 MPa <= 333
    assert (round(plan.socket.od_m * 1e3), round(plan.socket.id_m * 1e3)) == (36, 32)


def test_spar_steps_down_inside_the_previous_tube():
    depth = lambda y: 0.030 if y < 0.8 else 0.017  # 18x16 root (259 MPa) no longer fits past 0.8 m
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
