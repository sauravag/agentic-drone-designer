"""R35: tail-surface loads q_A S CL_max (no n_limit factor) on the tail boom and on strength-sized tail spars."""

import math

import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.layout import TAIL_AIRFOIL_NAME
from src.tools.fabrication.kit import KitUnavailable
from src.tools.fabrication.tubes import (
    TAIL_CL_MAX,
    elliptic_moment_nm,
    elliptic_root_moment_nm,
    flight_dynamic_pressure_pa,
    tail_boom_loads,
    tail_surface_load_n,
    tube_ratio,
)
from src.tools.fabrication.wing_parts import (
    fin_start_m,
    spar_depth_at,
    tail_root_radius_m,
    tail_spar_loads,
    tail_spar_tubes,
)
from src.units import G0

SETTINGS = DEFAULT_FABRICATION
TAIL = load_airfoil(TAIL_AIRFOIL_NAME)
REL = 1e-9


def _candidates() -> list[TubeStock]:
    rods = [TubeStock(od_m=r, id_m=0.0) for r in SETTINGS.rods_m]
    return sorted([*SETTINGS.tubes, *rods], key=lambda t: (t.od_m, t.area_m2))


def test_tail_surface_load_is_q_times_area_times_cl_max(designs):
    sizing = designs["g2"][0]
    q = flight_dynamic_pressure_pa(sizing, SETTINGS)
    assert tail_surface_load_n(sizing, 0.3, SETTINGS) == pytest.approx(q * 0.3 * TAIL_CL_MAX, rel=REL)


def test_elliptic_root_moment_matches_the_wing_formula():
    """The wing formula with n W / 2 = P is the same elliptic moment."""
    mtow, n, span, y = 10.0, 4.0, 1.2, 0.15
    half_lift = n * mtow * G0 / 2.0
    assert elliptic_root_moment_nm(half_lift, span, y) == pytest.approx(elliptic_moment_nm(mtow, n, span, y), rel=REL)


def test_tail_boom_carries_no_load_factor(designs, g2_kit_inputs):
    """R35 revises R26: M = q_A CL_max |(S_h, S_v)| arm, with no extra n_limit."""
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    q = flight_dynamic_pressure_pa(sizing, SETTINGS)
    arm = max(layout.htail.x_ac_m, layout.vtail.x_ac_m) - layout.tail_boom.x_start_m
    expected = q * TAIL_CL_MAX * math.hypot(sizing.horizontal_tail_area_m2, sizing.vertical_tail_area_m2) * arm
    assert tail_boom_loads(sizing, layout, SETTINGS)[0] == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_each_tail_spar_passes_its_surface_load(case, designs, request):
    """Regression (C1): every tail spar carries its surface's elliptic load at its root within the allowable."""
    sizing, layout = designs[case][0], request.getfixturevalue(f"{case}_kit_inputs")[0]
    spars = tail_spar_tubes(sizing, layout, TAIL, SETTINGS)
    loads = tail_spar_loads(sizing, layout, SETTINGS)
    assert set(spars) == set(loads) == {"htail", "vtail"}
    for name, tube in spars.items():
        moment = loads[name]
        assert 0.0 < tube_ratio(tube, ((moment, 0.0),), SETTINGS) <= 1.0


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_each_tail_spar_is_the_smallest_stock_that_passes_and_fits(case, designs, request):
    sizing, layout = designs[case][0], request.getfixturevalue(f"{case}_kit_inputs")[0]
    spars = tail_spar_tubes(sizing, layout, TAIL, SETTINGS)
    loads = tail_spar_loads(sizing, layout, SETTINGS)
    roots = ((layout.htail, tail_root_radius_m(layout), spars["htail"], loads["htail"]),
             (layout.vtail, fin_start_m(layout, SETTINGS), spars["vtail"], loads["vtail"]))
    for surface, root, tube, moment in roots:
        depth = spar_depth_at(surface, TAIL, SETTINGS, root_doubler_m=None)(root)
        assert tube.od_m <= depth + 1e-9
        smaller = [t for t in _candidates() if (t.od_m, t.area_m2) < (tube.od_m, tube.area_m2)]
        assert all(tube_ratio(t, ((moment, 0.0),), SETTINGS) > 1.0 for t in smaller)


def test_tail_spar_moment_is_the_elliptic_root_moment(designs, g2_kit_inputs):
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    loads = tail_spar_loads(sizing, layout, SETTINGS)
    half_h = tail_surface_load_n(sizing, sizing.horizontal_tail_area_m2, SETTINGS) / 2.0
    fin = tail_surface_load_n(sizing, sizing.vertical_tail_area_m2, SETTINGS)
    assert loads["htail"] == pytest.approx(
        elliptic_root_moment_nm(half_h, layout.htail.semi_span_m, tail_root_radius_m(layout)), rel=REL)
    assert loads["vtail"] == pytest.approx(
        elliptic_root_moment_nm(fin, layout.vtail.semi_span_m, fin_start_m(layout, SETTINGS)), rel=REL)


def test_weak_stock_makes_the_tail_spar_unavailable_naming_surface_and_station(designs, g2_kit_inputs):
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    weak = SETTINGS.model_copy(update={"tubes": (TubeStock(od_m=0.006, id_m=0.004),), "rods_m": (0.003,)})
    with pytest.raises(KitUnavailable, match=r"tail spar: .*horizontal tail.* at y = \d+\.\d{3} m"):
        tail_spar_tubes(sizing, layout, TAIL, weak)


def test_a_tail_too_thin_for_any_stock_is_unavailable(designs, g1_kit_inputs):
    sizing, layout = designs["g1"][0], g1_kit_inputs[0]
    thick_only = SETTINGS.model_copy(update={"tubes": (SETTINGS.tubes[-1],), "rods_m": (0.030,)})
    with pytest.raises(KitUnavailable, match="tail spar"):
        tail_spar_tubes(sizing, layout, TAIL, thick_only)


# --------------------------------------------------------------------------------------- V-tail (Task 8) ---
@pytest.fixture(scope="module")
def vtail_layout(vtail_g1):
    from tests.cad_helpers import _fitted_layout

    return _fitted_layout(*vtail_g1, station=0.45, with_propeller=True)


def test_vtail_panel_spar_moment_is_its_own_elliptic_root_moment(vtail_g1, vtail_layout):
    """R35 per panel: each V-tail panel lifts q S_panel CL_max normal to its plane, elliptically along the panel from
    its root (the root block's face)."""
    from src.tools.fabrication.wing_parts import panel_start_m

    sizing, surface = vtail_g1[0], vtail_layout.tail_surfaces[0]
    loads = tail_spar_loads(sizing, vtail_layout, SETTINGS)
    panel = tail_surface_load_n(sizing, surface.planform.area_m2 / 2.0, SETTINGS)
    start = panel_start_m(vtail_layout, surface, SETTINGS)
    assert set(loads) == {"vtail_pair"}
    assert loads["vtail_pair"] == pytest.approx(
        elliptic_root_moment_nm(panel, surface.planform.semi_span_m, start), rel=REL)
    spars = tail_spar_tubes(sizing, vtail_layout, TAIL, SETTINGS)
    assert 0.0 < tube_ratio(spars["vtail_pair"], ((loads["vtail_pair"], 0.0),), SETTINGS) <= 1.0


def test_vtail_boom_loads_project_each_panel(vtail_g1, vtail_layout):
    """R26 per surface: each panel's load L (normal to it) bends the boom through its components, L cos G (pitch) and
    L sin G (side); in the antisymmetric (yaw) case each panel twists it by its full L at its MAC station's lever."""
    sizing, surface = vtail_g1[0], vtail_layout.tail_surfaces[0]
    g = math.radians(surface.dihedral_deg)
    panel = tail_surface_load_n(sizing, surface.planform.area_m2 / 2.0, SETTINGS)
    arm = surface.planform.x_ac_m - vtail_layout.tail_boom.x_start_m
    moment, torque = tail_boom_loads(sizing, vtail_layout, SETTINGS)
    assert moment == pytest.approx(math.hypot(2.0 * panel * math.cos(g), 2.0 * panel * math.sin(g)) * arm, rel=REL)
    assert torque == pytest.approx(2.0 * panel * surface.planform.y_mac_m, rel=REL)


def test_tractor_boom_torque_is_the_fin_load_at_its_mac_height(designs, g2_kit_inputs):
    """The fin (90 deg) twists the boom by its full load at its MAC height; the flat H-tail adds no torque."""
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    fin = tail_surface_load_n(sizing, sizing.vertical_tail_area_m2, SETTINGS)
    height = layout.vtail.z_root_m + layout.vtail.y_mac_m - layout.tail_boom.z_m
    assert tail_boom_loads(sizing, layout, SETTINGS)[1] == pytest.approx(fin * height, rel=REL)
