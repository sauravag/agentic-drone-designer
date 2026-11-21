"""Wing stiffness model: D-cell geometry, Bredt torsion, tip deflection and aileron reversal (R43-R45)."""
import math

import pytest
from scipy.integrate import quad

from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.layout import QUARTER_CHORD
from src.tools.aero_cad.structure import SPAR_CHORD_STATION
from src.tools.fabrication.build_kit import kit_spar_plan, stiffened_spar_plan
from src.tools.fabrication.tubes import SEA_LEVEL_DENSITY_KG_M3, KitUnavailable, SparPlan, spar_member_at
from src.tools.fabrication.wing_stiffness import (
    DCell, d_cell, dbox_cell_at, dbox_gj_nm2, dbox_material_m2, dbox_web_x_m, flap_derivatives, lift_slope_per_rad,
    reversal_speed_ms, WingStiffness, stiffen_spar, tip_deflection_m, wing_stiffness,
)
from tests.cad_helpers import _fitted_layout


def _half_t(x, t=0.12):  # NACA 00xx half thickness
    return 5 * t * (0.2969 * x**0.5 - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def test_d_cell_of_naca0012_matches_the_analytic_thickness():
    cell = d_cell(load_airfoil("naca0012"), 0.20)
    assert cell.area_ratio == pytest.approx(quad(lambda x: 2 * _half_t(x), 0, 0.20)[0], rel=0.02)
    assert cell.web_ratio == pytest.approx(2 * _half_t(0.20), rel=0.02)
    assert cell.arc_ratio > 2 * 0.20


def _arc_integrand(x, t=0.12):
    dt = 5 * t * (0.2969 * 0.5 * x**-0.5 - 0.1260 - 2 * 0.3516 * x + 3 * 0.2843 * x**2 - 4 * 0.1015 * x**3)
    return math.sqrt(1 + dt**2)


def test_d_cell_arc_matches_the_analytic_contour_length():
    cell = d_cell(load_airfoil("naca0012"), 0.20)
    assert cell.arc_ratio == pytest.approx(2 * quad(_arc_integrand, 0, 0.20)[0], rel=2e-3)


def test_d_cell_of_a_cambered_airfoil_uses_the_thickness_between_the_surfaces():
    cell = d_cell(load_airfoil("naca2412"), 0.20)
    assert cell.web_ratio == pytest.approx(2 * _half_t(0.20), rel=0.03)
    assert cell.area_ratio == pytest.approx(quad(lambda x: 2 * _half_t(x), 0, 0.20)[0], rel=0.03)
    assert cell.arc_ratio > 2 * 0.20


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.0, 1.5])
def test_d_cell_rejects_a_web_outside_the_chord(fraction):
    with pytest.raises(ValueError):
        d_cell(load_airfoil("naca0012"), fraction)


def test_dbox_web_keeps_the_spar_hole_web():
    s = FabricationSettings()
    x = dbox_web_x_m(0.30, 0.014, s)
    assert x == pytest.approx(0.25 * 0.30 - ((0.014 + s.fit_clearance_m) / 2 + 0.002 + 0.0005))


def test_dbox_torsion_is_bredt_with_the_midline_area():
    s, c = FabricationSettings(), 0.25
    t = s.dbox_sheet_thickness_m
    p = (0.55 + 0.12) * c
    a_m = 0.03 * c**2 - 0.5 * t * p
    assert dbox_gj_nm2(DCell(0.03, 0.55, 0.12), c, s) == pytest.approx(4 * a_m**2 * s.balsa_shear_modulus_pa * t / p)


def test_tip_deflection_matches_uniform_load_cantilever():
    ei, w, L = 120.0, 30.0, 0.9
    assert tip_deflection_m(lambda y: ei, lambda y: w * (L - y) ** 2 / 2, 0.0, L) == pytest.approx(
        w * L**4 / (8 * ei), rel=1e-3)


def test_tip_deflection_from_an_offset_root_matches_tip_load_cantilever():
    ei, p, r, s = 80.0, 10.0, 0.1, 1.0
    assert tip_deflection_m(lambda y: ei, lambda y: p * (s - y), r, s) == pytest.approx(
        p * (s - r) ** 3 / (3 * ei), rel=1e-3)


def test_flap_theory_at_quarter_chord_aileron():
    cld, cmd = flap_derivatives(0.25, 2 * math.pi)
    assert cld == pytest.approx(3.8264, abs=1e-3) and cmd == pytest.approx(-0.6495, abs=1e-3)


def test_helmbold_lift_slope():
    assert lift_slope_per_rad(10.0) == pytest.approx(2 * math.pi * 10 / (2 + math.sqrt(104)))


def test_reversal_matches_closed_form_for_a_uniform_full_span_aileron():
    gj, c, s, a = 40.0, 0.2, 1.0, 5.0
    cld, cmd = flap_derivatives(0.25, a)
    q_r = 12 * cld * gj / (5 * a * c**2 * abs(cmd) * s**2)
    v = reversal_speed_ms(lambda y: gj, lambda y: c, 0.0, s, (0.0, s), a, cld, cmd)
    assert v == pytest.approx(math.sqrt(2 * q_r / SEA_LEVEL_DENSITY_KG_M3), rel=1e-4)


def test_reversal_closed_form_with_an_offset_root_uses_the_centreline_arm():
    # theta = q c^2 Cmd f(y)/GJ, f = s(y-r) - (y^2-r^2)/2; zero of integral_r^s c(a theta + Cld) y dy:
    # q_R = Cld (s^2 - r^2)/2 * GJ / (a c^2 |Cmd| integral_r^s f(y) y dy)
    gj, c, s, a, r = 40.0, 0.2, 1.0, 5.0, 0.15
    cld, cmd = flap_derivatives(0.25, a)
    f_int = quad(lambda y: (s * (y - r) - (y**2 - r**2) / 2) * y, r, s)[0]
    q_r = cld * (s**2 - r**2) / 2 * gj / (a * c**2 * abs(cmd) * f_int)
    v = reversal_speed_ms(lambda y: gj, lambda y: c, r, s, (r, s), a, cld, cmd)
    assert v == pytest.approx(math.sqrt(2 * q_r / SEA_LEVEL_DENSITY_KG_M3), rel=1e-4)


def test_reversal_weights_strips_cut_by_the_aileron_edges():
    # aileron edges fall mid-strip (strip width 0.0025); theta(y) = integral_0^y T/GJ with
    # T(y) = c^2 Cmd (a1 - max(y, a0)) inside the span, and B = -c a integral_0^s theta y dy
    gj, c, s, a, a0, a1 = 40.0, 0.2, 1.0, 5.0, 0.5013, 0.9507
    cld, cmd = flap_derivatives(0.25, a)

    def theta(y):
        return c**2 * cmd / gj * quad(lambda u: max(a1 - max(u, a0), 0.0), 0, y, points=[a0, a1])[0]

    b_term = -c * a * quad(lambda y: theta(y) * y, 0, s, points=[a0, a1], limit=200)[0]
    a_term = c * cld * (a1**2 - a0**2) / 2
    v = reversal_speed_ms(lambda y: gj, lambda y: c, 0.0, s, (a0, a1), a, cld, cmd)
    assert v == pytest.approx(math.sqrt(2 * (a_term / b_term) / SEA_LEVEL_DENSITY_KG_M3), rel=1e-4)


def test_reversal_speed_scales_with_the_root_of_torsional_stiffness():
    cld, cmd = flap_derivatives(0.25, 5.0)

    def speed(gj: float) -> float:
        return reversal_speed_ms(lambda y: gj, lambda y: 0.2, 0.1, 1.0, (0.5, 0.95), 5.0, cld, cmd)

    assert speed(160.0) == pytest.approx(2 * speed(40.0), rel=1e-9)


def test_reversal_is_infinite_without_a_torsional_wind_up():
    cld, cmd = flap_derivatives(0.25, 5.0)
    assert reversal_speed_ms(lambda y: 40.0, lambda y: 0.2, 0.0, 1.0, (0.5, 0.95), 5.0, cld, 0.0) == math.inf


def test_spar_sits_on_the_aerodynamic_centre():  # the reversal model's assumption
    assert SPAR_CHORD_STATION == QUARTER_CHORD


def test_wing_stiffness_uses_the_user_limits(g1_kit_inputs, designs):
    layout, airfoil, spar, settings = g1_kit_inputs
    sizing = designs["g1"][0]
    w = wing_stiffness(spar, airfoil, False, sizing.mtow_kg, layout.wing.semi_span_m, layout.wing.chord_at,
                       sizing.aspect_ratio, sizing.v_cruise_ms, settings)
    assert w.tip_deflection_limit_m == pytest.approx(0.05 * layout.wing.semi_span_m)
    assert w.reversal_required_ms == pytest.approx(1.15 * 1.4 * sizing.v_cruise_ms)
    assert not w.dbox
    assert w.deflection_ok  # whether g1 needs a D-box is not pinned here (its tube-only margin is thin)


def test_dbox_raises_reversal_speed_and_leaves_deflection(g2_kit_inputs, designs):
    layout, airfoil, spar, settings = g2_kit_inputs
    sizing = designs["g2"][0]

    def run(with_dbox: bool):
        return wing_stiffness(spar, airfoil, with_dbox, sizing.mtow_kg, layout.wing.semi_span_m,
                              layout.wing.chord_at, sizing.aspect_ratio, sizing.v_cruise_ms, settings)

    tube, box = run(False), run(True)
    assert box.dbox and not tube.dbox
    assert box.reversal_speed_ms > tube.reversal_speed_ms
    assert box.tip_deflection_m == pytest.approx(tube.tip_deflection_m)
    assert not tube.reversal_ok and box.reversal_ok and box.deflection_ok and box.passes


def _report(deflection_ok: bool, reversal_ok: bool, dbox: bool) -> WingStiffness:
    return WingStiffness(tip_deflection_m=0.10 if not deflection_ok else 0.04, tip_deflection_limit_m=0.05,
                         reversal_speed_ms=30.0 if not reversal_ok else 40.0, reversal_required_ms=35.0, dbox=dbox)


def _bare_spar() -> SparPlan:
    return SparPlan(segments=(), socket=None, joint_y_m=0.1, insertion_m=0.1, centre_moment_nm=0.0, pin=None)


def _recording_check(tube: WingStiffness, box: WingStiffness):
    calls: list[bool] = []

    def check(spar: SparPlan, with_dbox: bool) -> WingStiffness:
        calls.append(with_dbox)
        return box if with_dbox else tube

    return check, calls


def test_stiff_tube_gets_no_dbox():
    ok = _report(True, True, False)
    check, calls = _recording_check(ok, _report(True, True, True))
    spar = stiffen_spar(_bare_spar(), check, DEFAULT_FABRICATION)
    assert spar.dbox is False and spar.stiffness is ok and calls == [False]


def test_reversal_failure_falls_back_to_the_dbox():
    boxed = _report(True, True, True)
    check, calls = _recording_check(_report(True, False, False), boxed)
    spar = stiffen_spar(_bare_spar(), check, DEFAULT_FABRICATION)
    assert spar.dbox is True and spar.stiffness is boxed and calls == [False, True]


def test_deflection_failure_is_refused_without_trying_the_dbox():
    check, calls = _recording_check(_report(False, False, False), _report(True, True, True))
    with pytest.raises(KitUnavailable) as err:
        stiffen_spar(_bare_spar(), check, DEFAULT_FABRICATION)
    message = str(err.value)
    assert message.startswith("the wing spar is too flexible")
    assert "tip deflection" in message and "mm at 1 g" in message and "limit" in message
    assert calls == [False]


def test_dbox_that_still_reverses_is_refused():
    check, calls = _recording_check(_report(True, False, False), _report(True, False, True))
    with pytest.raises(KitUnavailable) as err:
        stiffen_spar(_bare_spar(), check, DEFAULT_FABRICATION)
    message = str(err.value)
    assert "even with the 1.5 mm balsa D-box" in message
    assert "aileron reversal at" in message and "1.15 x V_NE" in message
    assert calls == [False, True]


def test_kit_spar_plan_stays_strength_only(designs):
    sizing, bom = designs["g2"]
    layout = _fitted_layout(sizing, bom)
    spar = kit_spar_plan(sizing, layout, load_airfoil(layout.airfoil), DEFAULT_FABRICATION)
    assert spar.stiffness is None and spar.dbox is False  # kit_pod relies on it


def test_sizing_without_cruise_speed_is_refused(designs):
    sizing, bom = designs["g1"]
    layout = _fitted_layout(sizing, bom)
    old = sizing.model_copy(update={"v_cruise_ms": None})
    with pytest.raises(KitUnavailable, match="re-run"):
        stiffened_spar_plan(old, layout, load_airfoil(layout.airfoil), DEFAULT_FABRICATION)


def test_a_chord_too_small_for_the_spar_refuses_the_dbox_naming_the_station(g2_kit_inputs, designs):
    # Arrange: a 25 mm chord puts the D-box web ahead of the leading edge for any stock tube
    layout, airfoil, spar, settings = g2_kit_inputs
    sizing = designs["g2"][0]
    tiny_chord = 0.025
    # Act / Assert: the cell and the full stiffness path both refuse in words, not with a ValueError
    with pytest.raises(KitUnavailable, match=r"too large for a leading-edge D-box at y = 0\.\d{3} m"):
        dbox_cell_at(airfoil, tiny_chord, spar.segments[0].tube.od_m, 0.3, settings)
    with pytest.raises(KitUnavailable, match=r"too large for a leading-edge D-box at y = "):
        wing_stiffness(spar, airfoil, True, sizing.mtow_kg, layout.wing.semi_span_m, lambda y: tiny_chord,
                       sizing.aspect_ratio, sizing.v_cruise_ms, settings)


def test_dbox_material_area_matches_an_independent_quadrature(g2_kit_inputs):
    # Arrange: 2 x integral (arc + web)(y) c(y) dy from the joint to the tip, split at the spar segment steps
    layout, airfoil, spar, settings = g2_kit_inputs
    wing, root = layout.wing, spar.joint_y_m

    def material_per_m(y: float) -> float:
        c = wing.chord_at(y)
        od = spar_member_at(spar, y - 1e-7).od_m  # inboard member at a step
        cell = d_cell(airfoil, dbox_web_x_m(c, od, settings) / c)
        return (cell.arc_ratio + cell.web_ratio) * c

    steps = [s.y_end_m for s in spar.segments[:-1]]
    # Act
    expected = 2.0 * quad(material_per_m, root, wing.semi_span_m, points=steps or None, limit=200)[0]
    area = dbox_material_m2(spar, airfoil, wing.semi_span_m, wing.chord_at, settings)
    # Assert
    assert area == pytest.approx(expected, rel=1e-3)
