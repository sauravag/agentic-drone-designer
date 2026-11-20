"""Milestone 3 acceptance tests: OpenVSP OML + VSPAERO derivatives, Build123d structure, CG packing, inertia.

Pure-Python / Build123d tests run anywhere. The end-to-end OpenVSP tests need the Docker toolchain:
    docker compose run --rm isr-forge xvfb-run -a pytest tests/test_aero_cad.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

build123d = pytest.importorskip("build123d")

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS  # noqa: E402
from src.schemas.fabrication import DEFAULT_FABRICATION  # noqa: E402
from src.tools.aero_cad import build, cad_packer, components, mass_properties, stability, structure  # noqa: E402
from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil, write_selig_dat  # noqa: E402
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing  # noqa: E402
from src.tools.fabrication import kit_export  # noqa: E402
from src.tools.fabrication.kit import KitUnavailable  # noqa: E402
from src.tools.fabrication.kit_mass import GEAR_NAME, GLUE_NAME, kit_mass_kg, residual_gear_item  # noqa: E402
from src.tools.fabrication.kit_pod import STRUCTURE_CLEARANCE_M, kit_fuselage_diameter_m  # noqa: E402
from tests.cad_helpers import _fitted_layout, _synthetic_aero, _synthetic_oml, part_solid, tube_solid  # noqa: E402
from tests.reference_designs import CASES  # noqa: E402

SM_MIN: float = 0.08
SM_MAX: float = 0.14
DEG: float = math.pi / 180.0


# ---------------------------------------------------------------------------------------- airfoils ---
def test_airfoil_thickness_and_camber() -> None:
    naca = load_airfoil("naca4415")
    assert naca.thickness_ratio == pytest.approx(0.15, abs=0.005)
    assert camber_at(naca, 0.4) == pytest.approx(0.04, abs=0.005)


def test_selig_dat_round_trip(tmp_path: Path) -> None:
    shape = load_airfoil("sd7037")
    path = write_selig_dat(shape, tmp_path / "sd7037.dat")
    lines = path.read_text().splitlines()
    assert lines[0] == "sd7037"
    assert len(lines) - 1 == len(shape.coordinates)
    assert tuple(map(float, lines[1].split())) == pytest.approx(shape.coordinates[0], abs=1e-6)


# ------------------------------------------------------------------------------------------ layout ---
@pytest.mark.parametrize("case", ["g1", "g2", "g3"])
def test_layout_reproduces_sizing_geometry(case: str, designs: dict) -> None:
    sizing, _ = designs[case]
    lay = layout_from_sizing(sizing)
    assert lay.wing.area_m2 == pytest.approx(sizing.wing_area_m2, rel=1e-9)
    assert 2 * lay.wing.semi_span_m == pytest.approx(sizing.wingspan_m, rel=1e-9)
    assert lay.htail.area_m2 == pytest.approx(sizing.horizontal_tail_area_m2, rel=1e-9)
    assert lay.vtail.area_m2 == pytest.approx(sizing.vertical_tail_area_m2, rel=1e-9)
    assert lay.htail.x_ac_m - lay.wing.x_ac_m == pytest.approx(sizing.tail_arm_m, rel=1e-9)
    assert lay.vtail.x_ac_m - lay.wing.x_ac_m == pytest.approx(sizing.tail_arm_m, rel=1e-9)


def test_tapered_mac_formula(designs: dict) -> None:
    lay = layout_from_sizing(designs["g1"][0])
    lam = lay.wing.taper_ratio
    assert lay.wing.mac_m == pytest.approx(2 / 3 * lay.wing.root_chord_m * (1 + lam + lam**2) / (1 + lam))


def test_quadplane_rotors_clear_fuselage_and_wing(designs: dict) -> None:
    sizing, _ = designs["g2"]
    lay = layout_from_sizing(sizing)
    r: float = lay.rotor_diameter_m / 2
    assert len(lay.rotor_positions) == 4 and len(lay.vtol_booms) == 2
    wing_te: float = lay.wing.x_le_root_m + lay.wing.root_chord_m
    for x, y, _ in lay.rotor_positions:
        assert abs(y) - r >= lay.fuselage_diameter_m / 2
        assert x + r <= lay.wing.x_le_root_m or x - r >= wing_te
    centroid_x = sum(p[0] for p in lay.rotor_positions) / 4
    assert centroid_x == pytest.approx(lay.wing.x_ac_m, rel=1e-9)  # hover thrust centred near the CG


def test_fixed_wing_has_no_rotors(designs: dict) -> None:
    lay = layout_from_sizing(designs["g1"][0])
    assert lay.rotor_positions == () and lay.vtol_booms == ()
    assert lay.tail_boom is not None


@pytest.mark.parametrize("station", [0.0, 0.95])
def test_layout_rejects_unphysical_wing_station(station: float, designs: dict) -> None:
    with pytest.raises(ValueError, match="wing_station"):
        layout_from_sizing(designs["g1"][0], wing_station=station)


# --------------------------------------------------------------------------------- stability math ---
def _coeffs(alpha: float, beta: float = 0.0, d_cmx: float = 0.0, d_cmy: float = 0.0, d_cmz: float = 0.0,
            d_cl: float = 0.0) -> stability.AeroCoefficients:
    cl: float = 0.4 + 0.09 * alpha + d_cl
    return stability.AeroCoefficients(
        alpha_deg=alpha, beta_deg=beta, cl=cl, cdi=0.04 * cl**2,
        cmx=0.001 + d_cmx, cmy=0.05 - 0.02 * alpha + d_cmy, cmz=0.0 + d_cmz,
    )


def _cases(**overrides: stability.AeroCoefficients) -> stability.VspaeroCases:
    base = dict(
        sweep=tuple(_coeffs(a) for a in (-4.0, 0.0, 4.0, 8.0, 12.0)),
        beta=_coeffs(0.0, beta=5.0, d_cmx=0.005, d_cmz=-0.01),
        aileron=_coeffs(0.0, d_cmx=0.04),
        elevator=_coeffs(0.0, d_cmy=-0.06, d_cl=0.01),
        rudder=_coeffs(0.0, d_cmz=-0.02),
        deflection_deg=10.0,
    )
    return stability.VspaeroCases(**{**base, **overrides})


def test_derivatives_from_synthetic_vspaero(g1_layout: AircraftLayout) -> None:
    ref = stability.ReferenceGeometry(sref_m2=0.8, cref_m=0.3, bref_m=2.0, x_ref_m=0.6)
    out = stability.derive_stability(_cases(), ref, g1_layout, cd0=0.03, cl_cruise=0.6)
    fus_cm: float = stability.fuselage_cm_alpha_per_rad(g1_layout, ref)
    fus_cn: float = stability.fuselage_cn_beta_per_rad(g1_layout, ref)
    assert out.cl0 == pytest.approx(0.4)
    assert out.cl_alpha_per_rad == pytest.approx(0.09 / DEG)
    assert out.cm_alpha_per_rad == pytest.approx(-0.02 / DEG + fus_cm)
    # body axes: Cn = -CMz, Cl = -CMx (VSPAERO geometry axes are x-aft, z-up)
    assert out.cn_beta_per_rad == pytest.approx(0.01 / (5 * DEG) + fus_cn)
    assert out.cl_beta_per_rad == pytest.approx(-0.005 / (5 * DEG))
    assert out.cl_delta_a_per_rad == pytest.approx(0.04 / (10 * DEG))  # convention: right roll > 0
    assert out.cm_delta_e_per_rad == pytest.approx(-0.06 / (10 * DEG))
    assert out.cn_delta_r_per_rad == pytest.approx(-0.02 / (10 * DEG))  # convention: TE-left rudder < 0
    assert out.ld_cruise == pytest.approx(0.6 / (0.03 + 0.04 * 0.36))
    assert out.x_np_m == pytest.approx(0.6 - out.cm_alpha_per_rad / out.cl_alpha_per_rad * 0.3)


def test_fuselage_is_destabilising(g1_layout: AircraftLayout) -> None:
    ref = stability.ReferenceGeometry(sref_m2=0.8, cref_m=0.3, bref_m=2.0, x_ref_m=0.6)
    assert stability.fuselage_cm_alpha_per_rad(g1_layout, ref) > 0
    assert stability.fuselage_cn_beta_per_rad(g1_layout, ref) < 0


def test_symmetric_aileron_deflection_is_rejected(g1_layout: AircraftLayout) -> None:
    ref = stability.ReferenceGeometry(sref_m2=0.8, cref_m=0.3, bref_m=2.0, x_ref_m=0.6)
    bad = _cases(aileron=_coeffs(0.0, d_cl=0.2, d_cmx=0.0001))  # flap-like response: mirror gain wrong
    with pytest.raises(stability.AeroAnalysisError, match="aileron"):
        stability.derive_stability(bad, ref, g1_layout, cd0=0.03, cl_cruise=0.6)


def test_non_lifting_geometry_is_rejected(g1_layout: AircraftLayout) -> None:
    ref = stability.ReferenceGeometry(sref_m2=0.8, cref_m=0.3, bref_m=2.0, x_ref_m=0.6)
    flat = tuple(stability.AeroCoefficients(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for a in (-4.0, 0.0, 4.0, 8.0, 12.0))
    with pytest.raises(stability.AeroAnalysisError, match="lift-curve slope"):
        stability.derive_stability(_cases(sweep=flat), ref, g1_layout, cd0=0.03, cl_cruise=0.6)


def test_cruise_lift_coefficient() -> None:
    assert stability.cruise_lift_coefficient(10.0, 1.2, 20.0, 1.0) == pytest.approx(2 * 10 * 9.80665 / (1.2 * 400))


# ---------------------------------------------------------------------------------------- structure ---
def test_rib_stations_every_twenty_centimetres(g1_layout: AircraftLayout) -> None:
    stations = structure.rib_stations(g1_layout)
    assert stations[0] == pytest.approx(g1_layout.fuselage_diameter_m / 2)
    assert np.allclose(np.diff(stations), structure.RIB_PITCH_M)
    assert stations[-1] <= g1_layout.wing.semi_span_m


def test_spar_cutout_follows_srs_rule() -> None:
    assert structure.spar_diameter_m(chord_m=0.3, thickness_ratio=0.12) == pytest.approx(0.08 * 0.036)


def test_ribs_are_airfoil_profiles_pierced_by_spar(g1_layout: AircraftLayout) -> None:
    shape = load_airfoil(g1_layout.airfoil)
    wing = structure.build_wing_structure(g1_layout, shape)
    stations = structure.rib_stations(g1_layout)
    assert len(wing.ribs) == 2 * len(stations)
    assert len(wing.spars) == 2
    y: float = stations[0]
    chord: float = g1_layout.wing.chord_at(y)
    xs, zs = np.array(shape.coordinates).T * chord
    profile_area: float = 0.5 * abs(np.dot(xs, np.roll(zs, 1)) - np.dot(zs, np.roll(xs, 1)))
    hole: float = math.pi * (structure.spar_diameter_m(chord, shape.thickness_ratio) / 2) ** 2
    assert wing.ribs[0].volume == pytest.approx((profile_area - hole) * structure.RIB_THICKNESS_M, rel=1e-4)
    assert wing.rib_mass_kg > 0 and wing.spar_mass_kg > 0


def test_spar_fits_every_rib_cutout(g1_layout: AircraftLayout) -> None:
    wing = structure.build_wing_structure(g1_layout, load_airfoil(g1_layout.airfoil))
    for rib in wing.ribs:
        for spar in wing.spars:
            assert cad_packer.overlap_volume_m3(rib, spar) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------------- mass properties ---
def test_box_inertia_matches_closed_form() -> None:
    item = mass_properties.box_item("b", 6.0, (0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert np.allclose(np.diag(item.inertia_kgm2), [0.5 * 13, 0.5 * 10, 0.5 * 5])


def test_occ_solid_inertia_matches_box_item() -> None:
    bd = build123d
    solid = bd.Pos(1.0, 2.0, 3.0) * bd.Box(0.4, 0.2, 0.1)
    occ = mass_properties.combine([mass_properties.solid_item("s", solid, density=1000.0)])
    ref = mass_properties.combine([mass_properties.box_item("b", 8.0, (1.0, 2.0, 3.0), (0.4, 0.2, 0.1))])
    assert occ.mass_kg == pytest.approx(8.0)
    assert np.allclose(occ.cg_m, (1.0, 2.0, 3.0))
    assert np.allclose(occ.inertia_kgm2, ref.inertia_kgm2)


def test_product_of_inertia_convention() -> None:
    items = [mass_properties.point_item("a", 1.0, (1.0, 0.0, 1.0)), mass_properties.point_item("b", 1.0, (-1.0, 0.0, -1.0))]
    props = mass_properties.combine(items)
    ixx, iyy, izz, ixz = mass_properties.body_inertia(props)
    assert (ixx, iyy, izz) == pytest.approx((2.0, 4.0, 2.0))
    assert ixz == pytest.approx(2.0)  # +sum(m x z); equal in CAD (x aft, z up) and body (x fwd, z down)
    assert props.inertia_kgm2[0][2] == pytest.approx(-2.0)  # tensor element


def test_shell_item_mass_is_area_times_areal_density() -> None:
    box = build123d.Box(1.0, 1.0, 1.0)
    item = mass_properties.shell_item("skin", box, areal_density=2.0)
    assert item.mass_kg == pytest.approx(12.0)


def test_positive_definite_check() -> None:
    assert mass_properties.is_positive_definite(np.diag([1.0, 2.0, 3.0]))
    assert not mass_properties.is_positive_definite(np.diag([1.0, 0.0, 3.0]))


# ------------------------------------------------------------------------------------- packaging ---
def test_collision_counter() -> None:
    a = cad_packer.PlacedComponent("a", "box", 1.0, (0.2, 0.1, 0.1), (0.0, 0.0, 0.0))
    overlapping = cad_packer.PlacedComponent("b", "box", 1.0, (0.2, 0.1, 0.1), (0.1, 0.0, 0.0))
    touching = cad_packer.PlacedComponent("c", "cylinder", 1.0, (0.2, 0.1, 0.1), (0.2, 0.0, 0.0))
    assert cad_packer.count_collisions((a, overlapping)) == 1
    assert cad_packer.count_collisions((a, touching)) == 0


def _synthetic_analyser(tmp_path: Path, np_offset_mac: float = 0.25):
    def analyse(layout: AircraftLayout) -> build.OmlAnalysis:
        step = _synthetic_oml(layout, tmp_path / f"oml_{layout.wing_station:.4f}.step")
        return build.OmlAnalysis(_synthetic_aero(layout, np_offset_mac), tmp_path / "oml.vsp3", step)
    return analyse


@pytest.mark.parametrize("case", ["g1", "g2", "g3"])
def test_balancing_hits_static_margin_without_collisions(case: str, designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs[case]
    design = build.balance_design(sizing, bom, _synthetic_analyser(tmp_path))
    result = design.packing
    assert SM_MIN <= result.static_margin <= SM_MAX
    assert result.collision_count == 0
    names = {c.name for c in result.components}
    assert {"payload", "battery", "avionics_tray", "cruise_motor"} <= names
    assert (result.kit is not None) == (case in ("g1", "g2")), result.kit_warning  # R23: no shell fallback
    assert ("fuel_tank" in names) == (sizing.propulsion_type == "series_hybrid")


def test_wing_slides_aft_when_battery_cannot_balance(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g3"]  # 28 kg catalog turret + conformal tank leave no forward battery slot at 0.30
    design = build.balance_design(sizing, bom, _synthetic_analyser(tmp_path))
    assert design.iterations > 1
    assert design.layout.wing_station > layout_from_sizing(sizing).wing_station


def test_unbalanceable_design_is_returned_for_the_critic(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]

    def nose_np(layout: AircraftLayout) -> build.OmlAnalysis:
        aero = _synthetic_aero(layout).model_copy(update={"x_np_m": 0.02})  # NP pinned at the nose
        return build.OmlAnalysis(aero, tmp_path / "oml.vsp3", _synthetic_oml(layout, tmp_path / "oml.step"))

    design = build.balance_design(sizing, bom, nose_np)
    assert not design.packing.is_balanced


def test_airframe_mass_round_trips_the_sizing_breakdown() -> None:
    from src.tools.sizing import mass_model as mm

    mtow: float = 20.0
    geom = mm.size_geometry(mtow, 1.0, 12.0, 0.5, 0.04, 1.0)
    loads = mm.PropulsionLoads(cruise_motor_power_w=800.0, vtol_total_thrust_n=280.0, generator_power_w=0.0)
    avionics: float = DEFAULT_ASSUMPTIONS.avionics_mass_kg(mtow, "cube_orange_plus", False)
    parts = mm.empty_mass_breakdown(mtow, geom, "quadplane_vtol", loads, avionics)
    structural: float = parts["wing"] + parts["tail"] + parts["fuselage"] + parts["landing_gear"] + mm.BOOM_FRACTION * mtow
    assert mm.airframe_mass_kg(mtow, sum(parts.values()), loads, avionics) == pytest.approx(structural)


@pytest.fixture()
def no_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the shell-model fallback (as if the kit could not be built)."""
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise KitUnavailable("forced by the test")
    monkeypatch.setattr(cad_packer, "build_kit", unavailable)


def test_as_built_mass_counts_every_part(designs: dict, tmp_path: Path, no_kit: None) -> None:
    sizing, bom = designs["g2"]
    lay = _fitted_layout(sizing, bom)
    result = cad_packer.pack_layout(sizing, bom, _synthetic_aero(lay), lay, _synthetic_oml(lay, tmp_path / "o.step"))
    airframe: float = cad_packer.sized_airframe_mass_kg(sizing)
    assert result.mass.mass_kg == pytest.approx(airframe + sum(c.mass_kg for c in result.components), rel=1e-9)
    vtol = [c for c in result.components if c.name.startswith("vtol_motor")]
    assert vtol and all(c.mass_kg > bom.vtol_motor.mass_kg + bom.vtol_propeller.mass_kg for c in vtol)  # ESC/mounts
    assert result.shell_areal_density_kg_m2 > 0


def test_part_poking_through_the_skin_is_counted(tmp_path: Path) -> None:
    oml = build123d.Pos(0.5, 0, 0) * build123d.Rot(0, 90, 0) * build123d.Cylinder(0.1, 1.0)
    inside = components.PlacedComponent("in", "box", 1.0, (0.2, 0.1, 0.1), (0.5, 0.0, 0.0))
    outside = components.PlacedComponent("out", "box", 1.0, (0.2, 0.3, 0.1), (0.5, 0.0, 0.0))
    external = components.PlacedComponent("ext", "box", 1.0, (0.2, 0.3, 0.1), (0.5, 0.0, 0.0), is_internal=False)
    assert components.containment_violations((inside, external), oml) == 0
    assert components.containment_violations((outside,), oml) == 1


def test_radial_fit_follows_the_pod_profile(designs: dict) -> None:
    sizing, bom = designs["g2"]
    lay = _fitted_layout(sizing, bom)
    battery, _ = components.sliders(sizing, bom)
    assert components.fits_radially(battery.at_x(0.5 * lay.fuselage_length_m), lay)
    assert not components.fits_radially(battery.at_x(battery.size_m[0] / 2 + 0.001), lay)  # nose cone


def test_pod_is_widened_to_house_the_cots_parts(designs: dict) -> None:
    sizing, bom = designs["g1"]  # catalog battery is wider than the fineness-ratio fuselage
    required: float = components.required_fuselage_diameter_m(sizing, bom)
    assert required > sizing.fuselage_diameter_m
    kit_pod: float = kit_fuselage_diameter_m(sizing, bom)  # R23: then widened so the battery passes under the wing box
    assert kit_pod > required
    assert _fitted_layout(sizing, bom).fuselage_diameter_m == pytest.approx(kit_pod)


def test_pod_is_lengthened_to_hold_the_hybrid_stack(designs: dict) -> None:
    sizing, bom = designs["g3"]  # battery + tray + tank + generator overflow the sized 3.1 m pod
    lay = _fitted_layout(sizing, bom)
    assert lay.fuselage_length_m > sizing.fuselage_length_m
    assert layout_from_sizing(sizing).fuselage_length_m == pytest.approx(sizing.fuselage_length_m)


def test_part_through_a_boom_is_a_collision(designs: dict) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    booms = components.boom_obstacles(lay)
    boom = booms[0]
    blocking = components.PlacedComponent("x", "box", 1.0, (0.05, 0.05, 0.05), boom.center_m)
    assert components.count_collisions((blocking,), booms) == 1


def test_open_oml_surface_is_rejected(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    open_shell = tmp_path / "open.step"
    build123d.export_step(build123d.Box(1, 1, 1).faces()[0], str(open_shell), unit=build123d.Unit.M)
    with pytest.raises(cad_packer.CADPackingError, match="not watertight"):
        cad_packer.pack_layout(sizing, bom, _synthetic_aero(lay), lay, open_shell)


def test_search_does_not_extrapolate_across_a_packing_jump(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g3"]  # battery flips fore/aft of the tank as the wing moves
    design = build.balance_design(sizing, bom, _synthetic_analyser(tmp_path))
    assert design.packing.is_balanced
    assert design.total_iterations <= build.MAX_BALANCE_ITERATIONS

def test_fuel_tank_centroid_sits_on_neutral_point(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g3"]
    lay = _fitted_layout(sizing, bom)
    aero = _synthetic_aero(lay)
    result = cad_packer.pack_layout(sizing, bom, aero, lay, _synthetic_oml(lay, tmp_path / "oml.step"))
    tank = next(c for c in result.components if c.name == "fuel_tank")
    assert tank.center_m[0] == pytest.approx(aero.x_np_m)


def test_unreachable_margin_is_reported_not_hidden(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    aero = _synthetic_aero(lay, np_offset_mac=-1.0)  # NP ahead of anything the battery can balance
    result = cad_packer.pack_layout(sizing, bom, aero, lay, _synthetic_oml(lay, tmp_path / "oml.step"))
    assert result.static_margin < SM_MIN
    assert not result.is_balanced


def test_sized_airframe_excludes_propulsion_and_systems(designs: dict) -> None:
    sizing, _ = designs["g2"]
    airframe: float = cad_packer.sized_airframe_mass_kg(sizing)
    assert 0 < airframe < sizing.empty_mass_kg


def test_airframe_lighter_than_ribs_and_spars_raises(designs: dict, tmp_path: Path, no_kit: None) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    starved = sizing.model_copy(update={"empty_mass_kg": 0.01})
    with pytest.raises(cad_packer.CADPackingError, match="no mass left for the airframe"):
        cad_packer.pack_layout(starved, bom, _synthetic_aero(lay), lay, _synthetic_oml(lay, tmp_path / "oml.step"))


def test_pack_aircraft_exports_assembly(designs: dict, tmp_path: Path, no_kit: None) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    oml = _synthetic_oml(lay, tmp_path / "oml.step")
    art = cad_packer.pack_aircraft(sizing, bom, _synthetic_aero(lay), lay, tmp_path / "oml.vsp3", oml, tmp_path)
    assembly = Path(art.assembly_step_path)
    assert assembly.exists() and assembly.stat().st_size > 0
    assert art.viewer_model_path is not None and Path(art.viewer_model_path).name == "viewer_model.json"
    assert Path(art.viewer_model_path).stat().st_size > 0
    reloaded = build123d.import_step(str(assembly))
    assert len(reloaded.solids()) >= 2 * len(structure.rib_stations(lay)) + 4
    tensor = np.array([[art.ixx_kgm2, 0, -art.ixz_kgm2], [0, art.iyy_kgm2, 0], [-art.ixz_kgm2, 0, art.izz_kgm2]])
    assert mass_properties.is_positive_definite(tensor)


# ---------------------------------------------------------------------------------- laser-cut kit ---
SLIDERS: tuple[str, ...] = ("battery", "avionics_tray")
SOLID_TOL_M3: float = 1e-12


def _pack(designs: dict, case: str, out_dir: Path) -> cad_packer.PackingResult:
    sizing, bom = designs[case]
    lay = _fitted_layout(sizing, bom)
    return cad_packer.pack_layout(sizing, bom, _synthetic_aero(lay), lay, _synthetic_oml(lay, out_dir / "oml.step"))


@pytest.fixture(scope="module")
def packed(designs: dict, tmp_path_factory: pytest.TempPathFactory):
    """Pack a reference design once per module (the kit is rebuilt inside every pack)."""
    cache: dict[str, cad_packer.PackingResult] = {}

    def get(case: str) -> cad_packer.PackingResult:
        if case not in cache:
            cache[case] = _pack(designs, case, tmp_path_factory.mktemp(f"pack_{case}"))
        return cache[case]
    return get


def _box(c: components.PlacedComponent) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(c.size_m) / 2
    return np.asarray(c.center_m) - half, np.asarray(c.center_m) + half


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_kit_replaces_the_shell_mass(case: str, packed) -> None:
    r = packed(case)
    assert r.kit is not None and r.kit_warning is None
    assert r.shell_areal_density_kg_m2 == 0.0 and r.structure is None
    names = {i.name for i in r.mass_items}
    assert "oml_shell" not in names and GEAR_NAME in names and GLUE_NAME in names


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_as_built_mass_is_kit_plus_gear_plus_components(case: str, designs: dict, packed) -> None:
    r = packed(case)
    sizing, bom = designs[case]
    gear = residual_gear_item(sizing, _fitted_layout(sizing, bom)).mass_kg
    expected = kit_mass_kg(r.kit, DEFAULT_FABRICATION) + gear + sum(c.mass_kg for c in r.components)
    assert r.mass.mass_kg == pytest.approx(expected, rel=1e-9)
    assert r.mass.mass_kg == pytest.approx(sum(i.mass_kg for i in r.mass_items), rel=1e-12)
    lay = _fitted_layout(sizing, bom)
    assert r.static_margin == pytest.approx((_synthetic_aero(lay).x_np_m - r.mass.cg_m[0]) / lay.wing.mac_m)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_kit_designs_pack_with_the_kit_and_a_positive_margin(case: str, packed) -> None:
    """R23 acceptance: widened pod, sliders seated low, kit rebuilt around them, SM > 0."""
    r = packed(case)
    assert r.kit is not None, r.kit_warning
    assert r.static_margin > 0.0 and r.interference_count == 0
    battery = next(c for c in r.components if c.name == "battery")
    assert battery.center_m[2] < 0.0  # seated low


def test_group_3_falls_back_with_a_warning(packed) -> None:
    r = packed("g3")
    assert r.kit is None and "Group 1–2" in r.kit_warning and r.kit_warning.startswith("No laser kit: ")
    assert r.structure is not None and r.shell_areal_density_kg_m2 > 0
    names = {i.name for i in r.mass_items}
    assert "oml_shell" in names and GLUE_NAME not in names and GEAR_NAME not in names


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_battery_never_overlaps_the_socket(case: str, packed) -> None:
    """The sliders clear the socket tubes in 3D (they may pass beneath, never through)."""
    r = packed(case)
    sockets = [t for t in r.kit.tubes if t.role == "socket"]
    assert len(sockets) == 2
    for name in SLIDERS:
        c = next(p for p in r.components if p.name == name)
        lo, hi = _box(c)
        for tube in sockets:
            t_lo, t_hi = (np.asarray(corner) for corner in cad_packer._tube_box(tube))
            gap = np.maximum(t_lo - hi, lo - t_hi)  # per-axis separation (negative = overlap)
            assert gap.max() >= STRUCTURE_CLEARANCE_M - 1e-9, f"{name} box is within 5 mm of the {tube.label} box"
            assert cad_packer.overlap_volume_m3(c.solid(), tube_solid(tube)) < SOLID_TOL_M3


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_formers_have_openings_for_the_placed_battery(case: str, packed) -> None:
    """The rebuilt kit cuts openings (or crutch notches) for the sliders: no part solid touches them."""
    r = packed(case)
    for name in SLIDERS:
        c = next(p for p in r.components if p.name == name)
        lo, hi = _box(c)
        solids = {p.label: part_solid(p, DEFAULT_FABRICATION) for p in r.kit.parts}
        crossing = [label for label, s in solids.items() if label.startswith("F-")
                    and s.bounding_box().min.X < hi[0] and lo[0] < s.bounding_box().max.X]
        assert crossing, f"no former crosses the {name}: the test would be vacuous"
        for label, solid in solids.items():
            volume = cad_packer.overlap_volume_m3(solid, c.solid())
            assert volume < SOLID_TOL_M3, f"{label} intersects the {name}: {volume:.3e} m^3"


def test_structure_boxes_block_in_3d_not_by_x_alone() -> None:
    socket = ((0.40, -0.10, 0.03), (0.45, 0.10, 0.05))
    beneath = components.PlacedComponent("b", "box", 1.0, (0.10, 0.05, 0.04), (0.42, 0.0, -0.01))
    through = components.PlacedComponent("t", "box", 1.0, (0.10, 0.05, 0.08), (0.42, 0.0, 0.0))
    too_close = components.PlacedComponent("c", "box", 1.0, (0.10, 0.05, 0.04), (0.42, 0.0, 0.007))  # top 3 mm under
    assert cad_packer._clear_of_structure(beneath, (socket,))
    assert not cad_packer._clear_of_structure(through, (socket,))
    assert not cad_packer._clear_of_structure(too_close, (socket,))
    assert cad_packer._clear_of_structure(through, ())


def test_kit_rebuild_failure_falls_back_to_the_shell(designs: dict, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    real = cad_packer.build_kit
    calls: list[int] = []

    def second_fails(*args: object, **kwargs: object):
        calls.append(1)
        if len(calls) > 1:
            raise KitUnavailable("F-04U: the opening for the battery leaves less than 2 mm of the former")
        return real(*args, **kwargs)

    monkeypatch.setattr(cad_packer, "build_kit", second_fails)
    r = _pack(designs, "g1", tmp_path)
    assert len(calls) == 2
    assert r.kit is None and r.kit_warning == "No laser kit: F-04U: the opening for the battery leaves less than 2 mm of the former"
    names = {i.name for i in r.mass_items}
    assert "oml_shell" in names and GLUE_NAME not in names
    assert r.mass.mass_kg == pytest.approx(sum(i.mass_kg for i in r.mass_items), rel=1e-12)
    sliders = [c for c in r.components if c.name in SLIDERS]
    assert all(c.center_m[2] == 0.0 for c in sliders)  # re-solved for the shell model: on the axis, not seated


def test_unexpected_kit_error_falls_back_to_the_shell(designs: dict, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    """Kit failure never fails the run, whatever the geometry kernel raises."""
    def broken(*_args: object, **_kwargs: object) -> None:
        raise ValueError("BRep_API: command not done")

    monkeypatch.setattr(cad_packer, "build_kit", broken)
    r = _pack(designs, "g1", tmp_path)
    assert r.kit is None and r.structure is not None
    assert r.kit_warning == "No laser kit: unexpected error building the kit (ValueError: BRep_API: command not done)"
    assert "oml_shell" in {i.name for i in r.mass_items}
    assert any(rec.levelname == "ERROR" and rec.exc_info for rec in caplog.records)  # logged with its traceback


def test_group_3_fallback_is_logged_at_info_not_warning(designs: dict, tmp_path: Path,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger=cad_packer.__name__)
    r = _pack(designs, "g3", tmp_path)
    no_kit = [rec for rec in caplog.records if cad_packer.NO_KIT_PREFIX in rec.getMessage()]
    assert r.kit is None and no_kit and all(rec.levelname == "INFO" for rec in no_kit)


def test_kit_design_exports_assembly_and_viewer(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    oml = _synthetic_oml(lay, tmp_path / "oml.step")
    art = cad_packer.pack_aircraft(sizing, bom, _synthetic_aero(lay), lay, tmp_path / "oml.vsp3", oml, tmp_path)
    assert Path(art.assembly_step_path).stat().st_size > 0
    assert art.viewer_model_path is not None and Path(art.viewer_model_path).stat().st_size > 0


# ------------------------------------------------------------------ end-to-end (Docker toolchain) ---
@pytest.fixture(scope="module", params=["g1", "g2", "g3"])
def built(request: pytest.FixtureRequest, designs: dict, tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("openvsp")
    from src.tools.aero_cad.build import build_aero_cad

    sizing, bom = designs[request.param]
    out_dir: Path = tmp_path_factory.mktemp(f"aero_cad_{request.param}")
    altitude: float = CASES[request.param].service_ceiling_m
    aero, cad = build_aero_cad(sizing, bom, altitude, out_dir)
    return sizing, aero, cad


def test_e2e_artifacts_are_valid_step(built) -> None:
    _, _, cad = built
    for path in (cad.oml_vsp3_path, cad.oml_step_path, cad.assembly_step_path):
        assert Path(path).exists() and Path(path).stat().st_size > 0
    oml = build123d.import_step(cad.oml_step_path)
    assert len(oml.solids()) >= 1  # trimmed, watertight surface intersection


def test_e2e_vspaero_derivatives_are_physical(built) -> None:
    sizing, aero, _ = built
    assert aero.cl_alpha_per_rad > 2.0
    assert aero.cm_alpha_per_rad < 0
    assert aero.cn_beta_per_rad > 0
    assert aero.cl_beta_per_rad < 0
    assert aero.cl_delta_a_per_rad > 1e-4
    assert aero.cm_delta_e_per_rad < -1e-4
    assert aero.cn_delta_r_per_rad < -1e-4
    assert 0.6 <= aero.ld_cruise / sizing.assumed_ld_cruise <= 1.5


def test_e2e_balanced_collision_free_and_inertia(built) -> None:
    _, _, cad = built
    assert SM_MIN <= cad.static_margin <= SM_MAX
    assert cad.collision_count == 0
    tensor = np.array([[cad.ixx_kgm2, 0, -cad.ixz_kgm2], [0, cad.iyy_kgm2, 0], [-cad.ixz_kgm2, 0, cad.izz_kgm2]])
    assert mass_properties.is_positive_definite(tensor)


def test_tray_mass_follows_sized_avionics(designs: dict) -> None:
    sizing, bom = designs["g1"]
    tray = components.sliders(sizing, bom)[1]
    assert tray.mass_kg == pytest.approx(sizing.avionics_mass_kg + mm_systems_fraction() * sizing.mtow_kg)
    heavier = sizing.model_copy(update={"avionics_mass_kg": sizing.avionics_mass_kg + 1.0})
    assert components.sliders(heavier, bom)[1].mass_kg == pytest.approx(tray.mass_kg + 1.0)


def mm_systems_fraction() -> float:
    from src.tools.sizing.mass_model import SYSTEMS_FRACTION
    return SYSTEMS_FRACTION


# ------------------------------------------------------------------------- kit files and 3D export ---
def test_artifacts_carry_kit_paths(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    art = cad_packer.pack_aircraft(sizing, bom, _synthetic_aero(lay), lay, tmp_path / "oml.vsp3",
                                   _synthetic_oml(lay, tmp_path / "oml.step"), tmp_path)
    assert art.kit_warning is None and art.kit_dir == str(tmp_path / "kit")
    assert art.build_sheet_path is not None and Path(art.build_sheet_path).exists()
    assert list((tmp_path / "kit" / "sheets").glob("*.dxf")) and (tmp_path / "kit" / "parts.csv").exists()
    assert art.kit_mass_kg is not None and art.kit_mass_kg > 0


def test_artifact_kit_mass_is_the_packed_kit_mass(packed, tmp_path: Path) -> None:
    r = packed("g1")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert art.kit_mass_kg == pytest.approx(kit_mass_kg(r.kit, DEFAULT_FABRICATION), rel=1e-12)


def test_one_export_builds_the_kit_mass_items_and_solids_once(packed, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: the export reuses the packed kit's mass items and builds its solids once for STEP and the viewer."""
    from src.tools.aero_cad import mesh_export
    from src.tools.fabrication import kit_mass

    r = packed("g1")
    assert r.kit_items and sum(i.mass_kg for i in r.kit_items) == pytest.approx(kit_mass_kg(r.kit, DEFAULT_FABRICATION))
    calls: dict[str, int] = {"items": 0, "ply": 0, "carbon": 0}

    def counted(name: str, real):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)
        return wrapper

    for module in (kit_mass, kit_export, cad_packer, mesh_export):
        if hasattr(module, "kit_mass_items"):
            monkeypatch.setattr(module, "kit_mass_items", counted("items", kit_mass.kit_mass_items))
    monkeypatch.setattr(kit_export, "ply_solids", counted("ply", kit_export.ply_solids))
    monkeypatch.setattr(kit_export, "carbon_solids", counted("carbon", kit_export.carbon_solids))
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert art.viewer_model_path is not None and art.kit_warning is None
    assert calls == {"items": 0, "ply": 1, "carbon": 1}


def test_fallback_artifacts_carry_the_warning(packed, tmp_path: Path) -> None:
    r = packed("g3")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert art.kit_dir is None and art.build_sheet_path is None and art.kit_mass_kg is None
    assert art.kit_warning == r.kit_warning and art.kit_warning.startswith("No laser kit: ")
    assert not (tmp_path / "kit").exists()


@pytest.mark.parametrize("error", [KitUnavailable("DXF writer broke"), RuntimeError("DXF writer broke")])
def test_kit_file_failure_does_not_fail_the_run(packed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture, error: Exception) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(kit_export, "write_sheets", broken)
    r = packed("g1")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert art.kit_warning == "kit files could not be written: DXF writer broke"
    assert art.kit_dir is None and art.build_sheet_path is None
    assert art.kit_mass_kg == pytest.approx(kit_mass_kg(r.kit, DEFAULT_FABRICATION))  # the kit was still packed
    assert Path(art.assembly_step_path).exists() and art.viewer_model_path is not None
    assert any(rec.levelname == "ERROR" and rec.exc_info for rec in caplog.records)


def test_kit_solid_failure_keeps_the_assembly_without_structure(packed, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise ValueError("BRep_API: command not done")

    monkeypatch.setattr(cad_packer, "kit_solid_sets", broken)
    r = packed("g1")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert Path(art.assembly_step_path).exists() and art.viewer_model_path is not None
    assert art.kit_warning == "kit solids could not be built: BRep_API: command not done"
    assert art.build_sheet_path is not None  # the kit files were still written


def test_assembly_carries_the_kit_solids_not_shell_ribs(packed, tmp_path: Path) -> None:
    r = packed("g1")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    solids = build123d.import_step(art.assembly_step_path).solids()
    expected = (len(r.oml.solids()) + len(kit_export.kit_solids(r.kit, DEFAULT_FABRICATION))
                + sum(len(c.solid().solids()) for c in r.components))
    assert len(solids) == expected


def test_fallback_assembly_still_carries_the_shell_ribs_and_spars(packed, tmp_path: Path) -> None:
    r = packed("g3")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    solids = build123d.import_step(art.assembly_step_path).solids()
    shell = len(r.structure.ribs) + len(r.structure.spars)
    assert shell > 0
    assert len(solids) == len(r.oml.solids()) + shell + sum(len(c.solid().solids()) for c in r.components)


def test_fallback_cleanup_failure_is_reported_as_cleanup(packed, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("kit/parts.csv: permission denied")

    monkeypatch.setattr(cad_packer, "remove_kit_files", denied)
    r = packed("g3")
    art = cad_packer.export_artifacts(r, tmp_path / "oml.vsp3", tmp_path / "oml.step", tmp_path)
    assert art.kit_warning == f"{r.kit_warning}; stale kit files could not be removed: kit/parts.csv: permission denied"
    assert "could not be written" not in art.kit_warning
