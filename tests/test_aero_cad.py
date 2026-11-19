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
from src.tools.aero_cad import build, cad_packer, components, mass_properties, stability, structure  # noqa: E402
from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil, write_selig_dat  # noqa: E402
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing  # noqa: E402
from tests.cad_helpers import _fitted_layout, _synthetic_aero, _synthetic_oml  # noqa: E402
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


def test_as_built_mass_counts_every_part(designs: dict, tmp_path: Path) -> None:
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
    assert _fitted_layout(sizing, bom).fuselage_diameter_m == pytest.approx(required)


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


def test_airframe_lighter_than_ribs_and_spars_raises(designs: dict, tmp_path: Path) -> None:
    sizing, bom = designs["g1"]
    lay = _fitted_layout(sizing, bom)
    starved = sizing.model_copy(update={"empty_mass_kg": 0.01})
    with pytest.raises(cad_packer.CADPackingError, match="no mass left for the airframe"):
        cad_packer.pack_layout(starved, bom, _synthetic_aero(lay), lay, _synthetic_oml(lay, tmp_path / "oml.step"))


def test_pack_aircraft_exports_assembly(designs: dict, tmp_path: Path) -> None:
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
