"""Task 6: `build_kit` assembles every part, tube and strip of the laser-cut kit; `kit_mass_items` weighs it."""

import math

import numpy as np
import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.aero_cad import components
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.layout import FUSELAGE_PROFILE, TAIL_AIRFOIL_NAME, fuselage_radius_at
from src.tools.fabrication.build_kit import MITRE_LABELS, build_kit, socket_end_m
from src.tools.fabrication.fittings import join_notes
from src.tools.fabrication.fuselage_parts import STRINGER_COUNT, former_stations, socket_axis_m
from src.tools.fabrication.kit import KitUnavailable, TubePart
from src.tools.fabrication.kit_mass import kit_mass_items, kit_mass_kg, residual_gear_item, tube_item
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, spar_depth_at
from src.tools.sizing.mass_model import GEAR_FRACTION
from tests.cad_helpers import part_solid, tube_solid
from tests.test_fabrication_fuselage import _assert_no_intersections, _real_components

SETTINGS = DEFAULT_FABRICATION
MASS_REL = 1e-6
INERTIA_REL = 1e-3
PIN_REACH_M = 0.020  # R18: the pin runs this far outboard of the root doubler
USABLE_LONG_M = 0.590
USABLE_SHORT_M = 0.390
RIM_ALLOWANCE_M = 0.003  # a square-cut tube end may stop this far short so its leaning rim clears a rib


def _placed(designs, case: str, layout):
    return (*_real_components(designs, case, layout), *components.vtol_motors(designs[case][1], layout))


def _build(designs, inputs, case: str):
    layout, airfoil = inputs[0], inputs[1]
    return build_kit(designs[case][0], layout, airfoil, _placed(designs, case, layout), SETTINGS)


@pytest.fixture(scope="module")
def g1_kit(designs, g1_kit_inputs):
    return _build(designs, g1_kit_inputs, "g1")


@pytest.fixture(scope="module")
def g2_kit(designs, g2_kit_inputs):
    return _build(designs, g2_kit_inputs, "g2")


def _items(kit) -> dict[str, mp.MassItem]:
    return {i.name: i for i in kit_mass_items(kit, SETTINGS)}


def _assert_item_matches(item: mp.MassItem, oracle: mp.MassItem, label: str) -> None:
    assert item.mass_kg == pytest.approx(oracle.mass_kg, rel=MASS_REL), label
    scale = max(abs(c) for c in oracle.cg_m)
    assert np.allclose(item.cg_m, oracle.cg_m, rtol=0.0, atol=MASS_REL * scale), label
    tensor, expected = np.array(item.inertia_kgm2), np.array(oracle.inertia_kgm2)
    assert np.abs(tensor - expected).max() <= INERTIA_REL * np.abs(expected).max(), label


# ------------------------------------------------------------------------------------------ group ---
def test_group_3_is_unavailable(designs, g1_kit_inputs):
    sizing = designs["g3"][0]
    assert sizing.dod_group == 3
    layout, airfoil = g1_kit_inputs[0], g1_kit_inputs[1]
    with pytest.raises(KitUnavailable, match="Group 1–2"):
        build_kit(sizing, layout, airfoil, (), SETTINGS)


def _mitre_name(label: str) -> str:
    """`_assert_no_intersections` skips pairs whose names both start with 'socket'."""
    return f"socket {label}" if label in MITRE_LABELS else label


@pytest.mark.xfail(strict=True, reason="R19/R20 pending")
@pytest.mark.parametrize("case", ["g1", "g2"])
def test_every_reference_design_in_group_1_2_gets_a_kit(case, request):
    """g1 (4.1 kg fixed wing) and g2 (19.4 kg quadplane) with real components: nothing intersects but the
    centreline mitre of the sockets and the spar roots inside them (R14)."""
    kit = request.getfixturevalue(f"{case}_kit")
    spar = request.getfixturevalue(f"{case}_kit_inputs")[2]
    assert kit.spar == spar
    assert kit.parts and kit.tubes and kit.linear
    solids = [(p.label, part_solid(p, SETTINGS)) for p in kit.parts]
    solids += [(_mitre_name(t.label), tube_solid(t)) for t in kit.tubes]
    _assert_no_intersections(solids)


def test_only_the_centreline_mitre_is_exempt():
    assert set(MITRE_LABELS) == {"SK-R", "SK-L", "SP-R1", "SP-L1"}


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_same_side_socket_and_spar_root_do_not_intersect(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    by = {t.label: t for t in kit.tubes}
    for side in "RL":
        common = tube_solid(by[f"SK-{side}"]).intersect(tube_solid(by[f"SP-{side}1"]))
        pieces = common if isinstance(common, list) else ([] if common is None else [common])
        assert sum(p.volume for p in pieces) < 1e-12


def test_every_part_fits_the_bed(g1_kit, g2_kit):
    for kit in (g1_kit, g2_kit):
        for part in kit.parts:
            size = part.outline.bounding_box().size
            long, short = max(size.X, size.Y), min(size.X, size.Y)
            assert long <= USABLE_LONG_M + 1e-9 and short <= USABLE_SHORT_M + 1e-9, part.label


def test_joins_describe_every_split(g2_kit):
    assert g2_kit.joins == join_notes(g2_kit.parts)
    assert len({p.label for p in g2_kit.parts}) == len(g2_kit.parts)


def test_kit_is_deterministic(designs, g1_kit_inputs, g1_kit):
    again = _build(designs, g1_kit_inputs, "g1")
    assert [p.label for p in again.parts] == [p.label for p in g1_kit.parts]
    assert [p.area_m2 for p in again.parts] == [p.area_m2 for p in g1_kit.parts]
    assert [t.label for t in again.tubes] == [t.label for t in g1_kit.tubes]
    assert [(i.name, i.mass_kg) for i in kit_mass_items(again, SETTINGS)] == [
        (i.name, i.mass_kg) for i in kit_mass_items(g1_kit, SETTINGS)
    ]


# ------------------------------------------------------------------------------------------ tubes ---
def _axis_point(layout, side: float, y: float, fraction: float = 0.25):
    x, z = socket_axis_m(layout, y, fraction)
    return np.array([x, side * y, z])


def _tube_end(tube: TubePart):
    return np.array(tube.start_m) + tube.length_m * np.array(tube.axis)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_spar_members_telescope_with_bonded_overlaps(case, request):
    """R18: each inner member reaches min_insertion_diameters x its OD into its outer member; the root runs
    from the centreline (inside the socket) to the first step."""
    kit = request.getfixturevalue(f"{case}_kit")
    layout, _, spar, _ = request.getfixturevalue(f"{case}_kit_inputs")
    by = {t.label: t for t in kit.tubes}
    for side, name in ((1.0, "R"), (-1.0, "L")):
        assert f"SP-{name}{len(spar.segments) + 1}" not in by
        for k, seg in enumerate(spar.segments):
            member = by[f"SP-{name}{k + 1}"]
            assert member.role == "spar" and member.tube == seg.tube
            end = _tube_end(member)
            assert np.allclose(np.cross(end - _axis_point(layout, side, 0.0), member.axis), 0.0, atol=1e-9)
            assert seg.y_end_m - RIM_ALLOWANCE_M <= abs(end[1]) <= seg.y_end_m + 1e-9  # trimmed clear of a rib
            overlap = 0.0 if k == 0 else SETTINGS.min_insertion_diameters * seg.tube.od_m
            inboard = 0.0 if k == 0 else seg.y_start_m
            start = _axis_point(layout, side, inboard) - overlap * np.array(member.axis)
            assert member.start_m == pytest.approx(tuple(start), abs=1e-9)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_sockets_run_from_the_centreline_to_the_skin_joint(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    layout, _, spar, _ = request.getfixturevalue(f"{case}_kit_inputs")
    by = {t.label: t for t in kit.tubes}
    for side, name in ((1.0, "R"), (-1.0, "L")):
        socket = by[f"SK-{name}"]
        assert socket.role == "socket" and socket.tube == spar.socket
        assert socket.start_m == pytest.approx(tuple(_axis_point(layout, side, 0.0)), abs=1e-9)
        end = socket_end_m(layout, spar)
        assert spar.joint_y_m - RIM_ALLOWANCE_M < end < spar.joint_y_m
        assert _tube_end(socket) == pytest.approx(_axis_point(layout, side, end), abs=1e-9)


def test_pins_run_from_the_plate_to_past_the_root_doubler(g1_kit, g1_kit_inputs):
    layout, _, spar, _ = g1_kit_inputs
    parts, by = {p.label: p for p in g1_kit.parts}, {t.label: t for t in g1_kit.tubes}
    t_plate, t_doubler = SETTINGS.sheet_for("plate").thickness_m, SETTINGS.sheet_for("doubler").thickness_m
    inner = parts["WM-R"].location.position.Y - t_plate
    outer = spar.joint_y_m + t_doubler + PIN_REACH_M
    for side, name in ((1.0, "R"), (-1.0, "L")):
        pin = by[f"PN-{name}"]
        assert pin.role == "pin" and (pin.tube.od_m, pin.tube.id_m) == (0.004, 0.0)
        assert pin.start_m == pytest.approx(tuple(_axis_point(layout, side, inner, PIN_CHORD_FRACTION)), abs=1e-9)
        assert _tube_end(pin) == pytest.approx(_axis_point(layout, side, outer, PIN_CHORD_FRACTION), abs=1e-9)


def test_booms_and_tail_spars_are_in_the_kit(g1_kit, g2_kit):
    def labels(kit) -> list[str]:
        return sorted(t.label for t in kit.tubes if t.role in ("tail_boom", "lift_boom", "tail_spar"))

    assert labels(g1_kit) == ["TB-1", "TS-HL", "TS-HR", "TS-V"]
    assert labels(g2_kit) == ["LB-L", "LB-R", "TB-1", "TS-HL", "TS-HR", "TS-V"]


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_tail_spars_run_out_as_far_as_the_tail_depth_admits(case, request):
    """Each tail spar reaches the tip, or the station where the tail grows too thin for it (never 'one rib')."""
    kit = request.getfixturevalue(f"{case}_kit")
    layout = request.getfixturevalue(f"{case}_kit_inputs")[0]
    tail = load_airfoil(TAIL_AIRFOIL_NAME)
    by = {t.label: t for t in kit.tubes}
    for label, surface, coord, origin in (("TS-HR", layout.htail, 1, 0.0), ("TS-V", layout.vtail, 2, layout.vtail.z_root_m)):
        spar = by[label]
        reach = _tube_end(spar)[coord] - origin
        depth = spar_depth_at(surface, tail, SETTINGS)
        assert depth(reach) >= spar.tube.od_m - 1e-6, label
        assert reach == pytest.approx(surface.semi_span_m, abs=1e-6) or depth(reach + 0.005) < spar.tube.od_m, label


# ----------------------------------------------------------------------------------------- linear ---
def test_stringers_run_between_the_first_and_last_former(designs, g1_kit, g1_kit_inputs):
    layout, _, spar, _ = g1_kit_inputs
    stations = former_stations(layout, spar, SETTINGS, _placed(designs, "g1", layout))
    stringers = [s for s in g1_kit.linear if s.kind == "stringer"]
    assert len(stringers) == STRINGER_COUNT
    t_former, side = SETTINGS.sheet_for("former").thickness_m, SETTINGS.stringer_side_m
    first, last = stations[1], stations[-1] + t_former
    for s in stringers:
        assert s.start_m[0] == pytest.approx(first) and s.end_m[0] == pytest.approx(last)
        for x, (_, y, z) in ((first, s.start_m), (last, s.end_m)):
            r_in = fuselage_radius_at(layout, x) - SETTINGS.fuselage_cover_thickness_m
            assert math.hypot(y, z) == pytest.approx(r_in - side / 2.0, abs=1e-9)
        assert s.section_area_m2 == pytest.approx(side**2)
        assert s.length_m == pytest.approx(math.dist(s.start_m, s.end_m))


def test_le_and_te_strips_per_panel_and_tail_surface(g1_kit):
    for kind in ("le", "te"):
        strips = sorted(s.label for s in g1_kit.linear if s.kind == kind)
        assert strips == sorted(f"{kind.upper()}-{s}" for s in ("WR", "WL", "HR", "HL", "V"))


# ------------------------------------------------------------------------------------------- mass ---
def test_plate_mass_matches_extruded_solid(g1_kit):
    items = _items(g1_kit)
    for part in g1_kit.parts:
        density = SETTINGS.sheet_for(part.family).density_kg_m3
        oracle = mp.solid_item(part.label, part_solid(part, SETTINGS), density)
        _assert_item_matches(items[part.label], oracle, part.label)


def test_tube_mass_matches_hollow_solid(g2_kit):
    items = _items(g2_kit)
    for tube in g2_kit.tubes:
        oracle = mp.solid_item(tube.label, tube_solid(tube), SETTINGS.carbon_density_kg_m3)
        _assert_item_matches(items[tube.label], oracle, tube.label)


def test_tube_item_follows_an_oblique_axis():
    direction = np.array([0.3, 0.8, -0.2])
    axis = tuple(float(v) for v in direction / np.linalg.norm(direction))
    tube = TubePart("T", "spar", SETTINGS.tubes[5], 0.4, (0.1, -0.2, 0.05), axis)
    density = SETTINGS.carbon_density_kg_m3
    item = tube_item("T", tube.tube.area_m2 * tube.length_m * density, tube.start_m, tube.axis, tube.length_m,
                     tube.tube.od_m, tube.tube.id_m)
    _assert_item_matches(item, mp.solid_item("T", tube_solid(tube), density), "oblique tube")


def _frustum_area(layout) -> float:
    length = layout.fuselage_length_m
    xs = [0.0, *(f * length for f, _ in FUSELAGE_PROFILE), length]
    rs = [fuselage_radius_at(layout, x) for x in xs]
    return sum(math.pi * (r0 + r1) * math.hypot(x1 - x0, r1 - r0) for x0, x1, r0, r1 in zip(xs, xs[1:], rs, rs[1:]))


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_kit_mass_is_sum_of_parts_tubes_stock_covering_and_glue(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    layout = request.getfixturevalue(f"{case}_kit_inputs")[0]
    items = kit_mass_items(kit, SETTINGS)
    by = {i.name: i for i in items}
    assert len(by) == len(items)
    assert kit_mass_kg(kit, SETTINGS) == pytest.approx(sum(i.mass_kg for i in items), rel=1e-12)
    structure = [by[p.label] for p in kit.parts] + [by[t.label] for t in kit.tubes] + [by[s.label] for s in kit.linear]
    glue = by["glue_hardware"]
    assert glue.mass_kg == pytest.approx(SETTINGS.glue_fraction * sum(i.mass_kg for i in structure), rel=1e-12)
    assert glue.cg_m == pytest.approx(mp.combine(structure).cg_m, abs=1e-12)
    planform = layout.wing.area_m2 + layout.htail.area_m2 + layout.vtail.area_m2
    assert kit.film_area_m2 == pytest.approx(2.0 * planform, rel=1e-9)
    assert kit.fuselage_cover_area_m2 == pytest.approx(_frustum_area(layout), rel=1e-6)
    film = sum(i.mass_kg for n, i in by.items() if n.startswith("film_"))
    assert film == pytest.approx(SETTINGS.film_kg_m2 * kit.film_area_m2, rel=1e-9)
    cover = SETTINGS.fuselage_cover_kg_m2 * kit.fuselage_cover_area_m2
    assert by["fuselage_cover"].mass_kg == pytest.approx(cover, rel=1e-9)
    named = {p.label for p in kit.parts} | {t.label for t in kit.tubes} | {s.label for s in kit.linear}
    assert set(by) - named == {"glue_hardware", "fuselage_cover", "film_wing", "film_htail", "film_vtail"}


def test_film_sits_on_the_wing_planform_centroid(g1_kit, g1_kit_inputs):
    wing = g1_kit_inputs[0].wing
    lam = wing.taper_ratio
    x_centroid = wing.x_le_root_m + wing.root_chord_m * (1 + lam + lam**2) / (3 * (1 + lam))
    item = _items(g1_kit)["film_wing"]
    assert item.cg_m[0] == pytest.approx(x_centroid, rel=1e-9)
    assert item.cg_m[1] == pytest.approx(0.0, abs=1e-12)


def test_residual_gear_item_is_the_gear_fraction_over_the_pod(designs, g2_kit_inputs):
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    item = residual_gear_item(sizing, layout)
    assert item.name == "landing_gear"
    assert item.mass_kg == pytest.approx(GEAR_FRACTION[sizing.airframe_topology] * sizing.mtow_kg)
    assert item.cg_m == pytest.approx((layout.fuselage_length_m / 2.0, 0.0, 0.0))
    expected = mp.cylinder_item("landing_gear", item.mass_kg, item.cg_m, layout.fuselage_length_m,
                                layout.fuselage_diameter_m)
    assert np.allclose(item.inertia_kgm2, expected.inertia_kgm2, rtol=1e-12, atol=0.0)
