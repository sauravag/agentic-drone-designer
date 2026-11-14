"""Build123d internal wing structure: airfoil ribs every 0.20 m pierced by a tapered tubular carbon spar.

Per SRS 4.3 the spar cut-out diameter is D = 0.08 * t_local (t = airfoil thickness at that rib). The
spar is a hollow carbon frustum whose outer diameter tracks the cut-outs minus an adhesive bond gap,
so it threads every rib with zero solid overlap despite dihedral and taper.
"""

import math
from dataclasses import dataclass
from typing import Final

import build123d as bd

from src.tools.aero_cad.airfoil_geometry import AirfoilShape, camber_at
from src.tools.aero_cad.layout import AircraftLayout

RIB_PITCH_M: Final[float] = 0.20
RIB_THICKNESS_M: Final[float] = 0.003
SPAR_DIAMETER_TO_THICKNESS: Final[float] = 0.08  # SRS 4.3: D_spar = 0.08 * t_airfoil
SPAR_CHORD_STATION: Final[float] = 0.25
SPAR_BOND_GAP_FRACTION: Final[float] = 0.05  # spar OD = (1 - gap) * cut-out diameter
SPAR_WALL_FRACTION: Final[float] = 0.15  # wall thickness / outer diameter
COMPOSITE_DENSITY_KG_M3: Final[float] = 1600.0
CUTTER_LENGTH_FACTOR: Final[float] = 4.0  # hole cutter length, in rib thicknesses
STATION_EPS: Final[float] = 1e-9

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class WingStructure:
    ribs: tuple[bd.Solid, ...]  # right-wing ribs root->tip, then mirrored left-wing ribs
    spars: tuple[bd.Solid, ...]  # right, left
    rib_mass_kg: float
    spar_mass_kg: float


def rib_stations(layout: AircraftLayout) -> tuple[float, ...]:
    """Spanwise rib positions from the fuselage side wall to the tip, every RIB_PITCH_M."""
    root: float = layout.fuselage_diameter_m / 2
    span: float = layout.wing.semi_span_m - root
    if span <= 0:
        raise ValueError(f"wing semi-span {layout.wing.semi_span_m:.3f} m does not clear the fuselage")
    count: int = math.floor(span / RIB_PITCH_M + STATION_EPS) + 1
    return tuple(root + k * RIB_PITCH_M for k in range(count))


def spar_diameter_m(chord_m: float, thickness_ratio: float) -> float:
    """SRS cut-out diameter at a rib of local chord `chord_m`."""
    return SPAR_DIAMETER_TO_THICKNESS * thickness_ratio * chord_m


def _spar_point(layout: AircraftLayout, shape: AirfoilShape, y: float) -> Point3:
    chord: float = layout.wing.chord_at(y)
    x: float = layout.wing.x_le_root_m + SPAR_CHORD_STATION * chord
    z: float = layout.wing.z_at(y) + camber_at(shape, SPAR_CHORD_STATION) * chord
    return (x, y, z)


def _profile_points(layout: AircraftLayout, shape: AirfoilShape, y: float) -> list[Point3]:
    chord: float = layout.wing.chord_at(y)
    x0, z0 = layout.wing.x_le_root_m, layout.wing.z_at(y)
    pts: list[Point3] = [(x0 + chord * xc, y, z0 + chord * zc) for xc, zc in shape.coordinates]
    return pts[:-1] if pts[0] == pts[-1] else pts  # make_polygon closes the loop itself


def _rib(layout: AircraftLayout, shape: AirfoilShape, y: float, spar_axis: bd.Vector) -> bd.Solid:
    """Airfoil slab with the spar cut-out drilled along the spar axis (not the rib normal), so the
    dihedral/taper-tilted spar stays coaxial with its hole through the full rib thickness."""
    face = bd.Face(bd.Wire.make_polygon(_profile_points(layout, shape, y), close=True))
    slab = bd.extrude(face, amount=RIB_THICKNESS_M / 2, dir=(0, 1, 0), both=True)
    radius: float = spar_diameter_m(layout.wing.chord_at(y), shape.thickness_ratio) / 2
    cutter_len: float = CUTTER_LENGTH_FACTOR * RIB_THICKNESS_M
    start = bd.Vector(_spar_point(layout, shape, y)) - spar_axis * (cutter_len / 2)
    cutter = bd.Solid.make_cylinder(radius, cutter_len, bd.Plane(origin=start, z_dir=spar_axis))
    return (slab - cutter).solid()


def _frustum(p0: Point3, p1: Point3, r0: float, r1: float) -> bd.Solid:
    axis = bd.Vector(p1) - bd.Vector(p0)
    return bd.Solid.make_cone(r0, r1, axis.length, bd.Plane(origin=p0, z_dir=axis))


def _spar(layout: AircraftLayout, shape: AirfoilShape, y_root: float) -> bd.Solid:
    """Hollow tapered tube along the right wing's 25 %-chord line, fuselage wall to tip."""
    y_tip: float = layout.wing.semi_span_m
    p0, p1 = _spar_point(layout, shape, y_root), _spar_point(layout, shape, y_tip)
    fit: float = (1 - SPAR_BOND_GAP_FRACTION) / 2
    r0: float = fit * spar_diameter_m(layout.wing.chord_at(y_root), shape.thickness_ratio)
    r1: float = fit * spar_diameter_m(layout.wing.chord_at(y_tip), shape.thickness_ratio)
    bore: float = 1 - 2 * SPAR_WALL_FRACTION
    return (_frustum(p0, p1, r0, r1) - _frustum(p0, p1, bore * r0, bore * r1)).solid()


def build_wing_structure(layout: AircraftLayout, shape: AirfoilShape) -> WingStructure:
    stations: tuple[float, ...] = rib_stations(layout)
    axis = (bd.Vector(_spar_point(layout, shape, layout.wing.semi_span_m))
            - bd.Vector(_spar_point(layout, shape, stations[0]))).normalized()
    right_ribs: list[bd.Solid] = [_rib(layout, shape, y, axis) for y in stations]
    right_spar: bd.Solid = _spar(layout, shape, stations[0])
    ribs = tuple(right_ribs + [r.mirror(bd.Plane.XZ) for r in right_ribs])
    spars = (right_spar, right_spar.mirror(bd.Plane.XZ))
    return WingStructure(
        ribs=ribs, spars=spars,
        rib_mass_kg=COMPOSITE_DENSITY_KG_M3 * sum(r.volume for r in ribs),
        spar_mass_kg=COMPOSITE_DENSITY_KG_M3 * sum(s.volume for s in spars),
    )
