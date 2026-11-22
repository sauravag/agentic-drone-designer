"""Mass items of a laser-cut kit (SI, aircraft frame: x aft, y right, z up; tensor convention as in
`mass_properties`).

- Each `KitPart`: a slab of its sheet (area x thickness x density), CG at mid-thickness over the outline's
  centroid. Its inertia is exact for a prism: the outline's areal second moments (OCC surface properties) x
  rho t, plus m t^2 / 12 about the two in-plane axes, rotated into the aircraft frame by the part's location.
- Each `TubePart`: a hollow cylinder (annulus area x length x carbon density) along its own axis.
- Each `LinearStock`: a slender rod (section x length x density).
- Each `CoverSurface`: a uniform shell over its surface carrying areal density x covered area.
- `OW-1`: the centreline overwrap (its own mass, not carbon density), one hollow cylinder per socket half.
- `glue_hardware`: glue_fraction x the structure (parts, tubes, linear stock and the glued balsa D-box; not film or
  fuselage cover, nor the wet-laid overwrap, which is its own resin), distributed like the structure (its CG, and
  the same fraction of its inertia).
"""

from typing import Final, Sequence

import build123d as bd
import numpy as np

from src.schemas.fabrication import FabricationSettings
from src.schemas.state import SizingOutput
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.layout import AircraftLayout
from src.tools.fabrication.kit import OVERWRAP_LABEL, CoverKind, CoverSurface, Kit, KitPart, LinearStock, TubePart
from src.tools.gear import gear_mass_kg

GLUE_NAME: Final[str] = "glue_hardware"
GEAR_NAME: Final[str] = "landing_gear"
PRISM_INERTIA_DIVISOR: Final[float] = 12.0
CYLINDER_AXIAL_DIVISOR: Final[float] = 2.0
TRANSVERSE_RADIUS_FACTOR: Final[float] = 3.0
IN_PLANE: Final[np.ndarray] = np.diag([1.0, 1.0, 0.0])  # local axes that see the slab thickness


def _rotation(location: bd.Location) -> np.ndarray:
    trsf = location.wrapped.Transformation()
    return np.array([[trsf.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])


def _row(row: np.ndarray) -> mp.Vec3:
    return float(row[0]), float(row[1]), float(row[2])


def _mat(a: np.ndarray) -> mp.Mat3:
    return _row(a[0]), _row(a[1]), _row(a[2])


def part_item(part: KitPart, settings: FabricationSettings) -> mp.MassItem:
    """Slab of the part's sheet over its outline, placed by its location (times its quantity)."""
    sheet = settings.sheet_for(part.family)
    t = sheet.thickness_m
    face = mp.shell_item(part.label, part.outline, sheet.density_kg_m3 * t)
    local = np.array(face.inertia_kgm2) + face.mass_kg * t**2 / PRISM_INERTIA_DIVISOR * IN_PLANE
    rotation = _rotation(part.location)
    cx, cy, _ = face.cg_m
    cg = (part.location * bd.Location((cx, cy, t / 2.0))).position
    q = part.quantity
    return mp.MassItem(part.label, q * face.mass_kg, (cg.X, cg.Y, cg.Z), _mat(q * rotation @ local @ rotation.T))


def tube_item(name: str, mass_kg: float, start_m: Sequence[float], axis: Sequence[float], length_m: float,
              od_m: float, id_m: float) -> mp.MassItem:
    """Hollow cylinder from `start_m` along `axis` (any direction); od = id = 0 gives a slender rod."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    r2 = (od_m / 2.0) ** 2 + (id_m / 2.0) ** 2
    axial = mass_kg * r2 / CYLINDER_AXIAL_DIVISOR
    transverse = mass_kg * (TRANSVERSE_RADIUS_FACTOR * r2 + length_m**2) / PRISM_INERTIA_DIVISOR
    along = np.outer(a, a)
    tensor = transverse * (np.eye(3) - along) + axial * along
    cg = np.asarray(start_m, dtype=float) + a * length_m / 2.0
    return mp.MassItem(name, mass_kg, (float(cg[0]), float(cg[1]), float(cg[2])), _mat(tensor))


def _tube(tube: TubePart, settings: FabricationSettings) -> mp.MassItem:
    mass = tube.tube.area_m2 * tube.length_m * settings.carbon_density_kg_m3
    return tube_item(tube.label, mass, tube.start_m, tube.axis, tube.length_m, tube.tube.od_m, tube.tube.id_m)


def _strip(strip: LinearStock) -> mp.MassItem:
    mass = strip.section_area_m2 * strip.length_m * strip.density_kg_m3
    axis = np.subtract(strip.end_m, strip.start_m)
    return tube_item(strip.label, mass, strip.start_m, axis, strip.length_m, 0.0, 0.0)


def areal_density_kg_m2(kind: CoverKind, settings: FabricationSettings) -> float:
    """Mass per material area of each covering kind; the D-box is balsa sheet of the kit's stringer wood."""
    densities: dict[CoverKind, float] = {
        "film": settings.film_kg_m2,
        "fuselage": settings.fuselage_cover_kg_m2,
        "dbox": settings.dbox_sheet_thickness_m * settings.stringer_density_kg_m3,
    }
    return densities[kind]


def _cover(cover: CoverSurface, settings: FabricationSettings) -> mp.MassItem:
    """`area_m2` of covering spread uniformly over the surface (film: both faces over one planform)."""
    kg_m2 = areal_density_kg_m2(cover.kind, settings)
    return mp.shell_item(cover.label, cover.surface, kg_m2 * cover.area_m2 / cover.surface.area)


def overwrap_item(kit: Kit) -> mp.MassItem:
    """`OW-1`: the overwrap mass split evenly over its halves (`Kit.overwrap_halves`), combined."""
    halves = kit.overwrap_halves()
    share = kit.overwrap.mass_kg / len(halves)
    pieces = [tube_item(OVERWRAP_LABEL, share, h.start_m, h.axis, h.length_m, h.tube.od_m, h.tube.id_m) for h in halves]
    whole = mp.combine(pieces)
    return mp.MassItem(OVERWRAP_LABEL, whole.mass_kg, whole.cg_m, whole.inertia_kgm2)


def _glue(structure: Sequence[mp.MassItem], settings: FabricationSettings) -> mp.MassItem:
    whole = mp.combine(structure)
    f = settings.glue_fraction
    return mp.MassItem(GLUE_NAME, f * whole.mass_kg, whole.cg_m, _mat(f * np.array(whole.inertia_kgm2)))


def kit_mass_items(kit: Kit, settings: FabricationSettings) -> list[mp.MassItem]:
    """Parts, tubes, linear stock, the overwrap, covering, then glue/hardware (module docstring)."""
    structure = [part_item(p, settings) for p in kit.parts]
    structure += [_tube(t, settings) for t in kit.tubes]
    structure += [_strip(s) for s in kit.linear]
    covering = [_cover(c, settings) for c in kit.covering]
    glued_wood = [item for cover, item in zip(kit.covering, covering) if cover.kind == "dbox"]  # glued, not ironed on
    return structure + [overwrap_item(kit)] + covering + [_glue(structure + glued_wood, settings)]


def kit_mass_kg(kit: Kit, settings: FabricationSettings) -> float:
    return sum(item.mass_kg for item in kit_mass_items(kit, settings))


def residual_gear_item(sizing: SizingOutput, layout: AircraftLayout) -> mp.MassItem:
    """Landing gear the kit does not cut: the shared gear mass spread over the pod as a solid cylinder."""
    length, diameter = layout.fuselage_length_m, layout.fuselage_diameter_m
    mass = gear_mass_kg(sizing.airframe_topology, sizing.mtow_kg)
    return mp.cylinder_item(GEAR_NAME, mass, (length / 2.0, 0.0, 0.0), length, diameter)
