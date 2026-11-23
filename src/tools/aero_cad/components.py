"""Discrete parts as placed solids: construction from the BOM, collision and skin-containment checks.

Arrangement (pod-and-boom): cruise motor at its propeller station (nose bulkhead for a tractor, tail-boom end for
a boom-end pusher), EO/IR gimbal in the chin bay
just aft (external turret, deliberately protruding), fuel tank centred on the neutral point,
generator directly aft of the tank, battery and avionics tray solved by the packer. Lift motors sit
on the VTOL booms. Booms are zero-mass obstacles (their mass is in the airframe shell).
"""

import math
from dataclasses import dataclass, replace
from typing import Final, Literal, Sequence

import build123d as bd
import numpy as np

from src.schemas.components import SelectedBOM
from src.schemas.state import SizingOutput
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.layout import CONSTANT_SECTION_FRACTION, AircraftLayout, fuselage_radius_at
from src.tools.sizing.mass_model import (
    FUEL_SYSTEM_FRACTION,
    SYSTEMS_FRACTION,
    VTOL_INSTALL_FACTOR,
)

FUEL_DENSITY_KG_M3: Final[float] = 720.0  # gasoline / avgas
TANK_ULLAGE_FACTOR: Final[float] = 1.10
TANK_DIAMETER_FRACTION: Final[float] = 0.80  # conformal tank, of fuselage diameter
AVIONICS_TRAY_DENSITY_KG_M3: Final[float] = 400.0
AVIONICS_TRAY_ASPECT: Final[tuple[float, float, float]] = (2.0, 1.5, 1.0)  # lx : ly : lz
NOSE_BAY_START_FRACTION: Final[float] = 0.04  # gimbal bay starts this far aft of the nose
COMPONENT_GAP_M: Final[float] = 0.01
RADIAL_FIT_MARGIN: Final[float] = 0.90  # usable fraction of the local skin radius (frames, skin, wiring)
COLLISION_VOLUME_TOL_M3: Final[float] = 1e-12
CONTAINMENT_TOL_FRACTION: Final[float] = 0.01  # of part volume allowed outside the OML (boolean noise)

Kind = Literal["box", "cylinder"]


@dataclass(frozen=True)
class PlacedComponent:
    name: str
    kind: Kind  # cylinders lie along x: size = (length, diameter, diameter)
    mass_kg: float
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    is_internal: bool = True  # must sit inside the fuselage skin

    @property
    def x_min(self) -> float:
        return self.center_m[0] - self.size_m[0] / 2

    @property
    def x_max(self) -> float:
        return self.center_m[0] + self.size_m[0] / 2

    @property
    def radial_extent_m(self) -> float:
        """Largest distance from the fuselage axis to any point of the part's cross-section."""
        _, ly, lz = self.size_m
        offset: float = math.hypot(self.center_m[1], self.center_m[2])
        return offset + (ly / 2 if self.kind == "cylinder" else math.hypot(ly, lz) / 2)

    def at_x(self, x_center: float) -> "PlacedComponent":
        return replace(self, center_m=(x_center, self.center_m[1], self.center_m[2]))

    def solid(self) -> bd.Solid:
        lx, ly, lz = self.size_m
        if self.kind == "cylinder":
            plane = bd.Plane(origin=(self.x_min, self.center_m[1], self.center_m[2]), z_dir=(1, 0, 0))
            return bd.Solid.make_cylinder(ly / 2, lx, plane)
        corner = (self.x_min, self.center_m[1] - ly / 2, self.center_m[2] - lz / 2)
        return bd.Solid.make_box(lx, ly, lz, bd.Plane(origin=corner))

    def mass_item(self) -> mp.MassItem:
        if self.kind == "cylinder":
            return mp.cylinder_item(self.name, self.mass_kg, self.center_m, self.size_m[0], self.size_m[1])
        return mp.box_item(self.name, self.mass_kg, self.center_m, self.size_m)


# ------------------------------------------------------------------------------ geometry checks ---
def overlap_volume_m3(a: bd.Shape, b: bd.Shape) -> float:
    """Volume of `a ∩ b`. build123d returns a Shape, a ShapeList (possibly empty) or None."""
    common = a.intersect(b)
    if common is None:
        return 0.0
    shapes = common if isinstance(common, (list, tuple)) else [common]
    return float(sum(s.volume for s in shapes))


def count_collisions(components: Sequence[PlacedComponent], obstacles: Sequence[PlacedComponent] = ()) -> int:
    """Overlapping pairs among components, and between components and structural obstacles
    (touching faces do not count)."""
    parts = [c.solid() for c in components]
    blocks = [o.solid() for o in obstacles]
    pairs = [(parts[i], parts[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))]
    pairs += [(p, b) for p in parts for b in blocks]
    return sum(1 for a, b in pairs if overlap_volume_m3(a, b) > COLLISION_VOLUME_TOL_M3)


def containment_violations(components: Sequence[PlacedComponent], oml: bd.Shape) -> int:
    """Internal parts with more than CONTAINMENT_TOL_FRACTION of their volume outside the OML."""
    violations: int = 0
    for c in components:
        if not c.is_internal:
            continue
        solid = c.solid()
        if solid.volume - overlap_volume_m3(solid, oml) > CONTAINMENT_TOL_FRACTION * solid.volume:
            violations += 1
    return violations


def fits_radially(item: PlacedComponent, layout: AircraftLayout) -> bool:
    """Cross-section fits the skin at both ends (the pod profile is unimodal, so the ends bound it)."""
    skin: float = min(fuselage_radius_at(layout, item.x_min), fuselage_radius_at(layout, item.x_max))
    return item.radial_extent_m <= RADIAL_FIT_MARGIN * skin


# --------------------------------------------------------------------------------------- parts ---
def payload(bom: SelectedBOM, layout: AircraftLayout) -> PlacedComponent:
    p = bom.payload
    x: float = NOSE_BAY_START_FRACTION * layout.fuselage_length_m + p.length_m / 2
    z: float = -layout.fuselage_diameter_m / 2 + p.height_m / 2  # chin turret: flush with the belly
    return PlacedComponent("payload", "box", p.mass_kg, (p.length_m, p.width_m, p.height_m), (x, 0.0, z), False)


def cruise_motor(bom: SelectedBOM, layout: AircraftLayout) -> PlacedComponent:
    """Cruise motor + propeller just ahead of the propeller station: a tractor's ahead of the nose bulkhead (x = 0),
    a boom-end pusher's between the boom end and the hub (the mount length leaves room for any catalog motor; a
    longer one runs into the boom and counts as a collision). The cruise ESC is not in the BOM and is omitted (a few
    % of motor mass)."""
    m = bom.cruise_motor
    station = layout.propeller
    mass: float = m.mass_kg + bom.cruise_propeller.mass_kg
    aft_face: float = station.x_m if station.is_pusher else station.x_m - COMPONENT_GAP_M
    return PlacedComponent("cruise_motor", "cylinder", mass, (m.length_m, m.diameter_m, m.diameter_m),
                           (aft_face - m.length_m / 2, station.y_m, station.z_m), False)


def hybrid_parts(sizing: SizingOutput, bom: SelectedBOM, layout: AircraftLayout, x_np: float) -> list[PlacedComponent]:
    """Conformal fuel tank centred on the neutral point and the generator directly aft of it."""
    if bom.ice_generator is None or sizing.fuel_mass_kg <= 0:
        return []
    diameter: float = TANK_DIAMETER_FRACTION * layout.fuselage_diameter_m
    length: float = _tank_length_m(sizing, layout.fuselage_diameter_m)
    tank_mass: float = sizing.fuel_mass_kg * (1 + FUEL_SYSTEM_FRACTION)
    tank = PlacedComponent("fuel_tank", "cylinder", tank_mass, (length, diameter, diameter), (x_np, 0.0, 0.0))
    g = bom.ice_generator
    x_gen: float = tank.x_max + COMPONENT_GAP_M + g.length_m / 2
    gen = PlacedComponent("generator", "box", g.mass_kg, (g.length_m, g.width_m, g.height_m), (x_gen, 0.0, 0.0))
    return [tank, gen]


def vtol_motors(bom: SelectedBOM, layout: AircraftLayout) -> list[PlacedComponent]:
    """Lift motor + propeller + ESC/mount allowance (the sizing VTOL_INSTALL_FACTOR) on each hub."""
    if bom.vtol_motor is None or bom.vtol_propeller is None:
        return []
    m = bom.vtol_motor
    install: float = max(0.0, (VTOL_INSTALL_FACTOR - 1) * m.mass_kg - bom.vtol_propeller.mass_kg)
    mass: float = m.mass_kg + bom.vtol_propeller.mass_kg + install
    return [
        PlacedComponent(f"vtol_motor_{i}", "box", mass, (m.diameter_m, m.diameter_m, m.length_m),
                        (x, y, z + m.length_m / 2), False)
        for i, (x, y, z) in enumerate(layout.rotor_positions)
    ]


def sliders(sizing: SizingOutput, bom: SelectedBOM) -> tuple[PlacedComponent, PlacedComponent]:
    """Battery and avionics tray on the fuselage axis; the packer solves their x positions."""
    b = bom.battery
    battery = PlacedComponent("battery", "box", b.mass_kg, (b.length_m, b.width_m, b.height_m), (0.0, 0.0, 0.0))
    tray_mass: float = sizing.avionics_mass_kg + SYSTEMS_FRACTION * sizing.mtow_kg  # the stack sizing used
    unit: float = (tray_mass / AVIONICS_TRAY_DENSITY_KG_M3 / float(np.prod(AVIONICS_TRAY_ASPECT))) ** (1 / 3)
    lx, ly, lz = (a * unit for a in AVIONICS_TRAY_ASPECT)
    return battery, PlacedComponent("avionics_tray", "box", tray_mass, (lx, ly, lz), (0.0, 0.0, 0.0))


def boom_obstacles(layout: AircraftLayout) -> list[PlacedComponent]:
    """Zero-mass solids for the tail boom (incl. its buried root) and the VTOL booms."""
    booms = [*layout.tail_booms, *layout.vtol_booms]
    return [
        PlacedComponent(f"boom_{i}", "cylinder", 0.0, (b.length_m, b.diameter_m, b.diameter_m),
                        (b.x_start_m + b.length_m / 2, b.y_m, b.z_m), False)
        for i, b in enumerate(booms)
    ]


def required_fuselage_diameter_m(sizing: SizingOutput, bom: SelectedBOM) -> float:
    """Smallest pod diameter whose constant section houses the internal COTS parts (the conformal
    tank scales with the pod, so it never sets the diameter)."""
    internal: list[PlacedComponent] = list(sliders(sizing, bom))
    gen = bom.ice_generator
    if gen is not None:
        internal.append(PlacedComponent("generator", "box", gen.mass_kg,
                                        (gen.length_m, gen.width_m, gen.height_m), (0.0, 0.0, 0.0)))
    return 2 * max(c.radial_extent_m for c in internal) / RADIAL_FIT_MARGIN


def _tank_length_m(sizing: SizingOutput, fuselage_diameter_m: float) -> float:
    diameter: float = TANK_DIAMETER_FRACTION * fuselage_diameter_m
    return sizing.fuel_mass_kg / FUEL_DENSITY_KG_M3 * TANK_ULLAGE_FACTOR / (math.pi * diameter**2 / 4)


def required_fuselage_length_m(sizing: SizingOutput, bom: SelectedBOM, fuselage_diameter_m: float) -> float:
    """Pod length whose constant section can hold the internal parts end to end."""
    lengths: list[float] = [c.size_m[0] for c in sliders(sizing, bom)]
    if bom.ice_generator is not None and sizing.fuel_mass_kg > 0:
        lengths += [bom.ice_generator.length_m, _tank_length_m(sizing, fuselage_diameter_m)]
    return (sum(lengths) + (len(lengths) + 1) * COMPONENT_GAP_M) / CONSTANT_SECTION_FRACTION
