"""Plain helpers shared by the CAD and fabrication tests (no fixtures here; those live in conftest.py)."""

from pathlib import Path
from typing import Iterable

import build123d as bd

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import FabricationSettings
from src.schemas.state import AeroStabilityOutput, SizingOutput
from src.tools.aero_cad import components
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing
from src.tools.fabrication.kit import KitPart, TubePart
from src.tools.fabrication.kit_pod import kit_fuselage_diameter_m


def _synthetic_aero(layout: AircraftLayout, np_offset_mac: float = 0.25) -> AeroStabilityOutput:
    """Stand-in for VSPAERO so packing can be tested without OpenVSP."""
    return AeroStabilityOutput(
        cl0=0.3, cl_alpha_per_rad=5.0, cd0=0.03, ld_cruise=15.0, cm_alpha_per_rad=-1.2, cn_beta_per_rad=0.08,
        cl_beta_per_rad=-0.05, cl_delta_a_per_rad=0.2, cm_delta_e_per_rad=-1.0, cn_delta_r_per_rad=-0.05,
        x_np_m=layout.wing.x_ac_m + np_offset_mac * layout.wing.mac_m,
    )


def _fitted_layout(sizing: SizingOutput, bom: SelectedBOM, station: float = 0.30,
                   with_propeller: bool = False) -> AircraftLayout:
    """The pod as build.py sizes it; `with_propeller` places the BOM's propeller (a boom-end pusher needs it)."""
    diameter: float = max(sizing.fuselage_diameter_m, kit_fuselage_diameter_m(sizing, bom))  # as build.py sizes it
    propeller: float = bom.cruise_propeller.diameter_m if with_propeller else 0.0
    return layout_from_sizing(sizing, station, diameter, components.required_fuselage_length_m(sizing, bom, diameter),
                              propeller_diameter_m=propeller)


def _synthetic_oml(layout: AircraftLayout, path: Path) -> Path:
    """Fuselage cylinder + wing slab exported in metres, like OpenVSP's trimmed STEP."""
    fus = bd.Pos(layout.fuselage_length_m / 2, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(
        layout.fuselage_diameter_m / 2, layout.fuselage_length_m)
    w = layout.wing
    wing = bd.Pos(w.x_le_root_m + w.root_chord_m / 2, 0, w.z_root_m) * bd.Box(
        w.root_chord_m, 2 * w.semi_span_m, 0.1 * w.root_chord_m)
    bd.export_step(fus + wing, str(path), unit=bd.Unit.M)
    return path


def hole_centre(wire: bd.Wire) -> bd.Vector:
    """Centre of a circular hole wire."""
    return wire.edges()[0].arc_center


def hole_diameters(face: bd.Face) -> list[float]:
    """Diameters of the circular holes in `face`, ordered fore to aft (local x)."""
    holes = sorted(face.inner_wires(), key=lambda w: hole_centre(w).X)
    return [2.0 * w.edges()[0].radius for w in holes]


def ribs_by_label(parts: Iterable[KitPart]) -> dict[str, KitPart]:
    return {p.label: p for p in parts}


def part_solid(part: KitPart, settings: FabricationSettings) -> bd.Solid:
    """The part as built: outline extruded along local +Z by its sheet thickness, then moved by `location`."""
    thickness = settings.sheet_for(part.family).thickness_m
    return bd.extrude(part.outline, amount=thickness, dir=(0, 0, 1)).moved(part.location)


def tube_solid(tube: TubePart) -> bd.Solid:
    """The tube as built: a hollow (or, for a rod, solid) cylinder from `start_m` along `axis` for `length_m`."""
    plane = bd.Plane(origin=tube.start_m, z_dir=tube.axis)
    outer = bd.Solid.make_cylinder(tube.tube.od_m / 2.0, tube.length_m, plane)
    if tube.tube.id_m <= 0.0:
        return outer
    return outer.cut(bd.Solid.make_cylinder(tube.tube.id_m / 2.0, tube.length_m, plane)).solids()[0]
