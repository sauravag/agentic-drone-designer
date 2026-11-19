"""Plain helpers shared by the CAD and fabrication tests (no fixtures here; those live in conftest.py)."""

from pathlib import Path
from typing import Iterable

import build123d as bd

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import FabricationSettings
from src.schemas.state import AeroStabilityOutput, SizingOutput
from src.tools.aero_cad import components
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing
from src.tools.fabrication.kit import KitPart


def _synthetic_aero(layout: AircraftLayout, np_offset_mac: float = 0.25) -> AeroStabilityOutput:
    """Stand-in for VSPAERO so packing can be tested without OpenVSP."""
    return AeroStabilityOutput(
        cl0=0.3, cl_alpha_per_rad=5.0, cd0=0.03, ld_cruise=15.0, cm_alpha_per_rad=-1.2, cn_beta_per_rad=0.08,
        cl_beta_per_rad=-0.05, cl_delta_a_per_rad=0.2, cm_delta_e_per_rad=-1.0, cn_delta_r_per_rad=-0.05,
        x_np_m=layout.wing.x_ac_m + np_offset_mac * layout.wing.mac_m,
    )


def _fitted_layout(sizing: SizingOutput, bom: SelectedBOM, station: float = 0.30) -> AircraftLayout:
    diameter: float = max(sizing.fuselage_diameter_m, components.required_fuselage_diameter_m(sizing, bom))
    return layout_from_sizing(sizing, station, diameter, components.required_fuselage_length_m(sizing, bom, diameter))


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
