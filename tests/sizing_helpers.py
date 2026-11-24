"""Plain sizing helpers shared by the tests (no CAD stack imports, so sizing-only tests can use them)."""

from src.schemas.state import SizingOutput
from src.tools.gear import GearGeometry
from src.tools.sizing.layouts import layout_spec
from src.tools.sizing.mass_model import Geometry, sizing_gear


def sizing_output_gear(sizing: SizingOutput) -> GearGeometry:
    """`sizing_gear` for a finished design (the same geometry, read back from the record)."""
    geom = Geometry(
        wing_area_m2=sizing.wing_area_m2, aspect_ratio=sizing.aspect_ratio, wingspan_m=sizing.wingspan_m,
        mean_chord_m=sizing.mean_chord_m, fuselage_length_m=sizing.fuselage_length_m,
        fuselage_diameter_m=sizing.fuselage_diameter_m, horizontal_tail_area_m2=sizing.horizontal_tail_area_m2,
        vertical_tail_area_m2=sizing.vertical_tail_area_m2, tail_arm_m=sizing.tail_arm_m,
        tail_surfaces=sizing.tail_surfaces)
    return sizing_gear(geom, layout_spec(sizing.airframe_layout), sizing.launch_method)
