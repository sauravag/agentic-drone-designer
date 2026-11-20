"""The fuselage skin radius over a stretch of the pod, for the kit's parts (SI, x aft from the nose).

The skin radius r(x) is piecewise linear between the FUSELAGE_PROFILE break stations, so over any [x0, x1] its least
and greatest values lie at the ends or at a break station inside. Shared by `wing_parts` (tail roots) and
`fuselage_parts` (formers, plates, the crutch), which cannot import each other both ways.
"""

from src.tools.aero_cad.layout import FUSELAGE_PROFILE, AircraftLayout, fuselage_radius_at


def profile_corners(layout: AircraftLayout, x0_m: float, x1_m: float) -> list[float]:
    """Fuselage-profile break stations strictly inside (x0, x1): r(x) is linear between them."""
    length = layout.fuselage_length_m
    return [f * length for f, _ in FUSELAGE_PROFILE if x0_m < f * length < x1_m]


def _radii(layout: AircraftLayout, x0_m: float, x1_m: float) -> list[float]:
    return [fuselage_radius_at(layout, x) for x in (x0_m, x1_m, *profile_corners(layout, x0_m, x1_m))]


def max_skin_radius_m(layout: AircraftLayout, x0_m: float, x1_m: float) -> float:
    """Largest skin radius over [x0, x1]."""
    return max(_radii(layout, x0_m, x1_m))


def min_skin_radius_m(layout: AircraftLayout, x0_m: float, x1_m: float) -> float:
    """Smallest skin radius over [x0, x1]."""
    return min(_radii(layout, x0_m, x1_m))
