"""Airframe layout specs: the structural choices that differ between layouts (tail, booms, propeller, wing)."""

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True)
class StaticMarginBand:
    """Acceptable static margin (fraction of mean aerodynamic chord) and the value sizing aims for."""

    minimum: float
    maximum: float
    target: float


@dataclass(frozen=True)
class LayoutSpec:
    tail: Literal["conventional", "v", "inverted_v", "twin_fin_h", "none"]
    booms: Literal["centre", "twin", "none"]
    propeller: Literal["tractor_nose", "pusher_pod", "pusher_boom_end"]
    wing: Literal["straight", "swept_reflex"]
    static_margin_knob: Literal["tail_volume_h", "wing_sweep_deg"]
    static_margin: StaticMarginBand


TAILED_BAND: Final[StaticMarginBand] = StaticMarginBand(0.08, 0.14, 0.11)
TAIL_ARM_FRACTION: Final[float] = 0.6  # of fuselage length, centre boom
HTAIL_ASPECT_RATIO: Final[float] = 4.0  # span^2 / area of the horizontal tail
HTAIL_TAPER_RATIO: Final[float] = 0.6
VTAIL_ASPECT_RATIO: Final[float] = 1.5  # height^2 / area of the single fin
VTAIL_TAPER_RATIO: Final[float] = 0.6

_SPECS: Final[dict[str, LayoutSpec]] = {
    "pod_boom_tractor": LayoutSpec("conventional", "centre", "tractor_nose", "straight", "tail_volume_h",
                                   TAILED_BAND),
}


def layout_spec(layout: str) -> LayoutSpec:
    """The spec of a layout; ValueError for one that is not built yet."""
    spec = _SPECS.get(layout)
    if spec is None:
        raise ValueError(f"airframe layout {layout!r} is not built yet; available: {', '.join(sorted(_SPECS))}")
    return spec


def static_margin_band(layout: str) -> StaticMarginBand:
    """The static-margin band every reader (packer, critic, pipeline, flight-control cases) takes from the layout."""
    return layout_spec(layout).static_margin


def tail_arm_m(spec: LayoutSpec, fuselage_length_m: float) -> float:
    """Distance from the wing to the tail aerodynamic centre."""
    if spec.booms != "centre":
        raise ValueError(f"no tail-arm rule for {spec.booms!r} booms yet")
    return TAIL_ARM_FRACTION * fuselage_length_m
