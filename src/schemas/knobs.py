"""The sizing knobs the critic (and a designer's change set) may move: defaults and physical bounds.

A leaf module (no imports from src.tools / src.agent) so schemas can use it without import cycles."""

from typing import Final

DEFAULT_ASPECT_RATIO: Final[float] = 12.0
DEFAULT_TAIL_VOLUME_H: Final[float] = 0.50
DEFAULT_TAIL_VOLUME_V: Final[float] = 0.04
DEFAULT_FUSELAGE_LENGTH_FACTOR: Final[float] = 1.0

OVERRIDE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "aspect_ratio": (4.0, 25.0),
    "tail_volume_h": (0.3, 1.2),
    "tail_volume_v": (0.02, 0.10),
    "fuselage_length_factor": (0.6, 2.5),
    # end-of-mission energy reserve: battery SoC (electric) or fuel fraction (series hybrid); default by powertrain
    "reserve_fraction": (0.10, 0.40),
}
OVERRIDE_DEFAULTS: Final[dict[str, float]] = {
    "aspect_ratio": DEFAULT_ASPECT_RATIO,
    "tail_volume_h": DEFAULT_TAIL_VOLUME_H,
    "tail_volume_v": DEFAULT_TAIL_VOLUME_V,
    "fuselage_length_factor": DEFAULT_FUSELAGE_LENGTH_FACTOR,
}
