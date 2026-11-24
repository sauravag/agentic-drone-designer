"""Inputs, limits and the error of the propulsion matching search (spec 6.3-6.4)."""

import math
from dataclasses import dataclass
from typing import Final, Optional

from src.schemas.assumptions import DesignAssumptions
from src.tools.database.catalog_variants import ESC_CURRENT_MARGIN  # noqa: F401  (re-exported: one value)
from src.tools.sizing.errors import SizingDivergenceError

PROP_TIP_MACH_MAX: Final[float] = 0.70  # noise and compressibility (controller's call)
MAX_MATCH_THROTTLE: Final[float] = 0.85  # at the nominal pack voltage
MAX_SAGGED_THROTTLE: Final[float] = 1.0  # at the end-of-reserve loaded voltage (climb and hover points)
CHEMISTRIES: Final[tuple[str, ...]] = ("li_ion", "lipo")
MAX_POINT_NAME: Final[int] = 60


class PropulsionMatchError(SizingDivergenceError):
    """No catalogue set meets every matching rule; the message names the limiting constraint(s)."""


def _require(value: float, name: str, low: float, low_open: bool = True, high: float = math.inf) -> None:
    ok_low: bool = value > low if low_open else value >= low
    if not (math.isfinite(value) and ok_low and value <= high):
        raise ValueError(f"{name} must be finite and in {'(' if low_open else '['}{low}, {high}], got {value!r}")


@dataclass(frozen=True)
class FlightPoint:
    """One flight point the propulsion must meet. `thrust_n` is per propulsor: the cruise propulsor for forward
    points, ONE lift rotor for hover points. `check_sagged` re-checks the point at the end-of-reserve voltage."""

    name: str
    is_hover: bool
    airspeed_ms: float
    rho: float
    mu: float
    speed_of_sound_ms: float
    thrust_n: float
    duration_s: float
    check_sagged: bool

    def __post_init__(self) -> None:
        if not 0 < len(self.name) <= MAX_POINT_NAME:
            raise ValueError(f"flight point name must be 1-{MAX_POINT_NAME} characters, got {self.name!r}")
        _require(self.airspeed_ms, "airspeed_ms", 0.0, low_open=False)
        for name in ("rho", "mu", "speed_of_sound_ms", "thrust_n"):
            _require(getattr(self, name), name, 0.0)
        _require(self.duration_s, "duration_s", 0.0, low_open=False)
        if self.is_hover and self.airspeed_ms != 0.0:
            raise ValueError(f"hover point {self.name!r} must have zero airspeed, got {self.airspeed_ms}")


@dataclass(frozen=True)
class MatchContext:
    """Everything the search needs from the current sizing pass. `chemistry` None lets the search choose."""

    points: tuple[FlightPoint, ...]
    chemistry: Optional[str]
    is_hybrid: bool
    lift_rotor_count: int
    min_lift_rotor_diameter_m: float
    max_cruise_prop_diameter_m: float
    require_folding_prop: bool
    pusher_factor: float
    hotel_w: float
    assumptions: DesignAssumptions

    def __post_init__(self) -> None:
        names: list[str] = [p.name for p in self.points]
        if len(set(names)) != len(names):
            raise ValueError(f"flight point names must be unique, got {names}")
        if not self.forward_points:
            raise ValueError("matching needs at least one forward (cruise propulsor) flight point")
        if self.chemistry is not None and self.chemistry not in CHEMISTRIES:
            raise ValueError(f"chemistry must be one of {CHEMISTRIES} or None, got {self.chemistry!r}")
        if isinstance(self.lift_rotor_count, bool) or not isinstance(self.lift_rotor_count, int) \
                or self.lift_rotor_count < 0:
            raise ValueError(f"lift_rotor_count must be a non-negative integer, got {self.lift_rotor_count!r}")
        if bool(self.hover_points) != (self.lift_rotor_count > 0):
            raise ValueError("hover points need lift rotors and lift rotors need hover points "
                             f"(lift_rotor_count={self.lift_rotor_count}, hover points={len(self.hover_points)})")
        if self.lift_rotor_count:
            _require(self.min_lift_rotor_diameter_m, "min_lift_rotor_diameter_m", 0.0)
        _require(self.max_cruise_prop_diameter_m, "max_cruise_prop_diameter_m", 0.0)
        _require(self.pusher_factor, "pusher_factor", 0.0, high=1.0)
        _require(self.hotel_w, "hotel_w", 0.0, low_open=False)
        if self.mission_duration_s <= 0.0:
            raise ValueError("the flight points must have a positive total duration (mission energy)")

    @property
    def forward_points(self) -> tuple[FlightPoint, ...]:
        return tuple(p for p in self.points if not p.is_hover)

    @property
    def hover_points(self) -> tuple[FlightPoint, ...]:
        return tuple(p for p in self.points if p.is_hover)

    @property
    def mission_duration_s(self) -> float:
        return sum(p.duration_s for p in self.points)
