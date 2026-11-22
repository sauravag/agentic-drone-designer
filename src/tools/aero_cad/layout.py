"""3D general arrangement derived from the sizing output (SI, x aft from nose, y right, z up).

The layout is the single source of truth for positions: OpenVSP builds the OML from it, Build123d
places ribs/spars from it, and the packer locates components against it. Wing area, span, tail
areas and tail arm are taken verbatim from `SizingOutput` so the 3D model matches the sized design.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from src.schemas.state import SizingOutput, TailSurface
from src.tools.sizing.layouts import (
    HTAIL_ASPECT_RATIO,
    HTAIL_TAPER_RATIO,
    VTAIL_ASPECT_RATIO,
    VTAIL_TAPER_RATIO,
    layout_spec,
)

WING_TAPER_RATIO: Final[float] = 0.5
WING_DIHEDRAL_DEG: Final[float] = 3.0
WING_Z_FRACTION: Final[float] = 0.25  # wing root height above fuselage axis, fraction of diameter
DEFAULT_WING_STATION: Final[float] = 0.30  # wing-root LE position, fraction of fuselage length
WING_STATION_BOUNDS: Final[tuple[float, float]] = (0.10, 0.70)
WING_STATION_SCAN: Final[tuple[float, ...]] = (0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55)  # balance search
QUARTER_CHORD: Final[float] = 0.25
VERTICAL_DIHEDRAL_DEG: Final[float] = 90.0  # a tail surface at this dihedral is a fin


# Pod profile (x/L, d/D) between the nose and tail points: long constant section for internal bays.
FUSELAGE_PROFILE: Final[tuple[tuple[float, float], ...]] = ((0.15, 0.90), (0.25, 1.00), (0.80, 1.00), (0.95, 0.50))
CONSTANT_SECTION_FRACTION: Final[float] = 0.55  # full-diameter part of the pod (0.25 L - 0.80 L)
TAIL_BOOM_START_FRACTION: Final[float] = 0.9  # boom root buried in the tail cone
TAIL_BOOM_DIAMETER_FRACTION: Final[float] = 0.25  # of fuselage diameter
MIN_BOOM_DIAMETER_M: Final[float] = 0.02

# Control surfaces (shared by OpenVSP and the laser-cut kit): span fractions and chord fractions aft of the hinge.
AILERON_ETA: Final[tuple[float, float]] = (0.55, 0.95)
AILERON_CHORD_FRACTION: Final[float] = 0.25
TAIL_CONTROL_ETA: Final[tuple[float, float]] = (0.05, 0.95)
TAIL_CONTROL_CHORD_FRACTION: Final[float] = 0.35
TAIL_AIRFOIL_NAME: Final[str] = "naca0010"  # OpenVSP's default WING section, used by the tail surfaces

ROTOR_CLEARANCE_FRACTION: Final[float] = 0.10  # gap to wing/fuselage, fraction of rotor diameter
VTOL_BOOM_DIAMETER_FRACTION: Final[float] = 0.08  # of rotor diameter
VTOL_BOOM_OVERHANG_FRACTION: Final[float] = 0.15  # boom extension past each rotor hub


@dataclass(frozen=True)
class SurfacePlanform:
    """Straight-leading-edge trapezoidal lifting surface. `semi_span_m` is the fin height when
    `is_symmetric` is False (vertical tail)."""

    x_le_root_m: float
    z_root_m: float
    semi_span_m: float
    root_chord_m: float
    tip_chord_m: float
    dihedral_deg: float
    is_symmetric: bool
    sweep_deg: float = 0.0

    @property
    def taper_ratio(self) -> float:
        return self.tip_chord_m / self.root_chord_m

    @property
    def area_m2(self) -> float:
        sides: int = 2 if self.is_symmetric else 1
        return sides * 0.5 * (self.root_chord_m + self.tip_chord_m) * self.semi_span_m

    @property
    def mac_m(self) -> float:
        lam: float = self.taper_ratio
        return 2.0 / 3.0 * self.root_chord_m * (1 + lam + lam**2) / (1 + lam)

    @property
    def y_mac_m(self) -> float:
        lam: float = self.taper_ratio
        return self.semi_span_m / 3.0 * (1 + 2 * lam) / (1 + lam)

    @property
    def x_ac_m(self) -> float:
        """Aerodynamic centre: quarter-chord of the MAC (straight LE, so MAC LE = root LE)."""
        return self.x_le_root_m + QUARTER_CHORD * self.mac_m

    def chord_at(self, y_m: float) -> float:
        frac: float = min(abs(y_m) / self.semi_span_m, 1.0)
        return self.root_chord_m + (self.tip_chord_m - self.root_chord_m) * frac

    def z_at(self, y_m: float) -> float:
        return self.z_root_m + abs(y_m) * math.tan(math.radians(self.dihedral_deg))


@dataclass(frozen=True)
class Boom:
    x_start_m: float
    length_m: float
    diameter_m: float
    y_m: float
    z_m: float

    @property
    def x_end_m(self) -> float:
        return self.x_start_m + self.length_m


@dataclass(frozen=True)
class PlacedTailSurface:
    """A tail surface in the layout. `is_vertical` marks a fin standing on the axis (its planform dihedral
    is then 0: the OpenVSP wing is rotated 90 degrees about x instead)."""

    name: str
    control: str
    planform: SurfacePlanform
    is_vertical: bool = False


@dataclass(frozen=True)
class PropellerStation:
    """Where the cruise propeller sits (hub centre, fuselage axes). The diameter is added with its first reader."""

    x_m: float
    y_m: float
    z_m: float
    is_pusher: bool


@dataclass(frozen=True)
class AircraftLayout:
    fuselage_length_m: float
    fuselage_diameter_m: float
    wing: SurfacePlanform
    tail_surfaces: tuple[PlacedTailSurface, ...]
    tail_booms: tuple[Boom, ...]
    propeller: PropellerStation
    vtol_booms: tuple[Boom, ...]
    rotor_positions: tuple[tuple[float, float, float], ...]  # hub centres
    rotor_diameter_m: float
    airfoil: str
    wing_station: float

    def _tail_planform(self, name: str) -> SurfacePlanform:
        for surface in self.tail_surfaces:
            if surface.name == name:
                return surface.planform
        raise ValueError(f"layout has no tail surface named {name!r}; it has {[t.name for t in self.tail_surfaces]}")

    @property
    def htail(self) -> SurfacePlanform:
        return self._tail_planform("htail")

    @property
    def vtail(self) -> SurfacePlanform:
        return self._tail_planform("vtail")

    @property
    def tail_boom(self) -> Optional[Boom]:
        """The single centre boom, or None when the empennage sits on the fuselage."""
        return self.tail_booms[0] if self.tail_booms else None


def fuselage_radius_at(layout: AircraftLayout, x_m: float) -> float:
    """Skin radius at station x (piecewise-linear through FUSELAGE_PROFILE; 0 at nose and tail)."""
    stations = (0.0, *(f for f, _ in FUSELAGE_PROFILE), 1.0)
    ratios = (0.0, *(d for _, d in FUSELAGE_PROFILE), 0.0)
    frac: float = x_m / layout.fuselage_length_m
    if not 0.0 <= frac <= 1.0:
        return 0.0
    return 0.5 * layout.fuselage_diameter_m * float(np.interp(frac, stations, ratios))


def _main_wing(sizing: SizingOutput, wing_station: float, length: float, diameter: float) -> SurfacePlanform:
    semi: float = sizing.wingspan_m / 2
    root: float = 2 * sizing.wing_area_m2 / (sizing.wingspan_m * (1 + WING_TAPER_RATIO))
    return SurfacePlanform(
        x_le_root_m=wing_station * length, z_root_m=WING_Z_FRACTION * diameter,
        semi_span_m=semi, root_chord_m=root, tip_chord_m=WING_TAPER_RATIO * root,
        dihedral_deg=WING_DIHEDRAL_DEG, is_symmetric=True,
    )


def _tail_planform(area: float, full_span: float, taper: float, x_ac: float, symmetric: bool) -> SurfacePlanform:
    """Trapezoid of `area` and tip-to-tip `full_span` whose MAC quarter-chord lands on `x_ac`."""
    root: float = 2 * area / (full_span * (1 + taper))
    mac: float = 2.0 / 3.0 * root * (1 + taper + taper**2) / (1 + taper)
    return SurfacePlanform(
        x_le_root_m=x_ac - QUARTER_CHORD * mac, z_root_m=0.0,
        semi_span_m=full_span / 2 if symmetric else full_span, root_chord_m=root, tip_chord_m=taper * root,
        dihedral_deg=0.0, is_symmetric=symmetric,
    )


def _legacy_tail_surfaces(sizing: SizingOutput) -> tuple[TailSurface, ...]:
    """Tail of a record saved before `tail_surfaces` existed: H-tail and fin from the area fields."""
    h, v = sizing.horizontal_tail_area_m2, sizing.vertical_tail_area_m2
    arm: float = sizing.tail_arm_m
    return (
        TailSurface(name="htail", control="elevator", area_m2=h, span_m=math.sqrt(h * HTAIL_ASPECT_RATIO),
                    taper_ratio=HTAIL_TAPER_RATIO, dihedral_deg=0.0, arm_m=arm, is_symmetric=True),
        TailSurface(name="vtail", control="rudder", area_m2=v, span_m=math.sqrt(v * VTAIL_ASPECT_RATIO),
                    taper_ratio=VTAIL_TAPER_RATIO, dihedral_deg=VERTICAL_DIHEDRAL_DEG, arm_m=arm, is_symmetric=False),
    )


def _place_tail(sizing: SizingOutput, wing: SurfacePlanform) -> tuple[PlacedTailSurface, ...]:
    surfaces = sizing.tail_surfaces or _legacy_tail_surfaces(sizing)
    placed: list[PlacedTailSurface] = []
    for t in surfaces:
        is_vertical: bool = t.dihedral_deg == VERTICAL_DIHEDRAL_DEG
        planform = _tail_planform(t.area_m2, t.span_m, t.taper_ratio, wing.x_ac_m + t.arm_m, t.is_symmetric)
        if not is_vertical:
            planform = dataclasses.replace(planform, dihedral_deg=t.dihedral_deg)
        placed.append(PlacedTailSurface(t.name, t.control, planform, is_vertical))
    return tuple(placed)


def _tail_booms(length: float, diameter_fus: float, tail: tuple[PlacedTailSurface, ...]) -> tuple[Boom, ...]:
    start: float = TAIL_BOOM_START_FRACTION * length
    end: float = max(t.planform.x_le_root_m + t.planform.root_chord_m for t in tail)
    if end <= length:
        return ()  # empennage fits on the fuselage itself
    diameter: float = max(MIN_BOOM_DIAMETER_M, TAIL_BOOM_DIAMETER_FRACTION * diameter_fus)
    return (Boom(x_start_m=start, length_m=end - start, diameter_m=diameter, y_m=0.0, z_m=0.0),)


def _propeller_station(sizing: SizingOutput) -> PropellerStation:
    """Hub on the fuselage axis; the spec decides nose or pusher placement."""
    if layout_spec(sizing.airframe_layout).propeller != "tractor_nose":
        raise ValueError(f"no propeller placement for layout {sizing.airframe_layout!r} yet")
    return PropellerStation(x_m=0.0, y_m=0.0, z_m=0.0, is_pusher=False)


def _vtol_arrangement(sizing: SizingOutput, wing: SurfacePlanform, diameter_fus: float) -> tuple[tuple[Boom, ...], tuple]:
    """Two longitudinal booms carrying four lift rotors, centred on the wing AC so the hover thrust
    centroid sits near the CG, with every disc clear of the wing and fuselage."""
    radius: float = sizing.rotor_diameter_m / 2
    gap: float = ROTOR_CLEARANCE_FRACTION * sizing.rotor_diameter_m
    y_boom: float = diameter_fus / 2 + radius + gap
    x_mid: float = wing.x_ac_m
    half: float = max(x_mid - (wing.x_le_root_m - radius - gap), wing.x_le_root_m + wing.root_chord_m + radius + gap - x_mid)
    boom_d: float = max(MIN_BOOM_DIAMETER_M, VTOL_BOOM_DIAMETER_FRACTION * sizing.rotor_diameter_m)
    overhang: float = VTOL_BOOM_OVERHANG_FRACTION * sizing.rotor_diameter_m
    z_hub: float = wing.z_at(y_boom)
    booms = tuple(
        Boom(x_start_m=x_mid - half - overhang, length_m=2 * (half + overhang), diameter_m=boom_d,
             y_m=side * y_boom, z_m=z_hub - boom_d / 2)
        for side in (1.0, -1.0)
    )
    rotors = tuple((x_mid + sign * half, side * y_boom, z_hub) for side in (1.0, -1.0) for sign in (-1.0, 1.0))
    return booms, rotors


def layout_from_sizing(sizing: SizingOutput, wing_station: float = DEFAULT_WING_STATION,
                       min_fuselage_diameter_m: float = 0.0, min_fuselage_length_m: float = 0.0) -> AircraftLayout:
    """General arrangement for `sizing`. `wing_station` slides the wing+tail group along the
    fuselage; the `min_fuselage_*` floors grow the pod so the internal COTS parts fit (SRS 4.3: the
    pod is sized to house payload, tank and avionics). Tail arm stays the sized absolute value."""
    lo, hi = WING_STATION_BOUNDS
    if not lo <= wing_station <= hi:
        raise ValueError(f"wing_station {wing_station:.3f} must be within [{lo}, {hi}] of fuselage length")
    length: float = max(sizing.fuselage_length_m, min_fuselage_length_m)
    diameter: float = max(sizing.fuselage_diameter_m, min_fuselage_diameter_m)
    wing: SurfacePlanform = _main_wing(sizing, wing_station, length, diameter)
    tail = _place_tail(sizing, wing)
    is_quad: bool = sizing.airframe_topology == "quadplane_vtol"
    booms, rotors = _vtol_arrangement(sizing, wing, diameter) if is_quad else ((), ())
    return AircraftLayout(
        fuselage_length_m=length, fuselage_diameter_m=diameter,
        wing=wing, tail_surfaces=tail, tail_booms=_tail_booms(length, diameter, tail),
        propeller=_propeller_station(sizing),
        vtol_booms=booms, rotor_positions=rotors, rotor_diameter_m=sizing.rotor_diameter_m if is_quad else 0.0,
        airfoil=sizing.airfoil, wing_station=wing_station,
    )
