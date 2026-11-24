"""Airframe layout specs: the structural choices that differ between layouts (tail, booms, propeller, wing)."""

from dataclasses import dataclass
from typing import Final, Literal


BoomName = Literal["tail_boom", "lift_booms"]


@dataclass(frozen=True)
class BoomSet:
    """`count` identical round boom tubes; `name` is the empty-mass breakdown key they are booked under. Shared by the
    mass model (tube mass) and the aero model (boom drag)."""

    count: int
    length_m: float
    diameter_m: float
    name: BoomName


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


class PusherBoomClearanceError(ValueError):
    """A boom-end pusher's tail plus propeller gap ends ahead of the tail-boom root (a pod floored longer than its
    sized length, at a forward wing station): a clearance failure of that arrangement, not a missing boom."""


TAILED_BAND: Final[StaticMarginBand] = StaticMarginBand(0.08, 0.14, 0.11)
TAIL_ARM_FRACTION: Final[float] = 0.6  # of fuselage length, centre boom
HTAIL_ASPECT_RATIO: Final[float] = 4.0  # span^2 / area of the horizontal tail
HTAIL_TAPER_RATIO: Final[float] = 0.6
VTAIL_ASPECT_RATIO: Final[float] = 1.5  # height^2 / area of the single fin
VTAIL_TAPER_RATIO: Final[float] = 0.6
VTAIL_PAIR_ASPECT_RATIO: Final[float] = 4.0  # V-tail: tip-to-tip span^2 / total area of both panels (spec band 3-6)
# Boom-end pusher: axial gap from the tail trailing edge to the boom end, and boom end to propeller hub (motor plus
# its mount), as fractions of the propeller diameter. The mount length covers every catalog outrunner (<= 0.12 D).
PUSHER_TAIL_GAP_FRACTION: Final[float] = 0.10
PUSHER_MOTOR_MOUNT_FRACTION: Final[float] = 0.15

# Shared by the sizing boom estimate and the CAD layout (aero_cad.layout re-imports them).
WING_TAPER_RATIO: Final[float] = 0.5
QUARTER_CHORD: Final[float] = 0.25
DEFAULT_WING_STATION: Final[float] = 0.30  # wing-root LE position, fraction of fuselage length
TAIL_BOOM_START_FRACTION: Final[float] = 0.9  # boom root buried in the tail cone
TAIL_BOOM_DIAMETER_FRACTION: Final[float] = 0.25  # of fuselage diameter
MIN_BOOM_DIAMETER_M: Final[float] = 0.02
ROTOR_CLEARANCE_FRACTION: Final[float] = 0.10  # gap to wing/fuselage, fraction of rotor diameter
VTOL_BOOM_DIAMETER_FRACTION: Final[float] = 0.08  # of rotor diameter
VTOL_BOOM_OVERHANG_FRACTION: Final[float] = 0.15  # boom extension past each rotor hub

_SPECS: Final[dict[str, LayoutSpec]] = {
    "pod_boom_tractor": LayoutSpec("conventional", "centre", "tractor_nose", "straight", "tail_volume_h",
                                   TAILED_BAND),
    "single_boom_vtail_pusher": LayoutSpec("v", "centre", "pusher_boom_end", "straight", "tail_volume_h",
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


def pusher_tail_gap_m(propeller_diameter_m: float) -> float:
    """Tail trailing edge to tail-boom end for a boom-end pusher (read at call time, so tests can patch it)."""
    return PUSHER_TAIL_GAP_FRACTION * propeller_diameter_m


def pusher_mount_length_m(propeller_diameter_m: float) -> float:
    """Tail-boom end to propeller hub for a boom-end pusher: the motor and its mount."""
    return PUSHER_MOTOR_MOUNT_FRACTION * propeller_diameter_m


def has_tail_boom(spec: LayoutSpec, boom_start_m: float, boom_end_m: float, fuselage_length_m: float) -> bool:
    """Whether the centre tail boom is built (sizing books it, the CAD layout places it): a boom-end pusher's always,
    since it carries the motor (one ending ahead of its root raises PusherBoomClearanceError), otherwise only when the tail reaches past
    the pod."""
    if spec.propeller != "pusher_boom_end":
        return boom_end_m > fuselage_length_m
    if boom_end_m <= boom_start_m:
        raise PusherBoomClearanceError(f"the boom-end pusher's boom would end at {boom_end_m:.3f} m, ahead of its root at "
                         f"{boom_start_m:.3f} m")
    return True


def trapezoid_root_chord_m(area_m2: float, span_m: float, taper_ratio: float) -> float:
    """Root chord of a straight trapezoid of `area_m2` over `span_m` (tip-to-tip, or the fin height)."""
    return 2.0 * area_m2 / (span_m * (1.0 + taper_ratio))


def trapezoid_mac_m(root_chord_m: float, taper_ratio: float) -> float:
    """Mean aerodynamic chord of a straight trapezoid."""
    lam: float = taper_ratio
    return 2.0 / 3.0 * root_chord_m * (1.0 + lam + lam**2) / (1.0 + lam)


def tail_boom_diameter_m(fuselage_diameter_m: float) -> float:
    """Centre tail-boom tube diameter."""
    return max(MIN_BOOM_DIAMETER_M, TAIL_BOOM_DIAMETER_FRACTION * fuselage_diameter_m)


def lift_rotor_half_spacing_m(rotor_diameter_m: float, wing_root_chord_m: float, wing_x_le_m: float,
                              wing_x_ac_m: float) -> float:
    """Distance from the wing AC to each lift-rotor hub along a boom: the fore and aft discs clear the wing root
    by the rotor clearance."""
    radius: float = rotor_diameter_m / 2
    gap: float = ROTOR_CLEARANCE_FRACTION * rotor_diameter_m
    return max(wing_x_ac_m - (wing_x_le_m - radius - gap),
               wing_x_le_m + wing_root_chord_m + radius + gap - wing_x_ac_m)


def lift_boom_geometry(rotor_diameter_m: float, wing_root_chord_m: float, wing_x_le_m: float,
                       wing_x_ac_m: float) -> tuple[float, float]:
    """(length, diameter) of each of the two lift booms: centred on the wing AC, carrying a rotor at each end of the
    hub spacing, plus an overhang past each hub."""
    half: float = lift_rotor_half_spacing_m(rotor_diameter_m, wing_root_chord_m, wing_x_le_m, wing_x_ac_m)
    overhang: float = VTOL_BOOM_OVERHANG_FRACTION * rotor_diameter_m
    diameter: float = max(MIN_BOOM_DIAMETER_M, VTOL_BOOM_DIAMETER_FRACTION * rotor_diameter_m)
    return 2 * (half + overhang), diameter
