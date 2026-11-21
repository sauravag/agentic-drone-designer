"""Stock-tube spar and socket sizing, and the tube structural checks, for the laser-cut airframe kit.

Picks carbon tubes from stock so the spar survives every load case at each station, stepping down telescopically
where the wing thins. A load case is a (bending moment, torque) pair; a tube passes when the interaction
FS * sqrt((sigma / sigma_allow)^2 + (tau / tau_allow)^2) <= 1 (R25), which with no torque is the plain bending rule:
- flight: the exact elliptic moment (R4) with the wing's aerodynamic torque (Cm0 plus the lift's offset from the
  spar, integrated from the tip inboard, at the manoeuvre dynamic pressure, R30);
- hover (quadplane): the boom reaction moment with the pitch-control torque of one rotor at full thrust and its
  fore/aft partner at zero, carried inboard of the boom clamp.
Also here: the centreline overwrap that carries the socket mitre (R24/R29), the tail-surface load (R35) that the
tail-boom check (R26) and the tail spars use, the loads each wing-box plate bears (R27/R32) and the anti-rotation pin
that reacts the spar's root torque (R31). All values SI. Must not import from the schemas' consumers.

Limitations:
- R30: flight loads use q at max(V_A, the loiter speed). SizingOutput now carries v_cruise_ms, but the strength
  torsion and tail loads still use that max, so a design cruising faster than V_A = V_stall sqrt(n_limit) has them
  under-estimated.
- R35: a tail surface's load is q_A S CL_max with no extra n_limit factor (CL_max at V_A is already the ceiling of
  tail lift). High-speed gust cases on the tails are not modelled and could exceed it.
"""

import math
from dataclasses import dataclass
from typing import Callable, Final, Optional, Sequence

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.schemas.state import SizingOutput
from src.tools.aero_cad.layout import MIN_BOOM_DIAMETER_M, QUARTER_CHORD, AircraftLayout
from src.tools.aero_cad.structure import RIB_PITCH_M, SPAR_CHORD_STATION
from src.tools.sizing.aero_model import air_properties
from src.units import G0

MIN_WEB_M: Final[float] = 0.002  # wood left above and below the spar tube
FIT_EPS_M: Final[float] = 1e-9  # float tolerance on diameter comparisons
RATIO_EPS: Final[float] = 1e-9  # float tolerance on the interaction ratio's limit of 1
HALF_WING_FRACTION: Final[float] = 0.5  # one wing carries half the weight
ELLIPTIC_MOMENT_FACTOR: Final[float] = 2.0 / (3.0 * math.pi)  # prefactor of the exact elliptic moment
ELLIPTIC_SQUARE_OFFSET: Final[float] = 2.0
ELLIPTIC_ACOS_FACTOR: Final[float] = 3.0
ELLIPTIC_LIFT_FACTOR: Final[float] = 4.0 / math.pi  # L'(y) = (4 P / pi b) sqrt(1 - xi^2) integrates to P
BISECTION_ITERATIONS: Final[int] = 50
HOVER_ARMS: Final[int] = 2  # two booms' thrust reacts through the spar per side
HOVER_SHARE: Final[float] = 0.25  # each motor carries a quarter of the weight
ROOT_SPAN_M: Final[float] = 0.0
BOOM_DEFLECTION_FRACTION: Final[float] = 0.01  # R22: lift-boom tip deflection limit, fraction of its arm
CANTILEVER_DEFLECTION_FACTOR: Final[float] = 3.0  # tip load: delta = P L^3 / (3 E I)
DYNAMIC_PRESSURE_FACTOR: Final[float] = 0.5  # q = rho V^2 / 2
SEA_LEVEL_DENSITY_KG_M3: Final[float] = air_properties(0.0)[0]  # SizingOutput.v_stall_ms is a sea-level speed
AIRFOIL_CM0_DEFAULT: Final[float] = -0.10  # no airfoil Cm0 in SizingOutput: conservative for cambered UAV sections
AC_CHORD_FRACTION: Final[float] = QUARTER_CHORD  # aerodynamic centre of the section
TORQUE_INTEGRATION_STEPS: Final[int] = 400  # midpoint rule, tip to station
TAIL_CL_MAX: Final[float] = 1.0  # tail surface (NACA 0010, low Re, finite span) maximum lift coefficient
WRAP_LAP_DIAMETERS: Final[float] = 2.0  # the overwrap laps at least 2 socket OD onto each side
WRAP_SIDES: Final[int] = 2  # the wrap laps both socket tubes
NOMINAL_PIN_ROD_M: Final[float] = 0.004  # anti-rotation pin rod before R31 steps it up
BOSS_THICKNESS_FACTOR: Final[float] = 2.0  # a boss doubler doubles the bearing thickness


class KitUnavailable(Exception):
    """The airframe cannot be built from stock; the message is user-facing."""


@dataclass(frozen=True)
class SparSegment:
    tube: TubeStock
    y_start_m: float
    y_end_m: float


@dataclass(frozen=True)
class SparPlan:
    segments: tuple[SparSegment, ...]
    socket: TubeStock
    joint_y_m: float
    insertion_m: float
    centre_moment_nm: float  # socket moment at the centreline (worse of flight and hover): sizes the overwrap
    pin: "PinPlan"  # anti-rotation pin sized for the root torque (R31)


@dataclass(frozen=True)
class PinPlan:
    """Anti-rotation pin (R31): rod diameter, the bearing load it carries (root torque / pin arm), and whether the
    wing-box plate and the wing root doubler need pin boss doublers."""

    rod_m: float
    load_n: float
    plate_boss: bool
    root_boss: bool


@dataclass(frozen=True)
class PlateLoads:
    """Loads a wing-box plate carries at the socket: the wing's shear in bearing (R27) and the downward load on the
    web under the socket (R32)."""

    shear_n: float
    down_n: float


NO_PLATE_LOADS: Final[PlateLoads] = PlateLoads(shear_n=0.0, down_n=0.0)  # geometry-only callers
NOMINAL_PIN: Final[PinPlan] = PinPlan(rod_m=NOMINAL_PIN_ROD_M, load_n=0.0, plate_boss=False, root_boss=False)


@dataclass(frozen=True)
class SparTorsion:
    """Spar torque sources (R25): the flight torque at a span station, and the hover pitch-control torque carried
    inboard of the boom clamp with the single full-thrust rotor that causes it."""

    flight_torque_at: Callable[[float], float]
    hover_torque_nm: float
    hover_thrust_n: float


@dataclass(frozen=True)
class Overwrap:
    """Wet-laid carbon cloth over the centreline socket mitre (R24): `layers` x `layer_thickness_m` on the socket
    OD, lapping `lap_m` onto each socket tube; `od_m` is the wrapped outer diameter."""

    layers: int
    layer_thickness_m: float
    lap_m: float
    od_m: float
    mass_kg: float


LoadCase = tuple[float, float]  # (bending moment N m, torque N m)


# ------------------------------------------------------------------------------------- stresses ---
def elliptic_root_moment_nm(lift_n: float, semi_span_m: float, y_m: float) -> float:
    """Bending moment at station y (from the panel's root line) of a panel carrying `lift_n` spread elliptically
    over its semi-span b: M = (2 P b / 3 pi) [(2 + xi^2) sqrt(1 - xi^2) - 3 xi acos(xi)], xi = y / b."""
    xi = min(max(y_m / semi_span_m, 0.0), 1.0)
    shape = (ELLIPTIC_SQUARE_OFFSET + xi**2) * math.sqrt(1.0 - xi**2) - ELLIPTIC_ACOS_FACTOR * xi * math.acos(xi)
    return ELLIPTIC_MOMENT_FACTOR * lift_n * semi_span_m * shape


def elliptic_moment_nm(mtow_kg: float, n: float, semi_span_m: float, y_m: float) -> float:
    """Wing bending moment at span station y (from the centreline): the elliptic moment of P = n W / 2."""
    return elliptic_root_moment_nm(n * mtow_kg * G0 * HALF_WING_FRACTION, semi_span_m, y_m)


def hover_moment_nm(mtow_kg: float, arm_m: float) -> float:
    """Moment from the motors' thrust on one side acting at the given arm."""
    return HOVER_ARMS * (mtow_kg * G0 * HOVER_SHARE) * max(arm_m, 0.0)


def hover_pitch_torque_nm(thrust_n: float, arm_m: float) -> float:
    """Spar torque when one lift rotor gives `thrust_n` at a fore/aft `arm_m` from the spar and its partner none."""
    return thrust_n * abs(arm_m)


def bending_stress_pa(tube: TubeStock, moment_nm: float) -> float:
    """Outer-fibre bending stress of a tube under a moment."""
    return moment_nm * (tube.od_m / 2.0) / tube.second_moment_m4


def torsion_stress_pa(tube: TubeStock, torque_nm: float) -> float:
    """Outer-fibre shear stress of a tube under a torque: tau = T (OD / 2) / J."""
    return abs(torque_nm) * (tube.od_m / 2.0) / tube.polar_moment_m4


def interaction_ratio(sigma_pa: float, tau_pa: float, settings: FabricationSettings) -> float:
    """FS sqrt((sigma / sigma_allow)^2 + (tau / tau_allow)^2): at most 1 passes (R25)."""
    return settings.safety_factor * math.hypot(sigma_pa / settings.sigma_allow_pa,
                                               tau_pa / settings.carbon_shear_allow_pa)


def interaction_ok(sigma_pa: float, tau_pa: float, settings: FabricationSettings) -> bool:
    return interaction_ratio(sigma_pa, tau_pa, settings) <= 1.0 + RATIO_EPS


def tube_ratio(tube: TubeStock, cases: tuple[LoadCase, ...], settings: FabricationSettings) -> float:
    """Governing interaction ratio of a tube over its load cases."""
    return max(interaction_ratio(bending_stress_pa(tube, m), torsion_stress_pa(tube, t), settings) for m, t in cases)


def _passes(tube: TubeStock, cases: tuple[LoadCase, ...], settings: FabricationSettings) -> bool:
    return tube_ratio(tube, cases, settings) <= 1.0 + RATIO_EPS


# -------------------------------------------------------------------------------- flight torque ---
def manoeuvre_dynamic_pressure_pa(v_stall_ms: float, settings: FabricationSettings) -> float:
    """q_max at the manoeuvre speed V_A = V_stall sqrt(n_limit), the full-deflection-at-CL_max case (R30)."""
    v_a = v_stall_ms * math.sqrt(settings.n_limit)
    return DYNAMIC_PRESSURE_FACTOR * SEA_LEVEL_DENSITY_KG_M3 * v_a**2


def flight_dynamic_pressure_pa(sizing: SizingOutput, settings: FabricationSettings) -> float:
    """q_max for flight torsion and tail loads: at V_A (R30), or at the loiter speed if that is faster (module
    docstring: SizingOutput has no cruise or dash speed)."""
    loiter = DYNAMIC_PRESSURE_FACTOR * SEA_LEVEL_DENSITY_KG_M3 * sizing.v_loiter_ms**2
    return max(manoeuvre_dynamic_pressure_pa(sizing.v_stall_ms, settings), loiter)


def aero_torque_nm(mtow_kg: float, semi_span_m: float, chord_at: Callable[[float], float], q_pa: float, y_m: float,
                   settings: FabricationSettings, cm0: float = AIRFOIL_CM0_DEFAULT,
                   spar_fraction: float = SPAR_CHORD_STATION) -> float:
    """Wing torque about the spar at station y, integrated from the tip inboard: q c^2 |Cm0| per unit span plus the
    limit lift's offset from the spar, n L'(s) |x_spar - x_ac| c (elliptic L'). The two are added as magnitudes
    (conservative: their signs depend on the section and the spar station)."""
    if y_m >= semi_span_m:
        return 0.0
    half_lift = settings.n_limit * mtow_kg * G0 * HALF_WING_FRACTION
    offset, step = abs(spar_fraction - AC_CHORD_FRACTION), (semi_span_m - y_m) / TORQUE_INTEGRATION_STEPS
    total = 0.0
    for k in range(TORQUE_INTEGRATION_STEPS):
        s = y_m + (k + 0.5) * step
        c = chord_at(s)
        lift = ELLIPTIC_LIFT_FACTOR * half_lift / semi_span_m * math.sqrt(max(1.0 - (s / semi_span_m) ** 2, 0.0))
        total += (q_pa * c * c * abs(cm0) + lift * offset * c) * step
    return total


def load_cases(mtow_kg: float, semi_span_m: float, y_m: float, boom_y_m: Optional[float],
               torsion: Optional[SparTorsion], settings: FabricationSettings) -> tuple[LoadCase, ...]:
    """(moment, torque) cases at station y: flight, then hover when there are lift booms. In hover the moment is the
    larger of the balanced reaction (two rotors at MTOW g / 4) and the single full-thrust rotor; the pitch torque
    acts inboard of the boom clamp."""
    flight_torque = 0.0 if torsion is None else torsion.flight_torque_at(y_m)
    flight = (elliptic_moment_nm(mtow_kg, settings.n_limit, semi_span_m, y_m), flight_torque)
    if boom_y_m is None:
        return (flight,)
    arm = boom_y_m - y_m
    thrust = 0.0 if torsion is None else torsion.hover_thrust_n
    moment = max(hover_moment_nm(mtow_kg, arm), thrust * max(arm, 0.0))
    inboard = torsion is not None and y_m <= boom_y_m + FIT_EPS_M
    return flight, (moment, torsion.hover_torque_nm if inboard else 0.0)


# ---------------------------------------------------------------------------------------- booms ---
def joint_y_m(fuselage_radius_m: float, wing_z_m: float) -> float:
    """Span station where the wing spar meets the fuselage skin."""
    return math.sqrt(max(fuselage_radius_m**2 - wing_z_m**2, 0.0))


def boom_tube(diameter_m: float, settings: FabricationSettings) -> TubeStock:
    """Stock tube for a boom of layout `diameter_m`: the largest tube of at least MIN_BOOM_DIAMETER_M that is no
    wider than the layout boom, or, when the layout boom is thinner than that floor, the smallest floor tube."""
    floor_m = MIN_BOOM_DIAMETER_M - FIT_EPS_M
    within = [t for t in settings.tubes if floor_m <= t.od_m <= diameter_m + FIT_EPS_M]
    if within:
        return within[-1]  # tubes are sorted by OD
    floor = next((t for t in settings.tubes if t.od_m >= floor_m), None)
    if floor is None:
        raise KitUnavailable(f"no stock tube for a {diameter_m * 1e3:.0f} mm boom")
    return floor


def cantilever_deflection_m(tube: TubeStock, load_n: float, arm_m: float, settings: FabricationSettings) -> float:
    """Tip deflection of a tube cantilevered over `arm_m` under a tip load: P L^3 / (3 E I)."""
    stiffness = CANTILEVER_DEFLECTION_FACTOR * settings.carbon_modulus_pa * tube.second_moment_m4
    return load_n * max(arm_m, 0.0) ** 3 / stiffness


def lift_boom_tube(thrust_n: float, arm_m: float, max_diameter_m: float, settings: FabricationSettings) -> TubeStock:
    """The smallest stock tube (lighter wall first at equal OD) that carries one lift motor's thrust at `arm_m` from
    the clamp within sigma_allow / safety_factor (R21) and deflects at most BOOM_DEFLECTION_FRACTION x arm there
    (R22), no wider than the layout boom (or the MIN_BOOM_DIAMETER_M floor when the layout boom is thinner)."""
    floor_m, moment = MIN_BOOM_DIAMETER_M - FIT_EPS_M, thrust_n * max(arm_m, 0.0)
    cap_m = max(max_diameter_m, MIN_BOOM_DIAMETER_M) + FIT_EPS_M
    limit_m = BOOM_DEFLECTION_FRACTION * max(arm_m, 0.0)
    fits = sorted((t for t in settings.tubes if floor_m <= t.od_m <= cap_m), key=lambda t: (t.od_m, t.area_m2))
    tube = next((t for t in fits if _passes(t, ((moment, 0.0),), settings)
                 and cantilever_deflection_m(t, thrust_n, arm_m, settings) <= limit_m), None)
    if tube is None:
        raise KitUnavailable(_lift_boom_failure(fits, thrust_n, arm_m, cap_m, settings))
    return tube


def _lift_boom_failure(fits: Sequence[TubeStock], thrust_n: float, arm_m: float, cap_m: float,
                       settings: FabricationSettings) -> str:
    """D18: why no lift boom fits, named by the check the stiffest candidate fails (stress first)."""
    head = f"no stock lift boom up to {cap_m * 1e3:.0f} mm carries {thrust_n:.0f} N at a {arm_m * 1e3:.0f} mm arm"
    if not fits:
        return f"{head}: no stock tube of that size"
    stiffest = max(fits, key=lambda t: t.second_moment_m4)
    limit_m = BOOM_DEFLECTION_FRACTION * max(arm_m, 0.0)
    if not _passes(stiffest, ((thrust_n * max(arm_m, 0.0), 0.0),), settings):
        sigma = bending_stress_pa(stiffest, thrust_n * max(arm_m, 0.0))
        allow = settings.sigma_allow_pa / settings.safety_factor
        return f"{head}: too weak: even {_size(stiffest)} sees {sigma / 1e6:.0f} MPa (allowable {allow / 1e6:.0f} MPa)"
    deflection = cantilever_deflection_m(stiffest, thrust_n, arm_m, settings)
    return (f"{head}: too flexible: even {_size(stiffest)} deflects {deflection * 1e3:.1f} mm at the hub "
            f"(limit {limit_m * 1e3:.1f} mm)")


def _size(tube: TubeStock) -> str:
    return f"{tube.od_m * 1e3:.0f}x{tube.id_m * 1e3:.0f} mm"


def tail_surface_load_n(sizing: SizingOutput, area_m2: float, settings: FabricationSettings) -> float:
    """R35: the most a tail surface of `area_m2` can lift, q_max S CL_max (q_max at V_A, R30), with no n_limit
    factor; the safety factor applies through the allowable. High-speed gusts are not modelled (module docstring)."""
    return flight_dynamic_pressure_pa(sizing, settings) * area_m2 * TAIL_CL_MAX


def smallest_passing(cases: tuple[LoadCase, ...], depth_m: float,
                     settings: FabricationSettings) -> Optional[TubeStock]:
    """The smallest stock tube or rod (by OD, then lighter) no wider than `depth_m` that carries the load cases."""
    rods = tuple(TubeStock(od_m=r, id_m=0.0) for r in settings.rods_m)
    stock = sorted((*settings.tubes, *rods), key=lambda t: (t.od_m, t.area_m2))
    return next((t for t in stock if t.od_m <= depth_m + FIT_EPS_M and _passes(t, cases, settings)), None)


def tail_boom_loads(sizing: SizingOutput, layout: AircraftLayout, settings: FabricationSettings) -> LoadCase:
    """(moment, torque) on the tail boom (R26, loads per R35): q_max S CL_max on each tail; the horizontal and
    vertical forces' resultant bends the boom over the arm from its root (where the cone first holds it,
    conservative) to the tails' aerodynamic centre, and the fin force twists it at its centre-of-pressure height
    (the fin MAC station) above the boom axis. (0, 0) when the tail sits on the fuselage."""
    boom = layout.tail_boom
    if boom is None:
        return 0.0, 0.0
    h_force = tail_surface_load_n(sizing, sizing.horizontal_tail_area_m2, settings)
    v_force = tail_surface_load_n(sizing, sizing.vertical_tail_area_m2, settings)
    arm = max(layout.htail.x_ac_m, layout.vtail.x_ac_m) - boom.x_start_m
    height = layout.vtail.z_root_m + layout.vtail.y_mac_m - boom.z_m
    return math.hypot(h_force, v_force) * max(arm, 0.0), v_force * max(height, 0.0)


def tail_boom_ratio(tube: TubeStock, sizing: SizingOutput, layout: AircraftLayout,
                    settings: FabricationSettings) -> float:
    return tube_ratio(tube, (tail_boom_loads(sizing, layout, settings),), settings)


def tail_boom_check(tube: TubeStock, sizing: SizingOutput, layout: AircraftLayout,
                    settings: FabricationSettings) -> None:
    """Raises KitUnavailable when the stock tail boom (R13: the largest tube within the layout diameter) fails the
    tail loads' bending + torsion interaction."""
    ratio = tail_boom_ratio(tube, sizing, layout, settings)
    if ratio > 1.0 + RATIO_EPS:
        moment, torque = tail_boom_loads(sizing, layout, settings)
        raise KitUnavailable(
            f"tail boom {tube.od_m * 1e3:.0f}x{tube.id_m * 1e3:.0f} mm is too weak for the tail loads "
            f"({moment:.0f} N·m bending, {torque:.0f} N·m torsion; interaction {ratio:.2f} > 1)"
        )


# --------------------------------------------------------------------------------------- wing box ---
def wing_shear_n(sizing: SizingOutput, settings: FabricationSettings) -> float:
    """Shear each wing-box plate bears (R27): the wing's limit lift n MTOW g / 2, or in hover its two lift rotors
    at full thrust, whichever is larger."""
    flight = settings.n_limit * sizing.mtow_kg * G0 * HALF_WING_FRACTION
    return max(flight, HOVER_ARMS * sizing.vtol_thrust_per_motor_n)


def plate_loads(sizing: SizingOutput, settings: FabricationSettings) -> PlateLoads:
    """R27 shear, and the R32 downward load (hard landing or negative g) of magnitude n MTOW g / 2."""
    down = settings.n_limit * sizing.mtow_kg * G0 * HALF_WING_FRACTION
    return PlateLoads(shear_n=wing_shear_n(sizing, settings), down_n=down)


def _bears(force_n: float, rod_m: float, thickness_m: float, settings: FabricationSettings) -> bool:
    return force_n / (rod_m * thickness_m) <= settings.ply_bearing_allow_pa / settings.safety_factor * (1.0 + RATIO_EPS)


def size_pin(torque_nm: float, arm_m: float, settings: FabricationSettings) -> PinPlan:
    """R31: the pin reacts the spar's root torque at `arm_m` from the spar axis, F = T / arm (`pin_for_load`)."""
    return pin_for_load(abs(torque_nm) / arm_m, settings)


def pin_for_load(force: float, settings: FabricationSettings) -> PinPlan:
    """R31 bearing sigma_b = F / (d t) <= ply_bearing_allow_pa / FS in the wing-box plate and in the wing root
    doubler. Ladder: the smallest stock rod from NOMINAL_PIN_ROD_M up that bears F plainly; else the smallest that
    bears it with pin bosses doubling the thickness on whichever side needs one; else KitUnavailable."""
    t_plate, t_root = settings.sheet_for("plate").thickness_m, settings.sheet_for("doubler").thickness_m
    rods = sorted(r for r in settings.rods_m if r >= NOMINAL_PIN_ROD_M - FIT_EPS_M)
    if not rods:
        raise KitUnavailable(f"anti-rotation pin: no rod of at least {NOMINAL_PIN_ROD_M * 1e3:.0f} mm in stock")
    for rod in rods:
        if _bears(force, rod, t_plate, settings) and _bears(force, rod, t_root, settings):
            return PinPlan(rod_m=rod, load_n=force, plate_boss=False, root_boss=False)
    k = BOSS_THICKNESS_FACTOR
    for rod in rods:
        if _bears(force, rod, k * t_plate, settings) and _bears(force, rod, k * t_root, settings):
            return PinPlan(rod_m=rod, load_n=force, plate_boss=not _bears(force, rod, t_plate, settings),
                           root_boss=not _bears(force, rod, t_root, settings))
    raise KitUnavailable(
        f"anti-rotation pin: {force:.0f} N from the spar's root torque overloads even a {rods[-1] * 1e3:.0f} mm rod "
        f"with boss doublers"
    )


def overwrap_stress_pa(moment_nm: float, socket_od_m: float, layers: int, layer_thickness_m: float) -> float:
    """Bending stress of the wrap alone, a thin tube of `layers` x thickness on the socket OD: M c / I_wrap."""
    outer = socket_od_m + 2.0 * layers * layer_thickness_m
    second = math.pi / 64.0 * (outer**4 - socket_od_m**4)
    return moment_nm * (outer / 2.0) / second


def overwrap_layers(moment_nm: float, socket: TubeStock, settings: FabricationSettings) -> int:
    """Fewest cloth layers (<= overwrap_max_layers) whose wrap carries the centreline moment within
    overwrap_allow_pa / safety_factor."""
    allow, t = settings.overwrap_allow_pa / settings.safety_factor, settings.overwrap_layer_thickness_m
    for n in range(1, settings.overwrap_max_layers + 1):
        if overwrap_stress_pa(moment_nm, socket.od_m, n, t) <= allow * (1.0 + RATIO_EPS):
            return n
    raise KitUnavailable(
        f"the centreline overwrap needs more than {settings.overwrap_max_layers} layers of {t * 1e3:.2f} mm cloth "
        f"to carry {moment_nm:.0f} N·m on the {socket.od_m * 1e3:.0f} mm socket"
    )


def size_overwrap(moment_nm: float, socket: TubeStock, plate_gap_m: float,
                  settings: FabricationSettings) -> Overwrap:
    """The centreline overwrap (R24): fewest layers by bending, lap per side max(2 OD, sigma n t FS / tau_epoxy),
    the whole wrap (2 laps) between the wing-box plates' inner faces; mass pi (OD + n t) n t 2L rho."""
    n, t, od = overwrap_layers(moment_nm, socket, settings), settings.overwrap_layer_thickness_m, socket.od_m
    sigma = overwrap_stress_pa(moment_nm, od, n, t)
    lap = max(WRAP_LAP_DIAMETERS * od, sigma * n * t * settings.safety_factor / settings.epoxy_lap_shear_pa)
    if WRAP_SIDES * lap > plate_gap_m + FIT_EPS_M:
        raise KitUnavailable(
            f"the centreline overwrap ({WRAP_SIDES * lap * 1e3:.0f} mm long) does not fit between the wing-box "
            f"plates ({plate_gap_m * 1e3:.0f} mm apart)"
        )
    mass = math.pi * (od + n * t) * n * t * WRAP_SIDES * lap * settings.overwrap_density_kg_m3
    return Overwrap(layers=n, layer_thickness_m=t, lap_m=lap, od_m=od + 2.0 * n * t, mass_kg=mass)


# ------------------------------------------------------------------------------------------ spar ---
def usable_depth_m(local_thickness_m: float, settings: FabricationSettings) -> float:
    """Room for a spar tube once the top and bottom webs and the hole's fit clearance are reserved."""
    return local_thickness_m - 2.0 * MIN_WEB_M - settings.fit_clearance_m


def _unavailable_at(y_m: float) -> KitUnavailable:
    return KitUnavailable(f"no stock tube fits the wing at y = {y_m:.3f} m")


def _stations(joint_m: float, semi_span_m: float) -> list[float]:
    """Joint, every rib pitch outboard strictly inside the span, then the tip."""
    count = int(math.floor((semi_span_m - joint_m) / RIB_PITCH_M))
    inner = [joint_m + k * RIB_PITCH_M for k in range(1, count + 1)]
    return [joint_m] + [y for y in inner if y < semi_span_m - FIT_EPS_M] + [semi_span_m]


def _step_down(
    current: TubeStock, depth_m: float, cases: tuple[LoadCase, ...], settings: FabricationSettings
) -> Optional[TubeStock]:
    """Largest stock tube or rod that slides inside `current`, fits the depth and carries the load cases."""
    rods = tuple(TubeStock(od_m=r, id_m=0.0) for r in settings.rods_m)
    for tube in sorted(settings.tubes + rods, key=lambda t: (t.od_m, t.id_m), reverse=True):
        nests = tube.od_m <= current.id_m + FIT_EPS_M and current.id_m - tube.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if nests and tube.od_m <= depth_m + FIT_EPS_M and _passes(tube, cases, settings):
            return tube
    return None


def _pick_socket(root: TubeStock, moment_nm: float, settings: FabricationSettings) -> TubeStock:
    """The lightest stock tube that sleeves the root member and carries the centreline moment."""
    for tube in settings.tubes:
        sleeves = tube.id_m >= root.od_m - FIT_EPS_M and tube.id_m - root.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if sleeves and _passes(tube, ((moment_nm, 0.0),), settings):
            return tube
    raise KitUnavailable(
        f"no stock tube can sleeve the {root.od_m * 1e3:.0f} mm spar tube as a fuselage socket and carry the "
        f"{moment_nm:.0f} N·m centreline moment"
    )


def _check_insertion(insertion_m: float, root: TubeStock, settings: FabricationSettings) -> None:
    """The spar root must reach far enough into the socket (skin to centreline) to carry its moment."""
    needed = settings.min_insertion_diameters * root.od_m
    if insertion_m < needed - FIT_EPS_M:
        raise KitUnavailable(
            f"spar insertion {insertion_m * 1e3:.0f} mm (fuselage skin to centreline) is shorter than the "
            f"{needed * 1e3:.0f} mm needed for the {root.od_m * 1e3:.0f} mm spar tube"
        )


def depth_limit_m(depth_at: Callable[[float], float], od_m: float, inboard_m: float, outboard_m: float) -> float:
    """Outermost y in [inboard, outboard) where the depth still admits `od_m` (depth is non-increasing)."""
    lo, hi = inboard_m, outboard_m
    for _ in range(BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if depth_at(mid) >= od_m - FIT_EPS_M:
            lo = mid
        else:
            hi = mid
    return lo


def _members(root: TubeStock, joint_m: float, semi_span_m: float, depth_at: Callable[[float], float],
             cases: Callable[[float], tuple[LoadCase, ...]], settings: FabricationSettings) -> tuple[SparSegment, ...]:
    """Telescoping members from the joint to the tip: step down wherever the depth stops admitting the current tube,
    to the tube that passes the load cases at the step."""
    segments: list[SparSegment] = []
    current, start = root, joint_m
    for y in _stations(joint_m, semi_span_m)[1:]:
        while depth_at(y) < current.od_m - FIT_EPS_M:
            y_step = depth_limit_m(depth_at, current.od_m, start, y)
            if y_step <= start + FIT_EPS_M:
                raise _unavailable_at(y)
            smaller = _step_down(current, depth_at(y_step), cases(y_step), settings)
            if smaller is None:
                raise _unavailable_at(y)
            segments.append(SparSegment(current, start, y_step))
            current, start = smaller, y_step
    segments.append(SparSegment(current, start, semi_span_m))
    return tuple(segments)


def plan_spar(
    mtow_kg: float,
    semi_span_m: float,
    joint_m: float,
    depth_at: Callable[[float], float],
    boom_y_m: Optional[float],
    settings: FabricationSettings,
    torsion: Optional[SparTorsion] = None,
    pin_arm_m: Optional[float] = None,
) -> SparPlan:
    """Size a telescoping tube spar from the fuselage joint to the tip, plus its socket and anti-rotation pin. Each
    member is the tube that passes every load case (`load_cases`, with `torsion` when given) at its inboard end,
    where moment and torque are greatest; that also covers the boom clamp. The pin carries the root torque at
    `pin_arm_m` (R31), which `torsion` requires; with no torsion there is no torque and the pin is NOMINAL_PIN."""
    if boom_y_m is not None and boom_y_m < 0.0:
        raise ValueError(f"boom_y_m must be a distance from the centreline, got {boom_y_m}")
    if semi_span_m <= joint_m + FIT_EPS_M:  # D8: no exposed span, so no spar member would have any length
        raise KitUnavailable(f"the wing's semi-span ({semi_span_m:.3f} m) does not reach past the fuselage joint "
                             f"(y = {joint_m:.3f} m)")
    if torsion is not None and pin_arm_m is None:
        raise ValueError("pin_arm_m is required with torsion: the anti-rotation pin reacts the root torque")

    def cases(y: float) -> tuple[LoadCase, ...]:
        return load_cases(mtow_kg, semi_span_m, y, boom_y_m, torsion, settings)

    root_depth, root_cases = depth_at(joint_m), cases(joint_m)
    # No torsion, no torque: the nominal rod, whose stock is checked where its holes are cut (`pin_hole_m`).
    pin = NOMINAL_PIN if pin_arm_m is None else pin_for_load(max(t for _, t in root_cases) / pin_arm_m, settings)
    root = next((t for t in settings.tubes
                 if t.od_m <= root_depth + FIT_EPS_M and _passes(t, root_cases, settings)), None)
    if root is None:
        raise _unavailable_at(joint_m)

    segments = _members(root, joint_m, semi_span_m, depth_at, cases, settings)
    # The socket carries the bending moment "as above" (flight vs the balanced hover reaction) at the centreline.
    centre_moment = max(m for m, _ in load_cases(mtow_kg, semi_span_m, ROOT_SPAN_M, boom_y_m, None, settings))
    socket = _pick_socket(root, centre_moment, settings)
    _check_insertion(joint_m, root, settings)
    # The socket spans the fuselage only; each spar root plugs in from the skin to the centreline.
    return SparPlan(segments, socket, joint_m, joint_m, centre_moment, pin)
