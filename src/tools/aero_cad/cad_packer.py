"""Collision-free internal packaging solved for static margin, plus full-aircraft inertia (SRS 4.3).

Fixed stations come from `components` (tractor motor, chin gimbal, tank on the neutral point so fuel
burn causes no CG shift, generator aft of the tank). The battery and avionics tray slide along x
independently - tray scanned, battery solved with scipy `minimize_scalar` inside each free interval -
until SM = (X_np - X_cg) / MAC is centred in [0.08, 0.14]. A slot is free only if the part clears every
other part and boom and fits the local skin radius; the final OML-solid containment check counts any
skin interference as a collision.

Airframe mass comes from the laser-cut kit when one can be built (Group 1-2): kit parts, tubes, stock,
covering and glue plus the residual landing gear. Two passes: the kit is first built around the fixed parts
(estimate for the slider solve, and its socket tubes, wing-box plates and bulkheads become 3D obstacles kept
STRUCTURE_CLEARANCE_M away; the sliders sit as low as their former openings allow, R23), then
rebuilt around every placed part for exact openings and the final mass. When no kit can be built - in either
pass - packing falls back to the shell model with `kit_warning` saying why: the composite OML shell
(rho = 1600 kg/m^3, thickness derived) carries the sized airframe mass minus ribs and spars. Discrete parts use
their actual COTS masses, so the as-built mass exposes any catalog-snapping growth for the critic's
MTOW-convergence check.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Sequence

import build123d as bd
import numpy as np
from scipy.optimize import minimize_scalar

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.aero_cad import components as parts
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, load_airfoil
from src.tools.aero_cad.components import (  # noqa: F401  (re-exported for callers)
    COMPONENT_GAP_M,
    PlacedComponent,
    count_collisions,
    overlap_volume_m3,
)
from src.tools.aero_cad.layout import AircraftLayout
from src.tools.aero_cad.mesh_export import export_viewer_model
from src.tools.aero_cad.structure import COMPOSITE_DENSITY_KG_M3, WingStructure, build_wing_structure
from src.tools.fabrication.build_kit import MAX_DOD_GROUP, build_kit
from src.tools.fabrication.kit import Kit, KitPart, KitUnavailable, TubePart
from src.tools.fabrication.kit_export import KitFiles, KitSolids, export_kit, kit_solid_sets, remove_kit_files
from src.tools.fabrication.kit_mass import kit_mass_items, residual_gear_item
from src.tools.fabrication.kit_pod import STRUCTURE_CLEARANCE_M, fits_kit, seat_low
from src.tools.sizing import layouts
from src.tools.sizing.layouts import StaticMarginBand
from src.tools.sizing.mass_model import PropulsionLoads, airframe_mass_kg
from src.tools.sizing.mission_solver import VTOL_ROTOR_COUNT

logger = logging.getLogger(__name__)

SLIDER_SCAN_POINTS: Final[int] = 400
TRAY_SCAN_POINTS: Final[int] = 80
STEP_IMPORT_TO_M: Final[float] = 1e-3  # OCC's STEP reader always returns millimetres
STRUCTURE_PART_PREFIXES: Final[tuple[str, ...]] = ("WM-", "FW-", "BM-")  # wing-box plates, firewall, boom mounts
STRUCTURE_TUBE_ROLES: Final[tuple[str, ...]] = ("socket",)
NO_KIT_PREFIX: Final[str] = "No laser kit: "
KIT_FILES_PREFIX: Final[str] = "kit files could not be written: "
KIT_SOLIDS_PREFIX: Final[str] = "kit solids could not be built: "
KIT_CLEANUP_PREFIX: Final[str] = "stale kit files could not be removed: "
WARNING_SEPARATOR: Final[str] = "; "

Box3 = tuple[tuple[float, float, float], tuple[float, float, float]]  # axis-aligned (min corner, max corner)


class CADPackingError(RuntimeError):
    """The aircraft cannot be packaged (e.g. no airframe mass left, or no collision-free slot)."""


class PackingSlotError(CADPackingError):
    """No collision-free, skin-contained fuselage slot for the battery or avionics tray."""


@dataclass(frozen=True)
class PackingResult:
    components: tuple[PlacedComponent, ...]
    mass: mp.MassProperties
    static_margin: float
    band: StaticMarginBand  # the layout's band the margin is judged against
    collision_count: int  # part-part and part-boom overlaps
    containment_violations: int  # internal parts poking through the OML
    shell_areal_density_kg_m2: float  # 0.0 with a kit
    oml: bd.Shape
    structure: Optional[WingStructure]  # shell-model fallback only (None with a kit)
    kit: Optional[Kit] = None
    kit_warning: Optional[str] = None  # why there is no kit, when there is none
    mass_items: tuple[mp.MassItem, ...] = ()  # the items combined into `mass`
    kit_items: tuple[mp.MassItem, ...] = ()  # the kit's own mass items (empty without a kit); exports reuse them
    settings: FabricationSettings = DEFAULT_FABRICATION  # what the kit was built with (its export uses them too)

    @property
    def interference_count(self) -> int:
        return self.collision_count + self.containment_violations

    @property
    def is_balanced(self) -> bool:
        return self.band.minimum <= self.static_margin <= self.band.maximum and self.interference_count == 0


# ------------------------------------------------------------------------------- slider solver ---
@dataclass(frozen=True)
class _Balance:
    """Mass and x-moment of everything already placed, plus the SM reference."""

    static_mass: float
    static_moment: float
    x_np: float
    mac: float
    target: float  # static margin the sliders aim for

    def sm_error(self, extra_mass: float, extra_moment: float) -> float:
        x_cg: float = (self.static_moment + extra_moment) / (self.static_mass + extra_mass)
        return ((self.x_np - x_cg) / self.mac - self.target) ** 2


def _component_box(item: PlacedComponent) -> Box3:
    (x, y, z), (lx, ly, lz) = item.center_m, item.size_m
    return (x - lx / 2, y - ly / 2, z - lz / 2), (x + lx / 2, y + ly / 2, z + lz / 2)


def _clear_of_structure(item: PlacedComponent, boxes: Sequence[Box3]) -> bool:
    """Kit structure blocks in 3D: the item's box must keep STRUCTURE_CLEARANCE_M from each structure box along at
    least one axis, so a part may pass beneath the socket (unlike the x-only rule between components) without the
    kit having to cut the wing-box plates."""
    lo, hi = _component_box(item)
    c = STRUCTURE_CLEARANCE_M
    return all(any(hi[k] + c <= b_lo[k] or b_hi[k] + c <= lo[k] for k in range(3)) for b_lo, b_hi in boxes)


@dataclass(frozen=True)
class _KitRules:
    """Slot rules when the airframe is a kit: structure boxes to clear, former openings and crutch rails to fit (R23, R28)."""

    boxes: tuple[Box3, ...]
    settings: FabricationSettings


def _is_clear(item: PlacedComponent, obstacles: Sequence[PlacedComponent], layout: AircraftLayout,
              kit: Optional[_KitRules] = None) -> bool:
    """x-clear of every other component; with a kit, also clear of its structure and inside its former openings
    with crutch rails to spare (R28; this replaces the shell model's radial margin), else inside that margin."""
    gap = COMPONENT_GAP_M
    clear_x: bool = all(item.x_max + gap <= o.x_min or item.x_min >= o.x_max + gap for o in obstacles)
    if kit is None:
        return clear_x and parts.fits_radially(item, layout)
    return clear_x and _clear_of_structure(item, kit.boxes) and fits_kit(item, layout, kit.settings)


def _center_bounds(item: PlacedComponent, layout: AircraftLayout) -> tuple[float, float]:
    half: float = item.size_m[0] / 2
    lo, hi = COMPONENT_GAP_M + half, layout.fuselage_length_m - COMPONENT_GAP_M - half
    if hi <= lo:
        raise PackingSlotError(f"{item.name} ({item.size_m[0]:.2f} m) is longer than the fuselage")
    return lo, hi


def _clear_intervals(grid: np.ndarray, clear: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous runs of free grid points as (first, last) positions."""
    intervals: list[tuple[float, float]] = []
    start: Optional[float] = None
    prev: float = float(grid[0])
    for x, ok in zip(grid, clear):
        if ok and start is None:
            start = float(x)
        if not ok and start is not None:
            intervals.append((start, prev))
            start = None
        prev = float(x)
    if start is not None:
        intervals.append((start, prev))
    return intervals


def _solve_slider(item: PlacedComponent, obstacles: Sequence[PlacedComponent], balance: _Balance,
                  layout: AircraftLayout, extra: tuple[float, float] = (0.0, 0.0),
                  kit: Optional[_KitRules] = None) -> float:
    """Centre x of `item` closest to the balance's target margin among free positions: scan for free intervals, then
    bounded Brent (`minimize_scalar`) inside each. `extra` = (mass, x-moment) of other placed sliders."""
    lo, hi = _center_bounds(item, layout)

    def err(x: float) -> float:
        return balance.sm_error(extra[0] + item.mass_kg, extra[1] + item.mass_kg * x)

    grid = np.linspace(lo, hi, SLIDER_SCAN_POINTS)
    free = np.array([_is_clear(item.at_x(float(x)), obstacles, layout, kit) for x in grid])
    intervals = _clear_intervals(grid, free)
    if not intervals:
        raise PackingSlotError(f"no free fuselage slot for the {item.name}")
    candidates = [a if b <= a else float(minimize_scalar(err, bounds=(a, b), method="bounded").x) for a, b in intervals]
    return min(candidates, key=err)


def _solve_battery_and_tray(battery: PlacedComponent, tray: PlacedComponent, obstacles: Sequence[PlacedComponent],
                            balance: _Balance, layout: AircraftLayout,
                            kit: Optional[_KitRules] = None) -> tuple[PlacedComponent, PlacedComponent]:
    """Slide both independently: scan tray positions and, for each, solve the battery position."""
    lo, hi = _center_bounds(tray, layout)
    best: Optional[tuple[float, PlacedComponent, PlacedComponent]] = None
    for x_tray in np.linspace(lo, hi, TRAY_SCAN_POINTS):
        placed_tray = tray.at_x(float(x_tray))
        if not _is_clear(placed_tray, obstacles, layout, kit):
            continue
        extra = (tray.mass_kg, tray.mass_kg * float(x_tray))
        try:
            x_bat: float = _solve_slider(battery, [*obstacles, placed_tray], balance, layout, extra, kit)
        except PackingSlotError:
            continue
        error: float = balance.sm_error(extra[0] + battery.mass_kg, extra[1] + battery.mass_kg * x_bat)
        if best is None or error < best[0]:
            best = (error, battery.at_x(x_bat), placed_tray)
    if best is None:
        raise PackingSlotError("no free fuselage slots for the battery and avionics tray")
    return best[1], best[2]


# ---------------------------------------------------------------------------------------- shell ---
def _load_oml(oml_step: Path) -> bd.Shape:
    if not oml_step.is_file():
        raise FileNotFoundError(f"OML STEP not found: {oml_step}")
    oml: bd.Shape = bd.import_step(str(oml_step)).scale(STEP_IMPORT_TO_M)
    if not oml.solids():
        raise CADPackingError(f"OML STEP {oml_step} has no solid: the surface is not watertight")
    return oml


def sized_airframe_mass_kg(sizing: SizingOutput) -> float:
    """Structure, booms and landing gear from the sizing build-up (no propulsion/avionics/systems)."""
    loads = PropulsionLoads(
        cruise_motor_power_w=sizing.cruise_motor_power_w,
        vtol_total_thrust_n=sizing.vtol_thrust_per_motor_n * VTOL_ROTOR_COUNT,
        generator_power_w=sizing.generator_power_w,
        fuel_mass_kg=sizing.fuel_mass_kg,
    )
    return airframe_mass_kg(sizing.mtow_kg, sizing.empty_mass_kg, loads, sizing.avionics_mass_kg)


def _shell_areal_density(airframe_kg: float, wing: WingStructure, oml: bd.Shape) -> float:
    structure_kg: float = wing.rib_mass_kg + wing.spar_mass_kg
    if airframe_kg <= structure_kg:
        raise CADPackingError(
            f"no mass left for the airframe shell: sized airframe {airframe_kg:.3f} kg <= ribs/spars {structure_kg:.3f} kg"
        )
    return (airframe_kg - structure_kg) / oml.area


def _structure_items(oml: bd.Shape, wing: WingStructure, areal_density: float) -> list[mp.MassItem]:
    return [
        mp.shell_item("oml_shell", oml, areal_density),
        *(mp.solid_item(f"rib_{i}", r, COMPOSITE_DENSITY_KG_M3) for i, r in enumerate(wing.ribs)),
        *(mp.solid_item(f"spar_{i}", s, COMPOSITE_DENSITY_KG_M3) for i, s in enumerate(wing.spars)),
    ]


# ---------------------------------------------------------------------------------- airframe mass ---
@dataclass(frozen=True)
class _Airframe:
    """Airframe mass items (kit or shell model) and the kit structure the sliders must avoid."""

    items: tuple[mp.MassItem, ...]
    boxes: tuple[Box3, ...] = ()
    kit: Optional[Kit] = None
    kit_warning: Optional[str] = None
    structure: Optional[WingStructure] = None
    shell_areal_density_kg_m2: float = 0.0
    kit_items: tuple[mp.MassItem, ...] = ()  # the kit's own items (in `items`, without the residual gear)


def _shell_airframe(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape, oml: bd.Shape,
                    warning: str) -> _Airframe:
    wing: WingStructure = build_wing_structure(layout, airfoil)
    areal: float = _shell_areal_density(sized_airframe_mass_kg(sizing), wing, oml)
    return _Airframe(tuple(_structure_items(oml, wing, areal)), kit_warning=warning, structure=wing,
                     shell_areal_density_kg_m2=areal)


def _part_box(part: KitPart, settings: FabricationSettings) -> Box3:
    thickness: float = settings.sheet_for(part.family).thickness_m
    box = bd.extrude(part.outline, amount=thickness, dir=(0, 0, 1)).moved(part.location).bounding_box()
    return (box.min.X, box.min.Y, box.min.Z), (box.max.X, box.max.Y, box.max.Z)


def _tube_box(tube: TubePart) -> Box3:
    """Axis-aligned box around the tube: its axis segment grown by the outer radius (conservative when oblique)."""
    end = np.asarray(tube.start_m) + tube.length_m * np.asarray(tube.axis)
    radius: float = tube.tube.od_m / 2
    lo = np.minimum(tube.start_m, end) - radius
    hi = np.maximum(tube.start_m, end) + radius
    return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))


def _structure_boxes(kit: Kit, settings: FabricationSettings) -> tuple[Box3, ...]:
    """Kit structure no component may cross: socket tubes and their centreline overwrap (at its wrapped OD),
    wing-box plates and socket bosses, firewall and boom-mount bulkheads (formers, crutch and trays take openings or
    belong to components)."""
    plates = [_part_box(p, settings) for p in kit.parts if p.label.startswith(STRUCTURE_PART_PREFIXES)]
    tubes = [t for t in kit.tubes if t.role in STRUCTURE_TUBE_ROLES]
    return (*plates, *(_tube_box(t) for t in (*tubes, *kit.overwrap_halves())))


def _kit_airframe(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
                  placed: Sequence[PlacedComponent], settings: FabricationSettings) -> _Airframe:
    """Raises KitUnavailable when the kit cannot be built around `placed`."""
    kit: Kit = build_kit(sizing, layout, airfoil, placed, settings)
    kit_items = tuple(kit_mass_items(kit, settings))
    return _Airframe((*kit_items, residual_gear_item(sizing, layout)), _structure_boxes(kit, settings), kit=kit,
                     kit_items=kit_items)


def _airframe(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape, oml: bd.Shape,
              placed: Sequence[PlacedComponent], settings: FabricationSettings) -> _Airframe:
    """The kit built around `placed`, or the shell model when no kit can be built: a kit failure never fails the
    run, whatever raised it. Group 3+ designs expect no kit (info); any other failure is a warning, and an
    unexpected error is logged with its traceback."""
    try:
        return _kit_airframe(sizing, layout, airfoil, placed, settings)
    except KitUnavailable as exc:
        warning = f"{NO_KIT_PREFIX}{exc}"
        expected: bool = sizing.dod_group > MAX_DOD_GROUP
        logger.log(logging.INFO if expected else logging.WARNING, "%s - using the shell mass model", warning)
    except Exception as exc:  # noqa: BLE001 - geometry-kernel errors must not fail the run either
        warning = f"{NO_KIT_PREFIX}unexpected error building the kit ({type(exc).__name__}: {exc})"
        logger.exception("%s - using the shell mass model", warning)
    return _shell_airframe(sizing, layout, airfoil, oml, warning)


# ------------------------------------------------------------------------------------- packing ---
def _sliders(sizing: SizingOutput, bom: SelectedBOM, layout: AircraftLayout,
             kit: Optional[_KitRules]) -> tuple[PlacedComponent, PlacedComponent]:
    """Battery and tray on the axis, or seated as low as their former openings allow when the airframe is a kit."""
    battery, tray = parts.sliders(sizing, bom)
    if kit is None:
        return battery, tray
    d = layout.fuselage_diameter_m
    return seat_low(battery, d, kit.settings), seat_low(tray, d, kit.settings)


def _place_sliders(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput, layout: AircraftLayout,
                   fixed: Sequence[PlacedComponent], airframe: _Airframe, settings: FabricationSettings,
                   band: StaticMarginBand) -> tuple[PlacedComponent, PlacedComponent]:
    """Solve battery + tray for static margin against `airframe`'s mass, under its rules (kit or shell)."""
    static_items = [*airframe.items, *(c.mass_item() for c in fixed)]
    balance = _Balance(sum(i.mass_kg for i in static_items), sum(i.mass_kg * i.cg_m[0] for i in static_items),
                       aero.x_np_m, layout.wing.mac_m, band.target)
    booms = parts.boom_obstacles(layout)
    on_axis = [c for c in (*fixed, *booms) if abs(c.center_m[1]) < layout.fuselage_diameter_m / 2]
    rules = None if airframe.kit is None else _KitRules(airframe.boxes, settings)
    return _solve_battery_and_tray(*_sliders(sizing, bom, layout, rules), on_axis, balance, layout, rules)


def pack_layout(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput, layout: AircraftLayout,
                oml_step: Path, settings: FabricationSettings = DEFAULT_FABRICATION) -> PackingResult:
    """Place every part, slide battery + tray for static margin, build the mass properties.

    Pass 1 builds the kit around the fixed parts (mass estimate + structure obstacles) and solves the sliders;
    pass 2 rebuilds it around every placed part for exact openings and the final mass. If that rebuild fails, the
    sliders are re-solved for the shell model it falls back to. The static margin and `is_balanced` use the final
    mass."""
    band: StaticMarginBand = layouts.static_margin_band(sizing.airframe_layout)
    oml: bd.Shape = _load_oml(oml_step)
    airfoil: AirfoilShape = load_airfoil(layout.airfoil)
    fixed = [parts.cruise_motor(bom), parts.payload(bom, layout),
             *parts.hybrid_parts(sizing, bom, layout, aero.x_np_m), *parts.vtol_motors(bom, layout)]
    estimate: _Airframe = _airframe(sizing, layout, airfoil, oml, fixed, settings)
    placed = _place_sliders(sizing, bom, aero, layout, fixed, estimate, settings, band)
    final = estimate if estimate.kit is None else _airframe(sizing, layout, airfoil, oml, (*fixed, *placed), settings)
    if estimate.kit is not None and final.kit is None:
        placed = _place_sliders(sizing, bom, aero, layout, fixed, final, settings, band)
    components = (*fixed, *placed)
    mass_items = (*final.items, *(c.mass_item() for c in components))
    mass = mp.combine(list(mass_items))
    result = PackingResult(
        components=components, mass=mass, static_margin=(aero.x_np_m - mass.cg_m[0]) / layout.wing.mac_m, band=band,
        collision_count=count_collisions(components, parts.boom_obstacles(layout)),
        containment_violations=parts.containment_violations(components, oml),
        shell_areal_density_kg_m2=final.shell_areal_density_kg_m2, oml=oml, structure=final.structure,
        kit=final.kit, kit_warning=final.kit_warning, mass_items=mass_items, settings=settings,
        kit_items=final.kit_items,
    )
    if not result.is_balanced:
        logger.warning("packing unbalanced: SM %.3f (band %.2f-%.2f), %d collisions, %d skin interferences",
                       result.static_margin, band.minimum, band.maximum, result.collision_count, result.containment_violations)
    return result


def _kit_solids(result: PackingResult) -> tuple[Optional[KitSolids], Optional[str]]:
    """The kit's solids, built once per export for the assembly and the viewer (None without a kit). A failure never
    fails the run: it is logged, both go without kit structure (empty sets), and the warning says why."""
    if result.kit is None:
        return None, None
    try:
        return kit_solid_sets(result.kit, result.settings), None
    except Exception as exc:  # noqa: BLE001 - geometry-kernel errors must not fail the run
        warning = f"{KIT_SOLIDS_PREFIX}{exc}"
        logger.exception("%s - the assembly carries no kit structure", warning)
        return KitSolids(ply=(), carbon=()), warning


def _structure_solids(result: PackingResult, solids: Optional[KitSolids]) -> list[bd.Shape]:
    """The airframe structure for the assembly: the kit's solids (replacing the shell ribs and spars) when there is a
    kit, else the shell-model ribs and spars."""
    if solids is not None:
        return solids.all
    return [] if result.structure is None else [*result.structure.ribs, *result.structure.spars]


def _export_assembly(result: PackingResult, path: Path, structure: Sequence[bd.Shape]) -> Path:
    solids = [result.oml, *structure, *(c.solid() for c in result.components)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not bd.export_step(bd.Compound(solids), str(path), unit=bd.Unit.M):
        raise CADPackingError(f"STEP export failed: {path}")
    return path


def _export_kit_files(result: PackingResult, out_dir: Path) -> tuple[Optional[KitFiles], Optional[str]]:
    """The kit's DXF sheets, build sheet and parts list (none, and an earlier run's removed, without a kit). Any
    failure is logged and returned as a warning: the run goes on with the kit as packed."""
    prefix = KIT_CLEANUP_PREFIX if result.kit is None else KIT_FILES_PREFIX
    try:
        if result.kit is None:
            remove_kit_files(out_dir)
            return None, None
        return export_kit(result.kit, result.settings, out_dir, result.kit_items), None
    except Exception as exc:  # noqa: BLE001 - nesting, DXF or file errors must not fail the run
        warning = f"{prefix}{exc}"
        logger.exception("%s - the run continues", warning)
        return None, warning


def _kit_warning(*warnings: Optional[str]) -> Optional[str]:
    present = [w for w in warnings if w]
    return WARNING_SEPARATOR.join(present) if present else None


def export_artifacts(result: PackingResult, oml_vsp3: Path, oml_step: Path, out_dir: Path) -> CADArtifacts:
    """Write `assembly.step` (with the kit's solids when there is a kit), the viewer model and the kit files, and
    report CG, static margin, interference, as-built mass and inertia. `collision_count` covers part-part, part-boom
    and part-skin interference. No kit-file or kit-solid failure fails the run: `kit_warning` says what went wrong."""
    kit_files, files_warning = _export_kit_files(result, out_dir)
    solids, solids_warning = _kit_solids(result)
    assembly: Path = _export_assembly(result, out_dir / "assembly.step", _structure_solids(result, solids))
    viewer_model: Optional[Path] = export_viewer_model(result, out_dir, solids)
    ixx, iyy, izz, ixz = mp.body_inertia(result.mass)
    return CADArtifacts(
        oml_vsp3_path=str(oml_vsp3), oml_step_path=str(oml_step), assembly_step_path=str(assembly),
        x_cg_m=result.mass.cg_m[0], static_margin=result.static_margin, collision_count=result.interference_count,
        as_built_mass_kg=result.mass.mass_kg, ixx_kgm2=ixx, iyy_kgm2=iyy, izz_kgm2=izz, ixz_kgm2=ixz,
        viewer_model_path=str(viewer_model) if viewer_model else None,
        kit_dir=str(kit_files.kit_dir) if kit_files else None,
        build_sheet_path=str(kit_files.build_sheet) if kit_files else None,
        kit_mass_kg=sum(i.mass_kg for i in result.kit_items) if result.kit is not None else None,
        kit_warning=_kit_warning(result.kit_warning, files_warning, solids_warning),
    )


def pack_aircraft(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput, layout: AircraftLayout,
                  oml_vsp3: Path, oml_step: Path, out_dir: Path,
                  settings: FabricationSettings = DEFAULT_FABRICATION) -> CADArtifacts:
    """Pack for a fixed layout and export the assembly."""
    return export_artifacts(pack_layout(sizing, bom, aero, layout, oml_step, settings), oml_vsp3, oml_step, out_dir)
