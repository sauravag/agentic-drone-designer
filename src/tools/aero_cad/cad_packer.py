"""Collision-free internal packaging solved for static margin, plus full-aircraft inertia (SRS 4.3).

Fixed stations come from `components` (tractor motor, chin gimbal, tank on the neutral point so fuel
burn causes no CG shift, generator aft of the tank). The battery and avionics tray slide along x
independently - tray scanned, battery solved with scipy `minimize_scalar` inside each free interval -
until SM = (X_np - X_cg) / MAC is centred in [0.08, 0.14]. A slot is free only if the part clears every
other part and boom and fits the local skin radius; the final OML-solid containment check counts any
skin interference as a collision.

The composite OML shell (rho = 1600 kg/m^3, thickness derived) carries the sized airframe mass minus
ribs and spars; discrete parts use their actual COTS masses, so the as-built mass exposes any
catalog-snapping growth for the critic's MTOW-convergence check.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Sequence

import build123d as bd
import numpy as np
from scipy.optimize import minimize_scalar

from src.schemas.components import SelectedBOM
from src.schemas.state import AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.aero_cad import components as parts
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.components import (  # noqa: F401  (re-exported for callers)
    COMPONENT_GAP_M,
    PlacedComponent,
    count_collisions,
    overlap_volume_m3,
)
from src.tools.aero_cad.layout import AircraftLayout
from src.tools.aero_cad.mesh_export import export_viewer_model
from src.tools.aero_cad.structure import COMPOSITE_DENSITY_KG_M3, WingStructure, build_wing_structure
from src.tools.sizing.mass_model import PropulsionLoads, airframe_mass_kg
from src.tools.sizing.mission_solver import VTOL_ROTOR_COUNT

logger = logging.getLogger(__name__)

SM_MIN: Final[float] = 0.08
SM_MAX: Final[float] = 0.14
SM_TARGET: Final[float] = 0.5 * (SM_MIN + SM_MAX)
SLIDER_SCAN_POINTS: Final[int] = 400
TRAY_SCAN_POINTS: Final[int] = 80
STEP_IMPORT_TO_M: Final[float] = 1e-3  # OCC's STEP reader always returns millimetres


class CADPackingError(RuntimeError):
    """The aircraft cannot be packaged (e.g. no airframe mass left, or no collision-free slot)."""


class PackingSlotError(CADPackingError):
    """No collision-free, skin-contained fuselage slot for the battery or avionics tray."""


@dataclass(frozen=True)
class PackingResult:
    components: tuple[PlacedComponent, ...]
    mass: mp.MassProperties
    static_margin: float
    collision_count: int  # part-part and part-boom overlaps
    containment_violations: int  # internal parts poking through the OML
    shell_areal_density_kg_m2: float
    oml: bd.Shape
    structure: WingStructure

    @property
    def interference_count(self) -> int:
        return self.collision_count + self.containment_violations

    @property
    def is_balanced(self) -> bool:
        return SM_MIN <= self.static_margin <= SM_MAX and self.interference_count == 0


# ------------------------------------------------------------------------------- slider solver ---
@dataclass(frozen=True)
class _Balance:
    """Mass and x-moment of everything already placed, plus the SM reference."""

    static_mass: float
    static_moment: float
    x_np: float
    mac: float

    def sm_error(self, extra_mass: float, extra_moment: float) -> float:
        x_cg: float = (self.static_moment + extra_moment) / (self.static_mass + extra_mass)
        return ((self.x_np - x_cg) / self.mac - SM_TARGET) ** 2


def _is_clear(item: PlacedComponent, obstacles: Sequence[PlacedComponent], layout: AircraftLayout) -> bool:
    gap = COMPONENT_GAP_M
    clear_x: bool = all(item.x_max + gap <= o.x_min or item.x_min >= o.x_max + gap for o in obstacles)
    return clear_x and parts.fits_radially(item, layout)


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
                  layout: AircraftLayout, extra: tuple[float, float] = (0.0, 0.0)) -> float:
    """Centre x of `item` closest to SM_TARGET among free positions: scan for free intervals, then
    bounded Brent (`minimize_scalar`) inside each. `extra` = (mass, x-moment) of other placed sliders."""
    lo, hi = _center_bounds(item, layout)

    def err(x: float) -> float:
        return balance.sm_error(extra[0] + item.mass_kg, extra[1] + item.mass_kg * x)

    grid = np.linspace(lo, hi, SLIDER_SCAN_POINTS)
    intervals = _clear_intervals(grid, np.array([_is_clear(item.at_x(float(x)), obstacles, layout) for x in grid]))
    if not intervals:
        raise PackingSlotError(f"no free fuselage slot for the {item.name}")
    candidates = [a if b <= a else float(minimize_scalar(err, bounds=(a, b), method="bounded").x) for a, b in intervals]
    return min(candidates, key=err)


def _solve_battery_and_tray(battery: PlacedComponent, tray: PlacedComponent, obstacles: Sequence[PlacedComponent],
                            balance: _Balance, layout: AircraftLayout) -> tuple[PlacedComponent, PlacedComponent]:
    """Slide both independently: scan tray positions and, for each, solve the battery position."""
    lo, hi = _center_bounds(tray, layout)
    best: Optional[tuple[float, PlacedComponent, PlacedComponent]] = None
    for x_tray in np.linspace(lo, hi, TRAY_SCAN_POINTS):
        placed_tray = tray.at_x(float(x_tray))
        if not _is_clear(placed_tray, obstacles, layout):
            continue
        extra = (tray.mass_kg, tray.mass_kg * float(x_tray))
        try:
            x_bat: float = _solve_slider(battery, [*obstacles, placed_tray], balance, layout, extra)
        except PackingSlotError:
            continue
        error: float = balance.sm_error(extra[0] + battery.mass_kg, extra[1] + battery.mass_kg * x_bat)
        if best is None or error < best[0]:
            best = (error, battery.at_x(x_bat), placed_tray)
    if best is None:
        raise PackingSlotError("no free fuselage slots for the battery and avionics tray")
    return best[1], best[2]


# ------------------------------------------------------------------------------------- packing ---
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
    return airframe_mass_kg(sizing.mtow_kg, sizing.empty_mass_kg, loads)


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


def pack_layout(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput,
                layout: AircraftLayout, oml_step: Path) -> PackingResult:
    """Place every part, slide battery + tray for static margin, build the mass properties."""
    oml: bd.Shape = _load_oml(oml_step)
    wing: WingStructure = build_wing_structure(layout, load_airfoil(layout.airfoil))
    fixed = [parts.cruise_motor(bom), parts.payload(bom, layout),
             *parts.hybrid_parts(sizing, bom, layout, aero.x_np_m), *parts.vtol_motors(bom, layout)]
    booms = parts.boom_obstacles(layout)
    areal: float = _shell_areal_density(sized_airframe_mass_kg(sizing), wing, oml)
    static_items = _structure_items(oml, wing, areal) + [c.mass_item() for c in fixed]
    balance = _Balance(sum(i.mass_kg for i in static_items), sum(i.mass_kg * i.cg_m[0] for i in static_items),
                       aero.x_np_m, layout.wing.mac_m)
    on_axis = [c for c in (*fixed, *booms) if abs(c.center_m[1]) < layout.fuselage_diameter_m / 2]
    placed = _solve_battery_and_tray(*parts.sliders(sizing, bom), on_axis, balance, layout)
    mass = mp.combine(static_items + [c.mass_item() for c in placed])
    components = (*fixed, *placed)
    result = PackingResult(
        components=components, mass=mass, static_margin=(aero.x_np_m - mass.cg_m[0]) / layout.wing.mac_m,
        collision_count=count_collisions(components, booms),
        containment_violations=parts.containment_violations(components, oml),
        shell_areal_density_kg_m2=areal, oml=oml, structure=wing,
    )
    if not result.is_balanced:
        logger.warning("packing unbalanced: SM %.3f (band %.2f-%.2f), %d collisions, %d skin interferences",
                       result.static_margin, SM_MIN, SM_MAX, result.collision_count, result.containment_violations)
    return result


def _export_assembly(result: PackingResult, path: Path) -> Path:
    solids = [result.oml, *result.structure.ribs, *result.structure.spars, *(c.solid() for c in result.components)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not bd.export_step(bd.Compound(solids), str(path), unit=bd.Unit.M):
        raise CADPackingError(f"STEP export failed: {path}")
    return path


def export_artifacts(result: PackingResult, oml_vsp3: Path, oml_step: Path, out_dir: Path) -> CADArtifacts:
    """Write `assembly.step` and report CG, static margin, interference, as-built mass and inertia.
    `collision_count` covers part-part, part-boom and part-skin interference."""
    assembly: Path = _export_assembly(result, out_dir / "assembly.step")
    viewer_model: Optional[Path] = export_viewer_model(result, out_dir)
    ixx, iyy, izz, ixz = mp.body_inertia(result.mass)
    return CADArtifacts(
        oml_vsp3_path=str(oml_vsp3), oml_step_path=str(oml_step), assembly_step_path=str(assembly),
        x_cg_m=result.mass.cg_m[0], static_margin=result.static_margin, collision_count=result.interference_count,
        as_built_mass_kg=result.mass.mass_kg, ixx_kgm2=ixx, iyy_kgm2=iyy, izz_kgm2=izz, ixz_kgm2=ixz,
        viewer_model_path=str(viewer_model) if viewer_model else None,
    )


def pack_aircraft(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput, layout: AircraftLayout,
                  oml_vsp3: Path, oml_step: Path, out_dir: Path) -> CADArtifacts:
    """Pack for a fixed layout and export the assembly."""
    return export_artifacts(pack_layout(sizing, bom, aero, layout, oml_step), oml_vsp3, oml_step, out_dir)
