"""Laser-cut airframe kit model: flat ply parts, stock tubes and linear stock (SI units, metres).

Frame convention (Tasks 3-9 rely on it):
- Aircraft frame: x aft from the nose, y right, z up.
- `KitPart.outline` is the nominal cut shape (no kerf) as a planar `Face` in the part's own XY plane.
  Ribs: local x runs aft along the chord from the leading edge, local y is the airfoil thickness
  direction (up for wing / horizontal-tail ribs, aircraft +y for fin ribs), origin at the leading edge.
- `KitPart.location` maps local coordinates into the aircraft frame. The part's 3D solid is the outline
  extruded along local +Z by its sheet thickness (`settings.sheet_for(part.family).thickness_m`), then
  moved by `location`. Wing and horizontal-tail ribs use a +90 deg rotation about x, so local +Z points
  to aircraft -y; left-panel parts reuse the right-panel outline with the location's y negated.
- Pieces of one split rib (e.g. `WR-07` and its aileron part `WR-07A`) share the same location, so the
  pieces sit side by side exactly as cut from the full rib.
"""

from dataclasses import dataclass
from typing import Literal

import build123d as bd

from src.schemas.fabrication import PartFamily, TubeStock
from src.tools.fabrication.tubes import KitUnavailable, SparPlan

__all__ = ["Kit", "KitPart", "KitUnavailable", "LinearStock", "TubePart", "Vec3"]

Vec3 = tuple[float, float, float]
TubeRole = Literal["spar", "socket", "tail_boom", "tail_spar", "lift_boom", "pin"]
LinearKind = Literal["stringer", "le", "te"]


@dataclass(frozen=True)
class KitPart:
    label: str
    family: PartFamily
    outline: bd.Face
    location: bd.Location
    quantity: int = 1

    @property
    def area_m2(self) -> float:
        """Nominal outline area (holes removed) of one piece (m^2)."""
        return self.outline.area


@dataclass(frozen=True)
class TubePart:
    label: str
    role: TubeRole
    tube: TubeStock
    length_m: float
    start_m: Vec3
    axis: Vec3


@dataclass(frozen=True)
class LinearStock:
    label: str
    kind: LinearKind
    section_area_m2: float
    density_kg_m3: float
    length_m: float
    start_m: Vec3
    end_m: Vec3


@dataclass(frozen=True)
class Kit:
    parts: tuple[KitPart, ...]
    tubes: tuple[TubePart, ...]
    linear: tuple[LinearStock, ...]
    film_area_m2: float
    fuselage_cover_area_m2: float
    spar: SparPlan
    joins: tuple[str, ...]  # human-readable, e.g. "C-1a + C-1b at x = 0.512 m"
