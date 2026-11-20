"""Laser-cut airframe kit model: flat ply parts, stock tubes and linear stock (SI units, metres).

Frame convention (Tasks 3-9 rely on it):
- Aircraft frame: x aft from the nose, y right, z up.
- `KitPart.outline` is the nominal cut shape (no kerf) as a planar `Face` in the part's own XY plane.
  Ribs: local x runs aft along the chord from the leading edge, local y is the airfoil thickness
  direction (up for wing / horizontal-tail ribs, aircraft +y for fin ribs), origin at the leading edge.
- `KitPart.location` maps local coordinates into the aircraft frame. The part's 3D solid is the outline
  extruded along local +Z by its sheet thickness (`settings.sheet_for(part.family).thickness_m`), then
  moved by `location`. Wing and horizontal-tail ribs use a +90 deg rotation about x, so local +Z points
  to aircraft -y. Their location origin sits t/2 outboard of the rib station (t = sheet thickness), so
  the slab spans [y - t/2, y + t/2], centred on the station; a panel's root rib (`WR-01`, `HR-01`)
  instead lies just outboard of its root station, spanning [y_root, y_root + t], so it stays outside the
  fuselage skin or tail boom. Left-panel parts reuse the right-panel outline and are placed so their
  solids are exact y-mirrors of the right-panel solids (for ribs the origin y is t - y_right).
  Fin ribs (`V-…`) and flat plates are unrotated: they extrude upward (+z) from their origin.
- Pieces of one split rib (e.g. `WR-07` and its aileron part `WR-07A`) share the same location, so the
  pieces sit side by side exactly as cut from the full rib.
"""

from dataclasses import dataclass
from typing import Final, Literal

import build123d as bd

from src.schemas.fabrication import PartFamily, TubeStock
from src.tools.fabrication.tubes import KitUnavailable, Overwrap, SparPlan

__all__ = ["Box2", "CoverSurface", "Kit", "KitPart", "KitUnavailable", "LinearStock", "OVERWRAP_LABEL", "Overwrap",
           "TubePart", "Vec3"]

Vec3 = tuple[float, float, float]
Box2 = tuple[float, float, float, float]  # (x0, y0, x1, y1) in a part's local XY
TubeRole = Literal["spar", "socket", "tail_boom", "tail_spar", "lift_boom", "pin", "overwrap"]
OVERWRAP_LABEL: Final[str] = "OW-1"
LinearKind = Literal["stringer", "le", "te"]
CoverKind = Literal["film", "fuselage"]


@dataclass(frozen=True)
class KitPart:
    label: str
    family: PartFamily
    outline: bd.Face
    location: bd.Location
    quantity: int = 1
    openings: tuple[Box2, ...] = ()  # component openings (local XY boxes) a bed split may cut straight through

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
class CoverSurface:
    """Covering laid uniformly over `surface` (aircraft frame). `area_m2` is the material area: a lifting
    surface's film covers both faces (2 x planform), so it is twice the planform `surface`'s area."""

    label: str
    kind: CoverKind
    surface: bd.Shape
    area_m2: float


@dataclass(frozen=True)
class Kit:
    parts: tuple[KitPart, ...]
    tubes: tuple[TubePart, ...]
    linear: tuple[LinearStock, ...]
    covering: tuple[CoverSurface, ...]  # film and fuselage cover: area and where it lies (mass placement)
    spar: SparPlan
    joins: tuple[str, ...]  # human-readable, e.g. "C-1a + C-1b at x = 0.512 m"
    overwrap: Overwrap  # carbon cloth over the centreline socket mitre (R24)

    def overwrap_halves(self) -> tuple[TubePart, ...]:
        """The overwrap as one tube per socket (wrapped OD over the socket OD), each lapping its socket from the
        centreline: where it lies, for mass placement and as a packing obstacle."""
        wrap = self.overwrap
        bore = TubeStock(od_m=wrap.od_m, id_m=wrap.od_m - 2.0 * wrap.layers * wrap.layer_thickness_m)
        return tuple(TubePart(OVERWRAP_LABEL, "overwrap", bore, wrap.lap_m, t.start_m, t.axis)
                     for t in self.tubes if t.role == "socket")

    def _covered_m2(self, kind: CoverKind) -> float:
        return sum(c.area_m2 for c in self.covering if c.kind == kind)

    @property
    def film_area_m2(self) -> float:
        """Film material area: both faces of the wing and tails (2 x planform)."""
        return self._covered_m2("film")

    @property
    def fuselage_cover_area_m2(self) -> float:
        """Fuselage cover area: the pod's lateral area."""
        return self._covered_m2("fuselage")
