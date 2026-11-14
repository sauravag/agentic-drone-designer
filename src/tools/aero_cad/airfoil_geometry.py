"""Unit-chord airfoil coordinates shared by OpenVSP (OML sections) and Build123d (rib profiles).

Coordinates come from AeroSandbox (UIUC database / NACA generator), repanelled so every consumer
sees the same closed Selig-ordered loop: TE -> upper surface -> LE -> lower surface -> TE.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import aerosandbox as asb

POINTS_PER_SIDE: Final[int] = 60


@dataclass(frozen=True)
class AirfoilShape:
    name: str
    coordinates: tuple[tuple[float, float], ...]  # (x/c, z/c), Selig order
    thickness_ratio: float  # max t/c

    def to_asb(self) -> asb.Airfoil:
        return asb.Airfoil(name=self.name, coordinates=list(self.coordinates))


@lru_cache(maxsize=16)
def load_airfoil(name: str) -> AirfoilShape:
    """Repanelled airfoil by UIUC/NACA name; raises ValueError for unknown names."""
    try:
        foil: asb.Airfoil = asb.Airfoil(name).repanel(n_points_per_side=POINTS_PER_SIDE)
    except Exception as exc:  # AeroSandbox raises a variety of errors for unknown names
        raise ValueError(f"unknown airfoil {name!r}") from exc
    if foil.coordinates is None:
        raise ValueError(f"unknown airfoil {name!r}")
    coords = tuple((float(x), float(z)) for x, z in foil.coordinates)
    return AirfoilShape(name=name, coordinates=coords, thickness_ratio=float(foil.max_thickness()))


def camber_at(shape: AirfoilShape, x_over_c: float) -> float:
    """Mean-line height z/c at chordwise station x/c."""
    return float(shape.to_asb().local_camber(x_over_c))


def write_selig_dat(shape: AirfoilShape, path: Path) -> Path:
    """Write a Selig-format .dat file (name line, then x z pairs) readable by OpenVSP."""
    lines: list[str] = [shape.name, *(f"{x:.6f} {z:.6f}" for x, z in shape.coordinates)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
