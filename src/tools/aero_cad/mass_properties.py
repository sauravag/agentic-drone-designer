"""Mass, CG and inertia tensor build-up (SI, CAD axes: x aft, y right, z up).

Tensors use the tensor convention (off-diagonals = -sum m*x*y), matching OpenCascade's
MatrixOfInertia. `body_inertia` converts to the flight-dynamics product Ixz = +sum m*x*z, which is
identical in CAD axes and body axes (x fwd, z down) because both x and z flip sign.
"""

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

PD_RELATIVE_TOL: Final[float] = 1e-12
BOX_INERTIA_DIVISOR: Final[float] = 12.0


@dataclass(frozen=True)
class MassItem:
    name: str
    mass_kg: float
    cg_m: Vec3
    inertia_kgm2: Mat3  # about the item's own CG


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    cg_m: Vec3
    inertia_kgm2: Mat3  # about the combined CG


def _mat(a: np.ndarray) -> Mat3:
    return tuple(tuple(float(v) for v in row) for row in a)  # type: ignore[return-value]


def _vec(a: Sequence[float]) -> Vec3:
    return (float(a[0]), float(a[1]), float(a[2]))


def point_item(name: str, mass_kg: float, position_m: Sequence[float]) -> MassItem:
    return MassItem(name, mass_kg, _vec(position_m), _mat(np.zeros((3, 3))))


def box_item(name: str, mass_kg: float, center_m: Sequence[float], size_m: Sequence[float]) -> MassItem:
    lx, ly, lz = size_m
    diag = mass_kg / BOX_INERTIA_DIVISOR * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
    return MassItem(name, mass_kg, _vec(center_m), _mat(np.diag(diag)))


def cylinder_item(name: str, mass_kg: float, center_m: Sequence[float], length_m: float, diameter_m: float) -> MassItem:
    """Solid cylinder with its axis along x."""
    r2: float = (diameter_m / 2) ** 2
    axial: float = mass_kg * r2 / 2
    transverse: float = mass_kg * (3 * r2 + length_m**2) / BOX_INERTIA_DIVISOR
    return MassItem(name, mass_kg, _vec(center_m), _mat(np.diag([axial, transverse, transverse])))


def _occ_item(name: str, props: GProp_GProps, density: float) -> MassItem:
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()
    tensor = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) * density
    return MassItem(name, props.Mass() * density, (c.X(), c.Y(), c.Z()), _mat(tensor))


def solid_item(name: str, shape: Any, density: float) -> MassItem:
    """Exact volume properties of a B-rep solid (shape exposes OCC `.wrapped`)."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props)
    return _occ_item(name, props, density)


def shell_item(name: str, shape: Any, areal_density: float) -> MassItem:
    """Thin-shell properties of every face of `shape` at `areal_density` kg/m^2."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape.wrapped, props)
    return _occ_item(name, props, areal_density)


def combine(items: Sequence[MassItem]) -> MassProperties:
    """Total mass, CG and inertia about the combined CG (parallel-axis theorem)."""
    total: float = sum(i.mass_kg for i in items)
    if not items or total <= 0:
        raise ValueError("combine() needs at least one item with positive total mass")
    cg = sum(i.mass_kg * np.array(i.cg_m) for i in items) / total
    tensor = np.zeros((3, 3))
    for item in items:
        d = np.array(item.cg_m) - cg
        tensor += np.array(item.inertia_kgm2) + item.mass_kg * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return MassProperties(total, _vec(cg), _mat(tensor))


def body_inertia(props: MassProperties) -> tuple[float, float, float, float]:
    """(Ixx, Iyy, Izz, Ixz) with Ixz = +sum m*x*z (JSBSim / ArduPilot product convention)."""
    t = props.inertia_kgm2
    return t[0][0], t[1][1], t[2][2], -t[0][2]


def is_positive_definite(matrix: Any) -> bool:
    a = np.asarray(matrix, dtype=float)
    if not np.allclose(a, a.T):
        return False
    eig = np.linalg.eigvalsh(a)
    return bool(eig.min() > PD_RELATIVE_TOL * max(abs(eig.max()), 1.0))
