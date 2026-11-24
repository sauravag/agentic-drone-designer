"""Blade-element momentum propeller model: generic blade geometry, static and near-pitch-speed behaviour, and the
validation gate against UIUC APC thin-electric wind-tunnel data (spec 6.1)."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest

import src.tools.propulsion.bemt as bemt
from src.tools.propulsion.bemt import (BladeGeometry, PropModelError, PropPoint, bemt_point,
                                         generic_blade)

INCH_M: Final[float] = 0.0254
RHO: Final[float] = 1.225
MU: Final[float] = 1.81e-5
SPEED_OF_SOUND: Final[float] = 340.3
UIUC_DIR: Final[Path] = Path(__file__).parent / "data" / "uiuc"

PEAK_TOLERANCE: Final[float] = 0.10  # peak efficiency and J at peak (spec 6.1)
STATIC_CT_TOLERANCE: Final[float] = 0.15  # CT0 (spec 6.1)
NEAR_PITCH_SPEED_FRACTION: Final[float] = 0.95
# UIUC data show 26-29 % at 0.95 x pitch speed (APC E: zero thrust near 1.2 x pitch speed), so the plan's 0.10
# is unreachable for a model that passes the validation gate; 0.30 per controller ruling, ledger Task 1.
NEAR_PITCH_THRUST_RATIO_MAX: Final[float] = 0.30
J_SWEEP: Final[np.ndarray] = np.arange(0.30, 0.95, 0.01)
LIFT_ROTOR_TIP_MACH: Final[float] = 0.6


@dataclass(frozen=True)
class UiucDataset:
    name: str
    diameter_in: float
    pitch_in: float
    rpm: float
    static_file: str
    advance_files: tuple[str, ...]


UIUC_DATASETS: Final[tuple[UiucDataset, ...]] = (
    UiucDataset("apce_9x6", 9.0, 6.0, 5000.0, "apce_9x6_static_rd0987.txt",
                ("apce_9x6_rd0989_5013.txt", "apce_9x6_rd0990_5000.txt")),
    UiucDataset("apce_10x7", 10.0, 7.0, 5000.0, "apce_10x7_static_pg0811.txt",
                ("apce_10x7_pg0813_5018.txt", "apce_10x7_pg0814_5001.txt")),
    UiucDataset("apce_11x7", 11.0, 7.0, 5000.0, "apce_11x7_static_kt0534.txt",
                ("apce_11x7_kt0538_4997.txt", "apce_11x7_kt0539_4997.txt")),
)


def _load_table(file_name: str) -> np.ndarray:
    """Numeric rows of a UIUC text file (first line is the column header)."""
    return np.loadtxt(UIUC_DIR / file_name, skiprows=1, ndmin=2)


def _data_peak(dataset: UiucDataset) -> tuple[float, float]:
    """(peak efficiency, J at peak) over all advance-ratio rows of the dataset."""
    rows: np.ndarray = np.vstack([_load_table(f) for f in dataset.advance_files])
    best: int = int(np.argmax(rows[:, 3]))
    return float(rows[best, 3]), float(rows[best, 0])


def _data_static_ct(dataset: UiucDataset) -> float:
    """Static CT interpolated at the dataset rpm."""
    rows: np.ndarray = _load_table(dataset.static_file)
    return float(np.interp(dataset.rpm, rows[:, 0], rows[:, 1]))


def _model_point(dataset: UiucDataset, advance_ratio: float) -> PropPoint:
    diameter: float = dataset.diameter_in * INCH_M
    blade = generic_blade(diameter, dataset.pitch_in * INCH_M)
    speed: float = advance_ratio * dataset.rpm / 60.0 * diameter
    return bemt_point(blade, dataset.rpm, speed, RHO, MU, SPEED_OF_SOUND)


def _model_peak(dataset: UiucDataset) -> tuple[float, float]:
    etas: list[float] = [_model_point(dataset, float(j)).efficiency for j in J_SWEEP]
    best: int = int(np.argmax(etas))
    return etas[best], float(J_SWEEP[best])


def _model_static_ct(dataset: UiucDataset) -> float:
    thrust: float = _model_point(dataset, 0.0).thrust_n
    rev_per_s: float = dataset.rpm / 60.0
    return thrust / (RHO * rev_per_s ** 2 * (dataset.diameter_in * INCH_M) ** 4)


def test_generic_blade_has_constant_geometric_pitch() -> None:
    blade = generic_blade(0.254, 0.178)
    radius: float = blade.diameter_m / 2.0
    for station, beta in zip(blade.stations, blade.twist_rad):
        assert beta == pytest.approx(math.atan(blade.pitch_m / (2.0 * math.pi * station * radius)), rel=1e-12)
    assert len(blade.stations) == len(blade.chord_over_radius) and blade.n_blades == 2


def test_static_thrust_positive_and_efficiency_zero() -> None:
    p = bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, 1.225, 1.81e-5, 340.3)
    assert p.thrust_n > 0 and p.efficiency == 0.0
    assert p.shaft_power_w == pytest.approx(p.torque_nm * 6000 * 2.0 * math.pi / 60.0, rel=1e-12)


def test_thrust_falls_to_zero_near_pitch_speed() -> None:
    blade = generic_blade(0.254, 0.178)
    rpm: float = 6000.0
    static = bemt_point(blade, rpm, 0.0, RHO, MU, SPEED_OF_SOUND)
    near_pitch = bemt_point(blade, rpm, NEAR_PITCH_SPEED_FRACTION * blade.pitch_m * rpm / 60.0, RHO, MU,
                            SPEED_OF_SOUND)
    assert near_pitch.thrust_n < NEAR_PITCH_THRUST_RATIO_MAX * static.thrust_n


@pytest.mark.parametrize("dataset", UIUC_DATASETS, ids=lambda d: d.name)
def test_matches_uiuc_peak_efficiency_and_j(dataset: UiucDataset) -> None:
    eta_data, j_data = _data_peak(dataset)
    eta_model, j_model = _model_peak(dataset)
    assert abs(eta_model / eta_data - 1.0) <= PEAK_TOLERANCE, (eta_model, eta_data)
    assert abs(j_model / j_data - 1.0) <= PEAK_TOLERANCE, (j_model, j_data)


@pytest.mark.parametrize("dataset", UIUC_DATASETS, ids=lambda d: d.name)
def test_matches_uiuc_static_ct(dataset: UiucDataset) -> None:
    ct_model: float = _model_static_ct(dataset)
    ct_data: float = _data_static_ct(dataset)
    assert abs(ct_model / ct_data - 1.0) <= STATIC_CT_TOLERANCE, (ct_model, ct_data)


def test_rejects_non_physical_inputs() -> None:
    with pytest.raises(ValueError, match="airspeed"):
        bemt_point(generic_blade(0.254, 0.178), 6000, -1.0, RHO, MU, SPEED_OF_SOUND)
    with pytest.raises(ValueError, match="diameter"):
        generic_blade(0.0, 0.178)


def test_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        bemt_point(generic_blade(0.254, 0.178), math.nan, 0.0, RHO, MU, SPEED_OF_SOUND)
    with pytest.raises(ValueError, match="finite"):
        bemt_point(generic_blade(0.254, 0.178), 6000, math.inf, RHO, MU, SPEED_OF_SOUND)


def _constant_polar(cl: float, cd: float) -> bemt.SectionCoefficients:
    def coefficients(reynolds: np.ndarray, mach: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(alpha, cl), np.full_like(alpha, cd)
    return coefficients


def test_raises_when_inflow_has_no_sign_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # Negative lift everywhere at zero airspeed: the residual is positive on the whole bracket.
    monkeypatch.setattr(bemt, "_section_coefficients", lambda airfoil: _constant_polar(-1.0, 0.0))
    with pytest.raises(PropModelError, match="no sign change"):
        bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, RHO, MU, SPEED_OF_SOUND)


def test_raises_on_non_physical_relative_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bemt, "_relative_speed", lambda stations, omega, phi, section: -np.ones_like(phi))
    with pytest.raises(PropModelError, match="relative speed"):
        bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, RHO, MU, SPEED_OF_SOUND)


def test_raises_outside_polar_grid() -> None:
    blade = generic_blade(0.254, 0.178)
    supersonic_tip_rpm: float = 0.9 * SPEED_OF_SOUND / (math.pi * blade.diameter_m) * 60.0
    with pytest.raises(PropModelError, match="Mach"):
        bemt_point(blade, supersonic_tip_rpm, 0.0, RHO, MU, SPEED_OF_SOUND)
    with pytest.raises(PropModelError, match="Reynolds"):
        bemt_point(blade, 6000, 0.0, RHO, 1.0, SPEED_OF_SOUND)  # viscous enough to drop Re below the grid


def test_windmilling_point_is_flagged() -> None:
    blade = generic_blade(0.254, 0.178)
    rpm: float = 5000.0
    cruise = bemt_point(blade, rpm, 0.6 * rpm / 60.0 * blade.diameter_m, RHO, MU, SPEED_OF_SOUND)
    windmill = bemt_point(blade, rpm, 1.1 * rpm / 60.0 * blade.diameter_m, RHO, MU, SPEED_OF_SOUND)
    assert not cruise.is_windmilling and cruise.efficiency > 0.0
    assert windmill.is_windmilling and windmill.shaft_power_w <= 0.0 and windmill.efficiency == 0.0


@pytest.mark.parametrize(("stations", "chords", "message"), [
    ((0.5,), (0.1,), "stations"),
    ((0.3, 0.6, 0.9), (0.1, 0.1), "one chord per station"),
    ((0.6, 0.3, 0.9), (0.1, 0.1, 0.1), "increase"),
    ((0.3, 0.6, 0.98), (0.1, 0.1, 0.1), "increase"),
    ((0.3, 0.6, 0.9), (0.1, -0.1, 0.1), "chords"),
])
def test_blade_geometry_rejects_inconsistent_stations(stations: tuple[float, ...], chords: tuple[float, ...],
                                                      message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BladeGeometry(diameter_m=0.254, pitch_m=0.178, n_blades=2, stations=stations, chord_over_radius=chords)


def test_lift_rotor_hover_is_finite_with_figure_of_merit_below_one() -> None:
    diameter: float = 2.0
    rpm: float = LIFT_ROTOR_TIP_MACH * SPEED_OF_SOUND / (math.pi * diameter) * 60.0
    p = bemt_point(generic_blade(diameter, 0.8), rpm, 0.0, RHO, MU, SPEED_OF_SOUND)
    disk_area: float = math.pi * diameter ** 2 / 4.0
    figure_of_merit: float = p.thrust_n ** 1.5 / (p.shaft_power_w * math.sqrt(2.0 * RHO * disk_area))
    assert math.isfinite(p.thrust_n) and p.thrust_n > 0.0 and not p.is_windmilling
    assert 0.0 < figure_of_merit < 1.0


def test_raises_when_relative_speed_does_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bemt, "SPEED_REL_TOLERANCE", 0.0)  # no pass can change W by less than zero
    with pytest.raises(PropModelError, match="did not converge"):
        bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, RHO, MU, SPEED_OF_SOUND)


def test_raises_on_non_finite_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    def nan_state(blade: BladeGeometry, stations: bemt._Stations, *args: float) -> tuple[bemt._Section, np.ndarray]:
        nan: np.ndarray = np.full_like(stations.r_frac, math.nan)
        return bemt._Section(cn=nan, ct=nan, loss=np.ones_like(nan)), np.ones_like(nan)
    monkeypatch.setattr(bemt, "_converged_state", nan_state)
    with pytest.raises(PropModelError, match="non-finite loads"):
        bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, RHO, MU, SPEED_OF_SOUND)


def test_raises_when_representative_section_reynolds_below_floor() -> None:
    # A 9x6 at 800 rpm: Re ~8e3 at r/R 0.75, while every station stays inside the 1e3 grid floor.
    with pytest.raises(PropModelError, match="r/R 0.75"):
        bemt_point(generic_blade(9 * INCH_M, 6 * INCH_M), 800, 0.0, RHO, MU, SPEED_OF_SOUND)
