"""Step-response metrics for the SITL doublet test (SRS 4.4: overshoot < 20 %, settling < 3 s).

Definitions (standard control-engineering):
* step size = commanded target - mean response over the PRE_STEP_S before the step;
* overshoot = max excursion beyond the target, in the step direction, / |step size| (0 if none);
* settling time = time from the step until the response last leaves the +-SETTLING_BAND x |step|
  band around the target (the whole window if it is still outside at the end).
"""

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

SETTLING_BAND: Final[float] = 0.05  # +-5 % of the step
PRE_STEP_S: Final[float] = 1.0
TARGET_SETTLE_S: Final[float] = 0.2  # ignore the reference's own transition when reading the target
MIN_STEP_DEG: Final[float] = 1.0


class StepAnalysisError(ValueError):
    """Too little telemetry, or the commanded step is too small to analyse."""


@dataclass(frozen=True)
class StepResponse:
    step_deg: float
    target_deg: float
    peak_deg: float
    overshoot: float  # fraction of the step
    settling_s: float

    def passes(self, max_overshoot: float, max_settling_s: float) -> bool:
        return self.overshoot < max_overshoot and self.settling_s < max_settling_s


def analyse_step(times_s: Sequence[float], response_deg: Sequence[float], reference_deg: Sequence[float],
                 step_time_s: float, window_s: float) -> StepResponse:
    t = np.asarray(times_s, dtype=float)
    y = np.asarray(response_deg, dtype=float)
    ref = np.asarray(reference_deg, dtype=float)
    pre = (t >= step_time_s - PRE_STEP_S) & (t < step_time_s)
    win = (t >= step_time_s) & (t <= step_time_s + window_s)
    steady_ref = win & (t >= step_time_s + TARGET_SETTLE_S)
    if pre.sum() < 2 or win.sum() < 2 or not steady_ref.any():
        raise StepAnalysisError(f"not enough samples around the step at t={step_time_s:.2f} s")
    target: float = float(np.median(ref[steady_ref]))
    step: float = target - float(y[pre].mean())
    if abs(step) < MIN_STEP_DEG:
        raise StepAnalysisError(f"step of {step:.2f} deg is too small to analyse")
    direction: float = float(np.sign(step))
    excursion: np.ndarray = (y[win] - target) * direction
    outside: np.ndarray = np.abs(y[win] - target) > SETTLING_BAND * abs(step)
    t_win: np.ndarray = t[win]
    if outside[-1]:
        settling: float = window_s
    elif outside.any():
        settling = float(t_win[np.nonzero(outside)[0][-1] + 1] - step_time_s)
    else:
        settling = 0.0
    return StepResponse(step_deg=step, target_deg=target, peak_deg=float(target + direction * excursion.max()),
                        overshoot=max(0.0, float(excursion.max())) / abs(step), settling_s=settling)


def altitude_loss_m(altitudes_m: Sequence[float], start_alt_m: float) -> float:
    """Largest drop below the starting altitude (0 if it never descends)."""
    if not len(altitudes_m):
        raise StepAnalysisError("no altitude samples during the transition")
    return max(0.0, start_alt_m - float(min(altitudes_m)))
