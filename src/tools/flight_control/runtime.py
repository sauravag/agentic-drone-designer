"""Lightweight SITL runtime checks and errors, importable without pymavlink/JSBSim (used by the pipeline)."""

import shutil
from typing import Final

ARDUPLANE_BINARY: Final[str] = "arduplane"
# SRS 4.4 SITL acceptance limits (shared by the validator, the pipeline warnings and the UI)
MAX_OVERSHOOT: Final[float] = 0.20
MAX_SETTLING_S: Final[float] = 3.0
MAX_TRANSITION_ALT_LOSS_M: Final[float] = 15.0


class SitlUnavailableError(RuntimeError):
    """arduplane is not installed (run inside the ISR-Forge Docker image)."""


class SitlValidationError(RuntimeError):
    """No SITL attempt completed its flight (every attempt crashed or timed out)."""


def sitl_available(binary: str = ARDUPLANE_BINARY) -> bool:
    return shutil.which(binary) is not None
