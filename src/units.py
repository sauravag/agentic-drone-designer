"""Unit-conversion constants. Use ONLY at system I/O boundaries; internals are strict SI."""

from typing import Final

G0: Final[float] = 9.80665  # standard gravity, m/s^2
WH_TO_J: Final[float] = 3_600.0
KWH_TO_J: Final[float] = 3.6e6
INCH_TO_M: Final[float] = 0.0254
KNOT_TO_MS: Final[float] = 0.514444
KM_TO_M: Final[float] = 1_000.0
HOUR_TO_S: Final[float] = 3_600.0
LB_TO_KG: Final[float] = 0.45359237
