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
G_TO_KG: Final[float] = 1e-3
OZ_TO_KG: Final[float] = 0.028349523125
MIN_TO_S: Final[float] = 60.0
DAY_TO_S: Final[float] = 86_400.0
MILE_TO_M: Final[float] = 1_609.344
NMI_TO_M: Final[float] = 1_852.0
FT_TO_M: Final[float] = 0.3048
KMH_TO_MS: Final[float] = 1.0 / 3.6
MPH_TO_MS: Final[float] = 0.44704
KW_TO_W: Final[float] = 1_000.0

# Atmosphere (ISA troposphere) and temperature scales
ISA_SEA_LEVEL_TEMPERATURE_C: Final[float] = 15.0
ISA_LAPSE_RATE_C_PER_M: Final[float] = 0.0065
FAHRENHEIT_OFFSET: Final[float] = 32.0
FAHRENHEIT_PER_CELSIUS: Final[float] = 1.8
