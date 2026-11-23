"""Hand-launch conditions shared by the SITL bridge (initial state) and the pilot (take-off check).

Kept free of pymavlink/jsbsim imports so both sides can use it. `flight_test` re-exports the constants."""

from typing import Final

HAND_LAUNCH_SPEED_MS: Final[float] = 10.0  # throw speed along the body x axis
HAND_LAUNCH_HEIGHT_M: Final[float] = 2.0  # release height above the ground
HAND_LAUNCH_PITCH_DEG: Final[float] = 10.0  # nose-up attitude at release
