"""Brushless motor equivalent circuit: V = I*Rm + rpm/KV, I = torque/Kt + I0, Kt = 60/(2*pi*KV)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPoint:
    rpm: float
    current_a: float
    voltage_v: float
    shaft_power_w: float
    efficiency: float  # shaft power / electrical input power (0 at no load)


def motor_point(kv_rpm_per_v: float, rm_ohm: float, i0_a: float, rpm: float, torque_nm: float) -> MotorPoint:
    """Operating point of a motor delivering `torque_nm` at `rpm`."""
    for name, value in (("kv_rpm_per_v", kv_rpm_per_v), ("rm_ohm", rm_ohm)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if i0_a < 0 or rpm < 0 or torque_nm < 0 or not all(map(math.isfinite, (i0_a, rpm, torque_nm))):
        raise ValueError("i0_a, rpm and torque_nm must be finite and non-negative")
    kt: float = 60.0 / (2.0 * math.pi * kv_rpm_per_v)  # N*m per A
    current: float = torque_nm / kt + i0_a
    voltage: float = current * rm_ohm + rpm / kv_rpm_per_v
    shaft: float = torque_nm * rpm * 2.0 * math.pi / 60.0
    input_power: float = voltage * current
    # Zero input power only happens at rest with i0 == 0: no power in or out, so efficiency 0 is the
    # physical value (nothing is converted), not a masked failure. Inputs are validated above.
    efficiency: float = shaft / input_power if input_power > 0 else 0.0
    return MotorPoint(rpm, current, voltage, shaft, efficiency)
