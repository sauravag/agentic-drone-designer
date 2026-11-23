"""One landing-gear rule: runway rotation clearance, VTOL level clearance, hand launch has none."""
import math
from typing import Literal

import pytest

from src.tools.gear import (GEAR_FRACTION, PROP_TIP_CLEARANCE_FRACTION, ROTATION_ANGLE_DEG, GearInputs,
                            PropellerDisc, gear_mass_kg, landing_gear)

PROP_D_M = 0.279
TRACTOR_LEG_M = 0.175  # G1: 1.25 x propeller radius


def _inputs(launch: Literal["runway", "hand", "vertical"] = "runway", prop: PropellerDisc | None = None,
            x_cg: float = 0.355) -> GearInputs:
    return GearInputs(launch, fuselage_length_m=0.878, fuselage_diameter_m=0.166, mean_chord_m=0.25, wingspan_m=2.4,
                      x_cg_m=x_cg, propeller=prop or PropellerDisc(0.0, 0.0, PROP_D_M))


def test_unknown_topology_raises_a_named_value_error() -> None:
    with pytest.raises(ValueError, match="flying_saucer"):
        gear_mass_kg(landing_gear(_inputs()), "flying_saucer", 10.0)


def test_gear_fractions_are_immutable() -> None:
    with pytest.raises(TypeError):
        GEAR_FRACTION["fixed_wing"] = 0.5  # type: ignore[index]


def test_hand_launch_gear_has_no_contacts_and_zero_mass() -> None:
    gear = landing_gear(_inputs(launch="hand"))
    assert gear.kind == "none" and gear.contacts == () and gear.leg_height_m == 0.0
    assert gear_mass_kg(gear, "fixed_wing", 5.0) == 0.0


def test_tractor_runway_gear_matches_today() -> None:
    gear = landing_gear(_inputs(launch="runway"))
    assert gear.leg_height_m == pytest.approx(TRACTOR_LEG_M, abs=1e-3)
    assert gear_mass_kg(gear, "fixed_wing", 5.0) == pytest.approx(0.04 * 5.0)


def test_boom_end_pusher_runway_gear_clears_at_rotation() -> None:
    prop = PropellerDisc(x_m=1.097, z_m=0.0, diameter_m=PROP_D_M)
    gear = landing_gear(_inputs(launch="runway", prop=prop))
    main, theta, radius = gear.contacts[1], math.radians(ROTATION_ANGLE_DEG), PROP_D_M / 2
    clearance = (prop.z_m - radius - main.z_m) * math.cos(theta) - (prop.x_m - main.x_m) * math.sin(theta)
    assert clearance == pytest.approx(PROP_TIP_CLEARANCE_FRACTION * radius, abs=1e-6)
    assert gear.leg_height_m > TRACTOR_LEG_M
    assert gear_mass_kg(gear, "fixed_wing", 5.0) == pytest.approx(0.04 * 5.0 * gear.leg_height_m / TRACTOR_LEG_M,
                                                                  rel=5e-3)


def test_vertical_gear_level_clearance() -> None:
    high = PropellerDisc(x_m=1.0, z_m=0.12, diameter_m=0.5)  # above the axis: today's leg already clears
    low = PropellerDisc(x_m=1.0, z_m=-0.1, diameter_m=PROP_D_M)  # below the axis: needs a taller leg
    for disc in (high, low):
        gear = landing_gear(_inputs(launch="vertical", prop=disc))
        tip_height = disc.z_m - disc.diameter_m / 2 + gear.leg_height_m
        assert tip_height >= PROP_TIP_CLEARANCE_FRACTION * disc.diameter_m / 2 - 1e-9
    assert landing_gear(_inputs(launch="vertical", prop=low)).leg_height_m > TRACTOR_LEG_M


def test_main_gear_keeps_the_tip_back_margin_at_the_taller_leg() -> None:
    gear = landing_gear(_inputs(prop=PropellerDisc(1.097, 0.0, PROP_D_M)))
    main = gear.contacts[1]
    assert math.degrees(math.atan2(main.x_m - 0.355, gear.leg_height_m)) >= 15.0 - 1e-9
