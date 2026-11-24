"""Schema validation tests: system-boundary inputs must fail fast on non-physical values."""

import pytest
from pydantic import ValidationError

from src.schemas.components import Battery, CruiseMotor, Esc, Payload, Propeller, SelectedBOM, VtolMotor
from src.schemas.propulsion import OperatingPoint, PropulsionMatch
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import CADArtifacts, SizingOutput

VALID_REQ: dict = {
    "mission_name": "Test ISR",
    "payload_mass_kg": 3.5,
    "payload_power_w": 60.0,
    "endurance_hours": 6.0,
    "combat_radius_km": 80.0,
    "cruise_speed_ms": 25.0,
}


def test_requirements_defaults_applied() -> None:
    req = OperationalRequirements(**VALID_REQ)
    assert req.propulsion_type == "auto"
    assert req.airframe_topology == "auto"
    assert req.runway_available is False
    assert req.service_ceiling_m == 3000.0
    assert req.endurance_s == pytest.approx(6.0 * 3600.0)
    assert req.combat_radius_m == pytest.approx(80_000.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_mass_kg", 0.0),
        ("payload_mass_kg", 250.0),
        ("endurance_hours", 0.05),
        ("cruise_speed_ms", 5.0),
        ("combat_radius_km", 1.0),
        ("group_constraint", 4),
        ("propulsion_type", "nuclear"),
    ],
)
def test_requirements_reject_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        OperationalRequirements(**{**VALID_REQ, field: value})


def test_requirements_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        OperationalRequirements(**VALID_REQ, wing_span_ft=10.0)


def test_requirements_are_immutable() -> None:
    req = OperationalRequirements(**VALID_REQ)
    with pytest.raises(ValidationError):
        req.payload_mass_kg = 5.0  # type: ignore[misc]


def test_cruise_motor_continuous_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="continuous_power_w"):
        CruiseMotor(
            name="bad", mass_kg=0.1, max_power_w=100.0, continuous_power_w=200.0,
            kv=900.0, voltage_v=14.8, diameter_m=0.03, length_m=0.03,
        )


def test_battery_rejects_millimetre_dimensions() -> None:
    with pytest.raises(ValidationError):
        Battery(
            name="mm leak", chemistry="li_ion", capacity_wh=144.0, mass_kg=0.72, max_c_rate=3.0,
            voltage_v=14.4, length_m=150.0, width_m=0.07, height_m=0.05,
        )


def test_battery_rejects_non_physical_specific_energy() -> None:
    with pytest.raises(ValidationError, match="specific energy"):
        Battery(
            name="unobtainium", chemistry="li_ion", capacity_wh=1000.0, mass_kg=0.5, max_c_rate=3.0,
            voltage_v=14.4, length_m=0.15, width_m=0.07, height_m=0.05,
        )


def _bom_parts() -> dict:
    return {
        "cruise_motor": CruiseMotor(name="m", mass_kg=0.1, max_power_w=550.0, continuous_power_w=400.0,
                                    kv=900.0, voltage_v=14.8, diameter_m=0.035, length_m=0.03),
        "cruise_propeller": Propeller(name="p", diameter_m=0.33, pitch_m=0.2, mass_kg=0.035,
                                      max_power_w=1000.0, application="cruise"),
        "battery": Battery(name="b", chemistry="li_ion", capacity_wh=144.0, mass_kg=0.72, max_c_rate=3.0,
                           voltage_v=14.4, length_m=0.15, width_m=0.07, height_m=0.05),
        "payload": Payload(name="g", sensor_type="eo_ir", mass_kg=1.0, power_w=18.0,
                           length_m=0.1, width_m=0.1, height_m=0.1),
    }


VTOL_MOTOR = VtolMotor(name="v", mass_kg=0.15, max_thrust_n=45.0, max_power_w=750.0,
                       voltage_v=22.2, diameter_m=0.046, length_m=0.03)
VTOL_PROP = Propeller(name="vp", diameter_m=0.46, pitch_m=0.15, mass_kg=0.05, max_power_w=800.0, application="vtol")


@pytest.mark.parametrize(
    "vtol",
    [
        {"vtol_motor": VTOL_MOTOR, "vtol_propeller": VTOL_PROP, "vtol_motor_count": 0},  # rotors with no count
        {"vtol_motor_count": 4},  # count with no rotors
        {"vtol_motor": VTOL_MOTOR, "vtol_motor_count": 4},  # motor without propeller
    ],
)
def test_bom_rejects_inconsistent_vtol_set(vtol: dict) -> None:
    with pytest.raises(ValidationError, match="VTOL"):
        SelectedBOM(**_bom_parts(), **vtol)


def test_bom_accepts_complete_vtol_set() -> None:
    bom = SelectedBOM(**_bom_parts(), vtol_motor=VTOL_MOTOR, vtol_propeller=VTOL_PROP, vtol_motor_count=4)
    assert bom.total_hardware_mass_kg == pytest.approx(0.1 + 0.035 + 0.72 + 1.0 + 4 * (0.15 + 0.05))


def _sizing_kwargs(**overrides: object) -> dict:
    base: dict = dict(
        mtow_kg=10.0, empty_mass_kg=5.0, payload_mass_kg=1.0, fuel_mass_kg=0.0, battery_mass_kg=4.0,
        avionics_mass_kg=0.4, avionics_power_w=14.0,
        wing_area_m2=0.8, wingspan_m=3.0, mean_chord_m=0.27, aspect_ratio=11.25,
        fuselage_length_m=1.5, fuselage_diameter_m=0.2, cruise_power_w=300.0,
        hover_power_w=0.0, v_stall_ms=14.0, v_loiter_ms=18.0, assumed_ld_cruise=15.0,
        dod_group=2, propulsion_type="electric", airframe_topology="fixed_wing", airfoil="sd7037",
        cl_max=1.2, cd0=0.03, oswald_e=0.8, horizontal_tail_area_m2=0.12, vertical_tail_area_m2=0.06,
        tail_arm_m=0.8, battery_energy_j=2.9e6, battery_peak_power_w=900.0, generator_power_w=0.0,
        cruise_motor_power_w=600.0, vtol_thrust_per_motor_n=0.0, rotor_diameter_m=0.0,
        energy_reserve_fraction=0.2, phases=(),
    )
    return {**base, **overrides}


def test_sizing_output_accepts_closed_mass_budget() -> None:
    assert SizingOutput(**_sizing_kwargs()).mtow_kg == 10.0


def test_sizing_output_requires_mass_closure() -> None:
    with pytest.raises(ValidationError, match="mass closure"):
        SizingOutput(**_sizing_kwargs(battery_mass_kg=5.0))


def test_cad_artifacts_inertia_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CADArtifacts(
            oml_step_path="a.step", assembly_step_path="b.step", oml_vsp3_path="a.vsp3",
            x_cg_m=0.5, static_margin=0.1, collision_count=0, as_built_mass_kg=10.0,
            ixx_kgm2=-1.0, iyy_kgm2=1.0, izz_kgm2=1.0, ixz_kgm2=0.0,
        )


def test_old_sizing_records_without_cruise_speed_still_validate() -> None:
    sizing = SizingOutput(**_sizing_kwargs())
    assert sizing.v_cruise_ms is None
    assert SizingOutput(**_sizing_kwargs(v_cruise_ms=25.0)).v_cruise_ms == 25.0


CRUISE_ESC = Esc(name="esc40", mass_kg=0.04, continuous_current_a=40.0, burst_current_a=50.0, max_voltage_v=26.0,
                 efficiency=0.96, length_m=0.05, width_m=0.025, height_m=0.01)
LIFT_ESC = Esc(name="esc20", mass_kg=0.02, continuous_current_a=20.0, burst_current_a=25.0, max_voltage_v=26.0,
               efficiency=0.96, length_m=0.04, width_m=0.02, height_m=0.01)


def test_bom_hardware_mass_counts_the_escs() -> None:
    bom = SelectedBOM(**_bom_parts(), vtol_motor=VTOL_MOTOR, vtol_propeller=VTOL_PROP, vtol_motor_count=4,
                      cruise_esc=CRUISE_ESC, vtol_esc=LIFT_ESC)
    assert bom.total_hardware_mass_kg == pytest.approx(
        0.1 + 0.035 + 0.04 + 0.72 + 1.0 + 4 * (0.15 + 0.05 + 0.02))


def test_legacy_bom_without_escs_loads() -> None:
    legacy = SelectedBOM(**_bom_parts()).model_dump()
    legacy.pop("cruise_esc")
    legacy.pop("vtol_esc")
    legacy.pop("total_hardware_mass_kg")
    bom = SelectedBOM.model_validate(legacy)
    assert bom.cruise_esc is None and bom.vtol_esc is None


def test_bom_rejects_a_lift_esc_without_rotors() -> None:
    with pytest.raises(ValidationError, match="VTOL"):
        SelectedBOM(**_bom_parts(), vtol_esc=LIFT_ESC)


def test_legacy_sizing_record_without_match_loads() -> None:
    record = SizingOutput(**_sizing_kwargs()).model_dump(mode="json")
    record.pop("propulsion_match")
    assert SizingOutput.model_validate(record).propulsion_match is None


def test_sizing_record_round_trips_its_propulsion_match() -> None:
    match = PropulsionMatch(
        cell_name="c", series=4, cruise_motor="m", cruise_propeller="p", cruise_esc="esc40",
        operating_points=(OperatingPoint(flight_point="Loiter", propulsor="cruise", rpm=5000.0, current_a=8.0,
                                         throttle=0.6, prop_efficiency=0.7, motor_efficiency=0.85, tip_mach=0.3,
                                         electrical_power_w=120.0, sagged=False),),
        efficiencies={"Loiter": 0.6}, propulsion_mass_kg=0.2)
    sizing = SizingOutput(**_sizing_kwargs(propulsion_match=match))
    again = SizingOutput.model_validate_json(sizing.model_dump_json())
    assert again.propulsion_match == match and again.propulsion_match.set_key == (4, "m", "p", None, None)
