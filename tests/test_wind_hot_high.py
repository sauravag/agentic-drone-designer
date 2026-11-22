"""Design wind (a headwind on ingress and egress) and hot-and-high take-off (take-off altitude + ISA offset)."""

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.agent.agent_schemas import Altitude, Speed, Temperature
from src.agent.explorer import _cruise_scan_points
from src.agent.parsing import _optional_fields
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import PhaseResult, SizingOutput
from src.tools.sizing.aero_model import air_properties
from src.tools.sizing.mission_profile import CLIMB_RATE_MS
from src.tools.sizing.mission_solver import size_aircraft
from src.units import G0
from tests.agent_fakes import SRS_DRAFT
from tests.reference_designs import G1_ELECTRIC_FW, G2_ELECTRIC_VTOL

HOT_HIGH: dict[str, float] = {"takeoff_altitude_m": 2000.0, "isa_temperature_offset_c": 20.0,
                              "service_ceiling_m": 2500.0}
# The 3 h electric VTOL (G2) spirals at 2000 m ISA+20, so its hover check uses a milder site that still closes.
VTOL_SITE: dict[str, float] = {"takeoff_altitude_m": 1500.0, "isa_temperature_offset_c": 15.0,
                               "service_ceiling_m": 2000.0}
WIND_MS: float = 8.0


@pytest.fixture(scope="module")
def g1_hot_high() -> SizingOutput:
    return size_aircraft(G1_ELECTRIC_FW.model_copy(update=HOT_HIGH))


@pytest.fixture(scope="module")
def g2_hot_high() -> SizingOutput:
    return size_aircraft(G2_ELECTRIC_VTOL.model_copy(update=VTOL_SITE))


def _phase(s: SizingOutput, name: str) -> PhaseResult:
    return next(p for p in s.phases if p.name == name)


def _take_off_density(site: dict[str, float] = HOT_HIGH) -> float:
    return air_properties(site["takeoff_altitude_m"], site["isa_temperature_offset_c"])[0]


# ---------------------------------------------------------------- requirements ---


def test_new_fields_default_to_still_air_at_a_sea_level_standard_day() -> None:
    assert (G1_ELECTRIC_FW.design_wind_ms, G1_ELECTRIC_FW.takeoff_altitude_m,
            G1_ELECTRIC_FW.isa_temperature_offset_c) == (0.0, 0.0, 0.0)


def test_a_wind_at_or_above_the_cruise_speed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="design wind"):
        OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), "design_wind_ms": G1_ELECTRIC_FW.cruise_speed_ms})


def test_a_take_off_site_at_or_above_the_service_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationError, match="take-off altitude"):
        OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), "takeoff_altitude_m": 500.0,
                                   "service_ceiling_m": 500.0})


@pytest.mark.parametrize("field, bad", [("design_wind_ms", -1.0), ("isa_temperature_offset_c", 60.0),
                                        ("takeoff_altitude_m", 6000.0)])
def test_out_of_range_values_are_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), "service_ceiling_m": 8000.0, field: bad})


# --------------------------------------------------------------------- physics ---


def test_air_properties_apply_the_isa_temperature_offset() -> None:
    hot_rho, hot_mu = air_properties(2000.0, 20.0)
    std_rho, std_mu = air_properties(2000.0)
    assert hot_rho == pytest.approx(0.9369, rel=1e-3)
    assert hot_rho < std_rho and hot_mu > std_mu


def test_a_headwind_slows_ingress_and_egress_to_the_ground_speed() -> None:
    req = G1_ELECTRIC_FW.model_copy(update={"design_wind_ms": WIND_MS})
    s = size_aircraft(req)
    transit_s = req.combat_radius_m / (req.cruise_speed_ms - WIND_MS)
    assert _phase(s, "Ingress").duration_s == pytest.approx(transit_s)
    assert _phase(s, "Egress").duration_s == pytest.approx(transit_s)
    assert _phase(s, "Loiter").duration_s == pytest.approx(req.endurance_s)  # the loiter is not a transit
    assert s.battery_energy_j > size_aircraft(G1_ELECTRIC_FW).battery_energy_j


def test_the_climb_starts_at_the_take_off_altitude(g1_hot_high: SizingOutput) -> None:
    climb_s = (HOT_HIGH["service_ceiling_m"] - HOT_HIGH["takeoff_altitude_m"]) / CLIMB_RATE_MS
    assert _phase(g1_hot_high, "Climb").duration_s == pytest.approx(climb_s)


def test_the_stall_limit_holds_at_the_take_off_site(g1_hot_high: SizingOutput) -> None:
    rho_to, rho_sl = _take_off_density(), air_properties(0.0)[0]
    s = g1_hot_high
    assert s.v_stall_takeoff_ms == pytest.approx(
        math.sqrt(2.0 * s.mtow_kg * G0 / (rho_to * s.wing_area_m2 * s.cl_max)))
    assert s.v_stall_takeoff_ms <= G1_ELECTRIC_FW.max_stall_speed_ms + 1e-6
    # v_stall_ms stays the sea-level equivalent airspeed: ArduPilot's airspeed parameters are EAS
    assert s.v_stall_ms == pytest.approx(s.v_stall_takeoff_ms * math.sqrt(rho_to / rho_sl))
    assert s.wing_area_m2 > size_aircraft(G1_ELECTRIC_FW).wing_area_m2


def test_hover_uses_the_take_off_site_density(g2_hot_high: SizingOutput) -> None:
    s, a = g2_hot_high, DEFAULT_ASSUMPTIONS
    disk_m2 = 4.0 * math.pi * s.rotor_diameter_m**2 / 4.0
    hover_w = ((s.mtow_kg * G0) ** 1.5 / math.sqrt(2.0 * _take_off_density(VTOL_SITE) * disk_m2)
               / a.hover_figure_of_merit / a.lift_motor_efficiency)
    assert _phase(s, "VTOL_Takeoff").propulsive_power_w == pytest.approx(hover_w, rel=1e-6)
    assert _phase(s, "VTOL_Land").propulsive_power_w == pytest.approx(hover_w, rel=1e-6)


def test_still_air_at_sea_level_reproduces_the_reference_design() -> None:
    explicit = G2_ELECTRIC_VTOL.model_copy(update={"design_wind_ms": 0.0, "takeoff_altitude_m": 0.0,
                                                   "isa_temperature_offset_c": 0.0})
    assert size_aircraft(explicit) == size_aircraft(G2_ELECTRIC_VTOL)


# ------------------------------------------------------------- agent and chat ---


def test_the_cruise_scan_never_probes_a_speed_at_or_below_the_wind() -> None:
    req = G1_ELECTRIC_FW.model_copy(update={"design_wind_ms": 15.0})
    assert min(_cruise_scan_points(req)) > req.design_wind_ms


def test_the_parser_maps_wind_and_the_take_off_site() -> None:
    draft = SRS_DRAFT.model_copy(update={"design_wind": Speed(value=20, unit="kt"),
                                         "takeoff_altitude": Altitude(value=6562, unit="ft"),
                                         "isa_temperature_offset": 15.0})
    fields = _optional_fields(draft)
    assert fields["design_wind_ms"] == pytest.approx(10.289, rel=1e-3)
    assert fields["takeoff_altitude_m"] == pytest.approx(2000.1, rel=1e-3)
    assert fields["isa_temperature_offset_c"] == 15.0


def test_the_parser_turns_a_stated_air_temperature_into_an_isa_offset() -> None:
    """ISA at 2000 m is 15 - 6.5 x 2 = 2 degC, so 35 degC there is ISA+33; 95 degF is 35 degC."""
    for temperature in (Temperature(value=35, unit="C"), Temperature(value=95, unit="F")):
        draft = SRS_DRAFT.model_copy(update={"takeoff_altitude": Altitude(value=2, unit="km"),
                                             "takeoff_air_temperature": temperature})
        assert _optional_fields(draft)["isa_temperature_offset_c"] == pytest.approx(33.0)


def test_design_chat_can_change_the_wind_and_the_take_off_site() -> None:
    changes = ChangeSet(label="Hot and high, windy",
                        requirements=RequirementChanges(design_wind_ms=6.0, takeoff_altitude_m=1500.0, service_ceiling_m=2500.0,
                                                        isa_temperature_offset_c=10.0),
                        assumptions=AssumptionChanges(), knobs=KnobChanges())
    req = apply_changes(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, changes)[0]
    assert (req.design_wind_ms, req.takeoff_altitude_m, req.isa_temperature_offset_c) == (6.0, 1500.0, 10.0)


def test_vspaero_flies_the_design_day() -> None:
    builder = pytest.importorskip("src.tools.aero_cad.openvsp_builder")  # needs the compiled OpenVSP API (Docker)
    layout = SimpleNamespace(wing=SimpleNamespace(mac_m=0.3, x_ac_m=0.5, z_root_m=0.0))
    sizing = SimpleNamespace(v_loiter_ms=15.0)
    hot = builder.flight_condition(layout, sizing, 2000.0, 20.0)
    standard = builder.flight_condition(layout, sizing, 2000.0)
    assert hot.rho == pytest.approx(air_properties(2000.0, 20.0)[0])
    assert hot.re_cref < standard.re_cref and hot.mach < standard.mach  # thinner, warmer air
