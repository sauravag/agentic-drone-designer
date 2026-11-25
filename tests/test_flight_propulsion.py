"""Task 7: the simulator, the JSBSim export and the ArduPilot parameters use the matched propulsor (spec 6.6).

The saved G1/G2 designs predate propulsion matching (no ESCs, no motor constants, no cell build): they exercise the
legacy path. `matched_g2` swaps catalogue parts with constants into G2 to exercise the matched path.
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.components import SelectedBOM
from src.tools.database.catalog_variants import ESC_EFFICIENCY_HV, ESC_EFFICIENCY_SMALL, MAX_CELL_V, motor_constants
from src.tools.database.packs import LI_ION_LOADED_MIN_V_CELL, LIPO_LOADED_MIN_V_CELL
from src.tools.database.seed_data import catalog_rows
from src.tools.flight_control import flight_model as fm
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control.flight_model import FlightModel, FlightModelError, build_flight_model
from src.tools.flight_control.jsbsim_exporter import lift_thrust_curve
from src.tools.flight_control.propulsion_plant import legacy_esc_efficiency, pack_series, plant_propulsor
from src.tools.propulsion.prop_tables import propeller_table
from src.tools.propulsion.propulsor import Propulsor, propulsor_from_parts
from tests.test_flight_control import load_design

MONOTONE_RTOL: Final[float] = 1e-9  # the propulsor's rpm-solve tolerance (propulsor.ROOT_RTOL)

SAMPLE_THROTTLES: tuple[float, ...] = (0.3, 0.6, 1.0)
SAMPLE_SPEEDS_MS: tuple[float, ...] = (0.0, 10.0, 22.0)


def _part(table: str, prefix: str) -> Any:
    return next(row for row in catalog_rows(table) if row.name.startswith(prefix))


def _matched_bom(bom: SelectedBOM) -> SelectedBOM:
    """G2's BOM with catalogue parts that carry motor constants, ESCs and a cell-built 12S pack."""
    battery = bom.battery.model_copy(update={"cell_name": "Li-ion 21700 energy 5.0Ah", "series": 12, "parallel": 10})
    return bom.model_copy(update={
        "cruise_motor": _part("cruise_motors", "Outrunner 5025"), "cruise_esc": _part("escs", "ESC 120A 12S"),
        "vtol_motor": _part("vtol_motors", "Lift 6215"),
        "vtol_propeller": _part("propellers", "Carbon vtol prop 30x10in"),
        "vtol_esc": _part("escs", "ESC 60A 12S"), "battery": battery,
    })


@pytest.fixture(scope="module")
def matched_g2() -> tuple[Any, FlightModel]:
    req, sizing, bom, aero, cad = load_design("g2")
    inputs = (req, sizing, _matched_bom(bom), aero, cad)
    return inputs, build_flight_model(*inputs)


def _air(model: FlightModel) -> tuple[float, float, float]:
    return model.rho_kgm3, model.mu_pa_s, model.speed_of_sound_ms


# ------------------------------------------------------------------------------------ matched path ---
def test_cruise_thrust_matches_propulsor(matched_g2: Any) -> None:
    (_, _, bom, _, _), model = matched_g2
    expected = propulsor_from_parts(bom.cruise_motor, bom.cruise_propeller, bom.cruise_esc, bom.battery)
    for throttle in SAMPLE_THROTTLES:
        for speed in SAMPLE_SPEEDS_MS:
            assert model.cruise_thrust_n(throttle, speed) == pytest.approx(
                expected.thrust_n(throttle, speed, *_air(model)), rel=1e-12)
    assert model.cruise_thrust_n(2.0, 10.0) == model.cruise_thrust_n(1.0, 10.0)  # throttle clamps to [0, 1]
    assert model.cruise_thrust_n(-1.0, 10.0) == 0.0


def test_pusher_cruise_thrust_carries_the_installation_factor() -> None:
    req, sizing, bom, aero, cad = load_design("g1_vtail")
    model = build_flight_model(req, sizing, bom, aero, cad, pusher_installation_factor=0.9)
    assert model.cruise_propulsor.installation_factor == 0.9
    tractor = build_flight_model(*load_design("g1"), pusher_installation_factor=0.9)
    assert tractor.cruise_propulsor.installation_factor == 1.0  # a nose tractor has no pusher loss


def test_lift_max_thrust_from_propulsor_not_catalogue(matched_g2: Any) -> None:
    (req, sizing, bom, aero, cad), model = matched_g2
    expected = propulsor_from_parts(bom.vtol_motor, bom.vtol_propeller, bom.vtol_esc, bom.battery)
    t_max: float = expected.thrust_n(1.0, 0.0, *_air(model))
    assert [r.max_thrust_n for r in model.lift_rotors] == pytest.approx([t_max] * 4, rel=1e-12)
    inflated = bom.model_copy(update={"vtol_motor": bom.vtol_motor.model_copy(update={"max_thrust_n": 5_000.0})})
    assert build_flight_model(req, sizing, inflated, aero, cad).lift_rotors[0].max_thrust_n == pytest.approx(t_max)


def test_weak_lift_rotors_are_rejected(matched_g2: Any) -> None:
    (req, sizing, bom, aero, cad), _ = matched_g2
    weak = bom.model_copy(update={"vtol_motor": bom.vtol_motor.model_copy(update={"max_current_a": 1.0})})
    with pytest.raises(FlightModelError, match="lift rotors"):
        build_flight_model(req, sizing, weak, aero, cad)


def test_lift_thrust_curve_comes_from_the_lift_propulsor(matched_g2: Any) -> None:
    _, model = matched_g2
    outputs, thrusts = lift_thrust_curve(model)
    assert thrusts[0] == 0.0 and thrusts[-1] == pytest.approx(model.lift_rotors[0].max_thrust_n)
    # monotone up to the rpm solve's tolerance (two throttles a hair apart can swap in the last digits)
    assert all(b >= a * (1.0 - MONOTONE_RTOL) for a, b in zip(thrusts, thrusts[1:]))
    assert thrusts[outputs.index(0.1)] == 0.0  # below Q_M_SPIN_MIN the rotor gives no thrust
    u: float = (0.6 - pg.MOTOR_SPIN_MIN) / (pg.MOTOR_SPIN_MAX - pg.MOTOR_SPIN_MIN)
    assert thrusts[outputs.index(0.6)] == pytest.approx(model.lift_thrust_n(u))


# ------------------------------------------------------------------------------ ArduPilot params ---
def test_fit_thrust_expo_recovers_known_curve() -> None:
    throttles = [i / 20 for i in range(21)]
    assert pg.fit_thrust_expo(throttles, [0.35 * u + 0.65 * u * u for u in throttles]) == pytest.approx(0.65, abs=0.01)
    assert pg.fit_thrust_expo(throttles, [math.sqrt(u) for u in throttles]) == 0.0  # concave: clamped at 0
    assert pg.fit_thrust_expo(throttles, [u**4 for u in throttles]) == 1.0  # steeper than u^2: clamped at 1
    with pytest.raises(ValueError):
        pg.fit_thrust_expo([0.0, 1.0], [0.0, 1.0])  # no interior point: the expo is undetermined
    with pytest.raises(ValueError):
        pg.fit_thrust_expo([0.5], [0.2, 0.3])


def test_param_expo_and_hover_follow_the_rotor_model(matched_g2: Any) -> None:
    (_, sizing, bom, _, _), model = matched_g2
    params = pg.generate_params(model, sizing, bom)
    throttles, fractions = model.lift_thrust_fractions()
    assert params["Q_M_THST_EXPO"] == pytest.approx(pg.fit_thrust_expo(throttles, fractions), abs=1e-3)
    assert 0.0 <= params["Q_M_THST_EXPO"] <= 1.0
    assert params["Q_M_THST_HOVER"] == pytest.approx(model.weight_n / (4 * model.lift_rotors[0].max_thrust_n),
                                                     abs=1e-4)


def test_battery_failsafe_voltages_follow_series_count(matched_g2: Any) -> None:
    (_, sizing, bom, _, _), _ = matched_g2
    params = pg.battery_params(sizing, bom, True)
    assert params["BATT_CRT_VOLT"] == pytest.approx(12 * LI_ION_LOADED_MIN_V_CELL, abs=0.01)
    assert params["BATT_LOW_VOLT"] == pytest.approx(12 * LI_ION_LOADED_MIN_V_CELL * pg.BATT_LOW_VOLT_MARGIN, abs=0.01)
    lipo = bom.model_copy(update={"battery": bom.battery.model_copy(update={"chemistry": "lipo", "series": 6})})
    lipo_params = pg.battery_params(sizing, lipo, True)
    assert lipo_params["BATT_CRT_VOLT"] == pytest.approx(6 * LIPO_LOADED_MIN_V_CELL, abs=0.01)
    assert lipo_params["BATT_LOW_VOLT"] > lipo_params["BATT_CRT_VOLT"]
    assert pg.sitl_overrides()["BATT_MONITOR"] == 0  # SITL's JSON backend reports no battery


def test_legacy_battery_series_comes_from_the_pack_voltage() -> None:
    _, sizing, bom, _, _ = load_design("legacy_g1")  # 21.6 V Li-ion, no cell build: 6S
    assert bom.battery.series is None
    assert pg.battery_params(sizing, bom, False)["BATT_CRT_VOLT"] == pytest.approx(6 * LI_ION_LOADED_MIN_V_CELL,
                                                                                    abs=0.01)


def test_trim_throttle_solves_the_thrust_curve(matched_g2: Any) -> None:
    _, model = matched_g2
    drag: float = model.drag_n(model.v_cruise_ms, model.cruise_lift_coefficient(model.v_cruise_ms))
    throttle = model.cruise_throttle_for(drag, model.v_cruise_ms)
    assert throttle is not None and 0.0 < throttle < 1.0
    assert model.cruise_thrust_n(throttle, model.v_cruise_ms) == pytest.approx(drag, rel=1e-3)
    trim = pg.envelope_params(model)["TRIM_THROTTLE"]
    assert trim == max(pg.TRIM_THROTTLE_BOUNDS[0], min(pg.TRIM_THROTTLE_BOUNDS[1], round(100 * throttle)))


def test_flat_topped_thrust_trims_at_the_lowest_saturating_throttle() -> None:
    model = build_flight_model(*load_design("legacy_g1"))  # legacy 4S motor on a 6S pack: tip Mach caps the rpm
    v: float = model.v_cruise_ms
    t_full: float = model.cruise_thrust_n(1.0, v)
    assert model.cruise_thrust_n(0.9, v) == pytest.approx(t_full)  # the flat top
    throttle = model.cruise_throttle_for(t_full, v)
    assert throttle is not None and throttle < 0.9
    assert model.cruise_throttle_for(1.01 * t_full, v) is None


def test_a_flat_top_with_float_noise_still_trims_where_the_saturation_starts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A current-limited top is flat only up to the rpm solve's noise: full throttle may come out a few ulps above
    the rest of the plateau (seen on Linux libm). The trim must still find where the plateau starts."""
    model = build_flight_model(*load_design("legacy_g1"))
    plateau: float = 14.5

    def noisy_flat_top(self: FlightModel, throttle: float, speed_ms: float) -> float:
        return min(throttle / 0.7, 1.0) * plateau * (1.0 + (1e-15 if throttle == 1.0 else 0.0))
    monkeypatch.setattr(FlightModel, "cruise_thrust_n", noisy_flat_top)
    throttle = model.cruise_throttle_for(model.cruise_thrust_n(1.0, 18.0), 18.0)
    assert throttle is not None and throttle == pytest.approx(0.7, abs=1e-3)


def test_unreachable_cruise_is_a_readable_error(matched_g2: Any) -> None:
    (req, sizing, bom, aero, cad), model = matched_g2
    weak = replace(model.cruise_propulsor, max_current_a=0.5)
    with pytest.raises(pg.ParamGenerationError, match="cannot reach"):
        pg.envelope_params(replace(model, cruise_propulsor=weak))
    slow = bom.model_copy(update={"cruise_motor": bom.cruise_motor.model_copy(update={"max_current_a": 0.5})})
    with pytest.raises(FlightModelError, match="cruise propeller"):
        build_flight_model(req, sizing, slow, aero, cad)


# ----------------------------------------------------------------------------------- legacy path ---
def test_legacy_record_estimates_motor_constants_and_esc_class() -> None:
    _, _, bom, _, _ = load_design("legacy_g2")
    assert bom.cruise_esc is None and bom.cruise_motor.rm_ohm is None and bom.vtol_motor.kv is None
    model = build_flight_model(*load_design("legacy_g2"))
    motor = bom.cruise_motor
    expected = Propulsor(
        table=propeller_table(bom.cruise_propeller.diameter_m, bom.cruise_propeller.pitch_m), kv=motor.kv,
        esc_efficiency=ESC_EFFICIENCY_SMALL, bus_voltage_v=bom.battery.voltage_v,
        **motor_constants(motor.continuous_power_w, motor.max_power_w, motor.voltage_v))
    assert model.cruise_propulsor == expected
    assert model.lift_propulsor is not None and model.lift_propulsor.kv == 180.0  # KV from "Lift 6215-180KV"
    assert model.lift_propulsor.role == "lift"


def test_pack_series_refuses_an_unknown_chemistry() -> None:
    _, _, bom, _, _ = load_design("legacy_g1")  # a pack without a cell build
    unknown = bom.battery.model_construct(**{**dict(bom.battery), "chemistry": "nimh"})
    with pytest.raises(ValueError, match="nimh"):
        pack_series(unknown)


def test_legacy_esc_efficiency_follows_the_catalogue_classes() -> None:
    assert legacy_esc_efficiency(22.2) == ESC_EFFICIENCY_SMALL
    assert legacy_esc_efficiency(100.0) == ESC_EFFICIENCY_HV


def test_matched_parts_use_the_catalogue_esc(matched_g2: Any) -> None:
    (_, _, bom, _, _), model = matched_g2
    assert model.cruise_propulsor == plant_propulsor(bom.cruise_motor, bom.cruise_propeller, bom.cruise_esc,
                                                     bom.battery)
    assert model.cruise_propulsor.esc_efficiency == bom.cruise_esc.efficiency


def test_transition_time_is_solved_once_per_model(matched_g2: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _, nominal = matched_g2
    model = replace(nominal)  # a fresh model with nothing cached
    calls: list[float] = []
    original = FlightModel.cruise_thrust_n

    def counting(self: FlightModel, throttle: float, speed_ms: float) -> float:
        calls.append(speed_ms)
        return original(self, throttle, speed_ms)

    monkeypatch.setattr(FlightModel, "cruise_thrust_n", counting)
    first: float = fm.transition_time_s(model)
    solved: int = len(calls)
    assert solved > 0 and fm.transition_time_s(model) == first and len(calls) == solved
    assert fm.transition_time_s(replace(model, mass_kg=2 * model.mass_kg)) > first  # a changed model re-solves


# ------------------------------------------------------------- the run's assumptions reach SITL ---
LOW_PUSHER_FACTOR: float = 0.85


def test_sitl_flight_model_takes_the_runs_pusher_factor() -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    inputs = load_design("g1_vtail")
    lossy = DEFAULT_ASSUMPTIONS.model_copy(update={"pusher_efficiency_factor": LOW_PUSHER_FACTOR})
    default_model = sv.fcs_flight_model(*inputs, DEFAULT_ASSUMPTIONS)
    lossy_model = sv.fcs_flight_model(*inputs, lossy)
    assert lossy_model.cruise_propulsor.installation_factor == LOW_PUSHER_FACTOR
    v: float = lossy_model.v_cruise_ms
    assert lossy_model.cruise_thrust_n(0.5, v) == pytest.approx(
        default_model.cruise_thrust_n(0.5, v) * LOW_PUSHER_FACTOR / DEFAULT_ASSUMPTIONS.pusher_efficiency_factor)
    assert pg.envelope_params(lossy_model)["TRIM_THROTTLE"] > pg.envelope_params(default_model)["TRIM_THROTTLE"]


def test_pipeline_and_agent_hand_the_assumptions_to_sitl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    from src import pipeline
    from src.agent import nodes
    seen: list[Any] = []
    monkeypatch.setattr(sv, "generate_and_verify_fcs", lambda *args, **kwargs: seen.append(args[6]) or "fcs")
    lossy = DEFAULT_ASSUMPTIONS.model_copy(update={"pusher_efficiency_factor": LOW_PUSHER_FACTOR})
    req, sizing, bom, aero, cad = load_design("g1_vtail")
    assert pipeline.run_sitl(req, sizing, bom, aero, cad, tmp_path, lambda _: None, lossy) == "fcs"
    monkeypatch.setattr(nodes, "_write", lambda *args: None)
    state = {"requirements": req, "sizing_results": sizing, "selected_bom": bom, "aero_stability": aero,
             "cad_artifacts": cad, "out_dir": str(tmp_path), "design_assumptions": lossy}
    ctx = type("Ctx", (), {"progress": staticmethod(lambda _: None)})()
    assert nodes.generate_and_verify_fcs(state, ctx) == {"fcs_artifacts": "fcs"}
    assert seen == [lossy, lossy]


def test_quadplane_thrust_voltage_compensation_spans_the_pack(matched_g2: Any) -> None:
    (_, sizing, bom, _, _), model = matched_g2
    params = pg.generate_params(model, sizing, bom)
    cell = next(c for c in catalog_rows("cells") if c.name == bom.battery.cell_name)
    assert params["Q_M_BAT_VOLT_MAX"] == pytest.approx(12 * cell.max_voltage_v, abs=0.01)
    assert params["Q_M_BAT_VOLT_MIN"] == pytest.approx(12 * LI_ION_LOADED_MIN_V_CELL, abs=0.01)
    legacy_g2 = load_design("g2")  # no cell build: 12S Li-ion from 43.2 V, catalogue max cell voltage
    legacy = pg.generate_params(build_flight_model(*legacy_g2), legacy_g2[1], legacy_g2[2])
    assert legacy["Q_M_BAT_VOLT_MAX"] == pytest.approx(12 * MAX_CELL_V, abs=0.01)
    fixed_wing = load_design("g1")
    assert "Q_M_BAT_VOLT_MAX" not in pg.generate_params(build_flight_model(*fixed_wing), fixed_wing[1], fixed_wing[2])
