"""Requirements box tests.

The example picker sits above the box. The box itself is a live fragment, not a form, and holds:
- the mission name;
- Payload, Mission, Envelope and Avionics columns;
- configuration;
- the Design aircraft button.
"""

from pathlib import Path
from typing import Any

import pytest

from src.ui.examples import EXAMPLE_MISSIONS

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
ZT30 = "SIYI ZT30"
G2 = list(EXAMPLE_MISSIONS)[1]


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    from src.ui import agent_service

    monkeypatch.setattr(agent_service, "llm_problem", lambda: "no LLM in this test")
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception
    return at


def _pick(at: Any, model: str) -> None:
    at.selectbox(key="payload_model").set_value(model)
    at.run()
    assert not at.exception


def test_requirements_box_is_not_a_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    assert 'st.form("requirements")' not in APP_PATH.read_text()
    headings = [s.value for s in at.subheader]
    assert headings.index("Start from an example") < headings.index("…or enter the requirements")
    assert {"Payload", "Mission", "Envelope", "Avionics"} <= set(headings)


def test_model_change_fills_mass_and_power(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    _pick(at, ZT30)
    assert at.number_input(key="payload_mass_kg").value == pytest.approx(0.854)
    assert at.number_input(key="payload_power_w").value == pytest.approx(9.0)
    assert any("datasheet values" in c.value for c in at.caption)


def test_edited_mass_persists_across_reruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    _pick(at, ZT30)
    at.number_input(key="payload_mass_kg").set_value(1.0)
    at.run()
    at.number_input(key="cruise_speed_ms").set_value(25.0)
    at.run()
    assert at.number_input(key="payload_mass_kg").value == pytest.approx(1.0)
    assert at.selectbox(key="payload_model").value == ZT30


def test_switch_to_custom_keeps_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    _pick(at, ZT30)
    _pick(at, "custom")
    assert at.number_input(key="payload_mass_kg").value == pytest.approx(0.854)
    assert at.number_input(key="payload_power_w").value == pytest.approx(9.0)


def test_preset_resets_payload_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    _pick(at, ZT30)
    at.selectbox(key="preset").set_value(G2)
    at.run()
    assert at.selectbox(key="payload_model").value == "custom"
    assert at.number_input(key="payload_mass_kg").value == pytest.approx(EXAMPLE_MISSIONS[G2].payload_mass_kg)


def test_ark_greys_jetson_in_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    at.selectbox(key="flight_controller").set_value("ark_jetson_pab")
    at.run()
    assert at.checkbox(key="companion_computer_ark").disabled and at.checkbox(key="companion_computer_ark").value
    assert any("A description sets its own avionics." in c.value for c in at.caption)


def test_design_aircraft_runs_named_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    if not at.checkbox(key="with_aero_cad").disabled:
        at.checkbox(key="with_aero_cad").uncheck()
    _pick(at, ZT30)
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert any("Payload: SIYI ZT30 (0.854 kg, 9 W)" in c.value for c in at.caption)
    parts = next(df.value for df in at.dataframe if "Item" in df.value.columns)
    assert ZT30 in parts["Item"].tolist()


VTAIL_PRESET = "G1 V-tail pusher fixed-wing (reference)"
VTAIL = "single_boom_vtail_pusher"


def test_layout_select_offers_the_built_layouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.schemas.requirements import ENABLED_LAYOUTS

    at = _app(tmp_path, monkeypatch)
    layout = at.selectbox(key="airframe_layout")
    assert layout.label == "Airframe layout" and layout.value == "pod_boom_tractor"
    assert set(layout.options) == {"Pod-and-boom tractor", "Single-boom V-tail pusher"}
    assert {name for name, _ in ENABLED_LAYOUTS} == {"pod_boom_tractor", VTAIL}


def test_vtail_preset_sets_the_layout_and_the_form_runs_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    if not at.checkbox(key="with_aero_cad").disabled:
        at.checkbox(key="with_aero_cad").uncheck()
    at.selectbox(key="preset").set_value(VTAIL_PRESET)
    at.run()
    assert at.selectbox(key="airframe_layout").value == VTAIL
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    tables = [t.value for t in (*at.table, *at.dataframe)]
    assert any("Single-boom V-tail pusher" in table.to_string() for table in tables)
