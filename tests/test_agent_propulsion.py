"""Agent and report on the matched propulsion (split from test_agent.py): the report's propulsion tables and
estimates, the BOM lines, the critic's propulsion-stage refusal, and the assumptions threaded to every stage."""

from pathlib import Path
from typing import Any

import pytest

from src import pipeline
from src.agent import critic, nodes, report
from src.agent.agent_schemas import Adjustment
from src.schemas.assumptions import DesignAssumptions
from tests.agent_fakes import agent_context, seeded_catalog
from tests.reference_designs import G1_ELECTRIC_FW
from tests.test_flight_control import load_design

TAILED_KNOB = critic.TAILED_SM_KNOB

@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return seeded_catalog(tmp_path_factory.mktemp("db"))


_ctx = agent_context


# ------------------------------------------------------- propulsion match: report, BOM, critic ---
def _matched_inputs(tmp_path: Path) -> report.ReportInputs:
    from tests.matched_design import matched_g1

    sizing, bom = matched_g1()
    return report.ReportInputs(run_id="r1", out_dir=tmp_path, requirements=G1_ELECTRIC_FW, sizing=sizing, bom=bom,
                               aero=None, cad=None, fcs=None, converged=True, iterations=1)


def _matched_report(tmp_path: Path) -> str:
    inputs = _matched_inputs(tmp_path)
    return report.render(inputs, report.fallback_narrative(inputs, "no LLM"))


def test_report_has_propulsion_match_table(tmp_path: Path) -> None:
    from tests.matched_design import matched_g1

    text = _matched_report(tmp_path)
    section = text.split("## Propulsion match", 1)[1].split("\n## ", 1)[0]
    header = next(line for line in section.splitlines() if line.startswith("| Flight point"))
    for column in ("Propulsor", "RPM", "Current (A)", "Throttle", "Propeller efficiency", "Motor efficiency",
                   "Tip Mach"):
        assert column in header
    match = matched_g1()[0].propulsion_match
    assert match is not None
    for op in match.operating_points:
        assert op.flight_point in section
    assert f"{match.operating_points[0].rpm:.0f}" in section


def test_report_labels_hover_efficiency_as_figure_of_merit() -> None:
    from tests.matched_design import matched_g1
    from src.ui import presenters

    cruise_op = matched_g1()[0].propulsion_match.operating_points[0]  # type: ignore[union-attr]
    assert not str(presenters.operating_point_row(cruise_op)["Propeller efficiency"]).startswith("FM ")
    hover_op = cruise_op.model_copy(update={"propulsor": "lift", "flight_point": "Hover"})
    assert str(presenters.operating_point_row(hover_op)["Propeller efficiency"]).startswith("FM ")


def test_report_lists_propulsion_estimates_as_assumptions(tmp_path: Path) -> None:
    text = _matched_report(tmp_path)
    section = text.split("### Propulsion estimates", 1)[1].split("\n## ", 1)[0].lower()
    for phrase in ("generic blade shape", "motor constants", "70 %", "esc efficienc", "cell data",
                   "figure of merit", "windmilling", "tip mach"):
        assert phrase in section, phrase


def test_report_without_a_match_has_no_propulsion_section(tmp_path: Path) -> None:
    from tests.test_flight_control import load_design

    req, sizing, bom, *_ = load_design("legacy_g1")  # written before propulsion matching: no match record
    inputs = report.ReportInputs(run_id="r1", out_dir=tmp_path, requirements=req, sizing=sizing, bom=bom, aero=None,
                                 cad=None, fcs=None, converged=True, iterations=1)
    assert "## Propulsion match" not in report.render(inputs, report.fallback_narrative(inputs, "no LLM"))


def test_bom_lines_show_escs_and_pack_configuration() -> None:
    from tests.matched_design import matched_g1
    from src.ui import presenters

    _, bom = matched_g1()
    rows = {r["Part"]: r for r in presenters.bom_rows(bom)}
    assert rows["Cruise ESC"]["Item"] == bom.cruise_esc.name  # type: ignore[union-attr]
    battery = bom.battery
    assert rows["Battery"]["Item"] == f"{battery.cell_name} {battery.series}S{battery.parallel}P"


def test_critic_names_propulsion_constraint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Path) -> None:
    from src.tools.propulsion.match_types import PropulsionMatchError

    message = "no propulsion set meets every matching rule; limiting constraints - tip Mach at Climb"

    def refuse(*args: object, **kwargs: object) -> None:
        raise PropulsionMatchError(message)

    monkeypatch.setattr(nodes, "size_aircraft", refuse)
    state = {"requirements": G1_ELECTRIC_FW, "out_dir": str(tmp_path)}
    update = nodes.run_sizing(state, _ctx(tmp_path, db, None))  # type: ignore[arg-type]
    gates = nodes._gates(update)  # type: ignore[arg-type]
    (gate,) = gates
    assert "tip Mach at Climb" in gate.detail and "propulsion match" in gate.detail
    llm_pick = [Adjustment(knob="aspect_ratio", direction="increase", reason="LLM guess")]
    overrides, changes = critic.apply_adjustments({}, "electric", llm_pick, gates, None, update["stage_error"]["stage"],
                                                  band=None, sm_knob=TAILED_KNOB)
    assert changes == [] and overrides == {}
    assert critic.unfixable(gates, None, update["stage_error"]["stage"], band=None, sm_knob=TAILED_KNOB) == ["pipeline"]


class _SitlReached(Exception):
    """Stops a run at the SITL stage once the spy has seen its arguments."""


def test_run_design_threads_custom_assumptions(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = DesignAssumptions(liion_max_c_rate=6.0)
    seen: dict[str, DesignAssumptions] = {}
    real_size, real_bom = pipeline.size_aircraft, pipeline.select_bom
    _, _, _, saved_aero, saved_cad = load_design("g1")

    def size_spy(req: Any, overrides: Any = None, assumptions: DesignAssumptions = DesignAssumptions()) -> Any:
        seen["sizing"] = assumptions
        return real_size(req, overrides, assumptions)

    def bom_spy(db_path: Path, needs: Any, assumptions: DesignAssumptions = DesignAssumptions()) -> Any:
        seen["select_bom"] = assumptions
        return real_bom(db_path, needs, assumptions)

    def sitl_spy(*args: Any) -> Any:
        seen["run_sitl"] = args[-1]
        raise _SitlReached

    monkeypatch.setattr(pipeline, "size_aircraft", size_spy)
    monkeypatch.setattr(pipeline, "select_bom", bom_spy)
    monkeypatch.setattr(pipeline, "resolve_stages", lambda *_: {"aero_cad": True, "sitl": True})
    monkeypatch.setattr(pipeline, "run_aero_cad", lambda *_: (saved_aero, saved_cad))  # no OpenVSP on the host
    monkeypatch.setattr(pipeline, "run_sitl", sitl_spy)
    with pytest.raises(_SitlReached):
        pipeline.run_design(G1_ELECTRIC_FW, tmp_path / "o", db, assumptions=custom)
    assert seen == {"sizing": custom, "select_bom": custom, "run_sitl": custom}


def test_auto_propulsion_matching_failure_reaches_the_propulsion_stage(monkeypatch: pytest.MonkeyPatch,
                                                                       tmp_path: Path, db: Path) -> None:
    """Real size_aircraft under propulsion "auto": when neither powertrain finds a propulsion set, the stage is the
    propulsion matching (not "sizing"), so the critic refuses knob changes and names the constraint."""
    from src.tools.propulsion import matching

    real_rows = matching.catalog_rows
    monkeypatch.setattr(matching, "catalog_rows", lambda table: [
        row for row in real_rows(table) if table != "propellers" or row.application == "vtol"])
    update = nodes.run_sizing({"requirements": G1_ELECTRIC_FW, "out_dir": str(tmp_path)},  # type: ignore[arg-type]
                              _ctx(tmp_path, db, None))
    assert update["stage_error"]["stage"] == critic.PROPULSION_STAGE
    assert "Neither powertrain closes" in update["stage_error"]["message"]
