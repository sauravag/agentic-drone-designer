"""design_report.md (SRS 5.7). Every number and table is generated here from tool outputs; the LLM only
supplies the three prose sections (ReportNarrative), and `fallback_narrative` covers runs without an LLM.

Table builders are shared with the web form (src/ui/presenters.py) so both show identical figures.
"""

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Sequence

from src.agent.agent_schemas import ReportNarrative
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, SizingOutput
from src.ui import presenters

POLAR_ALPHAS_DEG: Final[tuple[float, ...]] = (-4.0, 0.0, 4.0, 8.0, 12.0)  # SRS VSPAERO sweep
NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?")
TRIVIAL_NUMBERS: Final[frozenset[str]] = frozenset({"0", "1", "2", "3", "4", "5"})  # counts, list items, "Group 2"


@dataclass(frozen=True)
class ReportInputs:
    run_id: str
    out_dir: Path
    requirements: OperationalRequirements
    sizing: Optional[SizingOutput]
    bom: Optional[SelectedBOM]
    aero: Optional[AeroStabilityOutput]
    cad: Optional[CADArtifacts]
    fcs: Optional[FCSArtifacts]
    converged: bool
    iterations: int
    final_gates: Sequence[dict[str, Any]] = ()
    critic_log: Sequence[dict[str, Any]] = ()
    assumptions: Sequence[str] = ()
    warnings: Sequence[str] = field(default_factory=tuple)


def cell(value: Any) -> str:
    """One markdown table cell: LLM- or user-supplied text cannot break the table or inject markup."""
    text: str = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|").replace("`", "'")


def defang(text: str) -> str:
    """Neutralise markdown link/image syntax in LLM- or user-authored text: a rendered `![](https://...)`
    would make the viewer's browser fetch an attacker-chosen URL (data exfiltration via prompt injection)."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("<", "&lt;")


def inline(text: str) -> str:
    """A single line of user/LLM text for headings and bullets (no newlines, heading markers or links)."""
    return defang(" ".join(text.split()).lstrip("#").strip())


def table(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "_none_\n"
    headers: list[str] = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def unsupported_numbers(narrative: ReportNarrative, facts_text: str) -> list[str]:
    """Numbers the LLM wrote that do not appear in the facts it was given (Zero-LLM-Math audit)."""
    prose: str = " ".join((narrative.executive_summary, narrative.trade_offs, narrative.risks_and_next_steps))
    known: set[str] = set(NUMBER_PATTERN.findall(facts_text))
    return sorted({n for n in NUMBER_PATTERN.findall(prose) if n not in known and n not in TRIVIAL_NUMBERS})


def pairs(rows: Sequence[tuple[str, str]]) -> str:
    return table([{"Quantity": k, "Value": v} for k, v in rows])


def polar_rows(aero: AeroStabilityOutput, sizing: SizingOutput) -> list[dict[str, Any]]:
    """Linearised polar from the VSPAERO derivatives: CL = CL0 + CLa*a, CD = CD0 + CL^2 / (pi e AR)."""
    k: float = 1.0 / (math.pi * sizing.oswald_e * sizing.aspect_ratio)
    rows: list[dict[str, Any]] = []
    for alpha in POLAR_ALPHAS_DEG:
        cl: float = aero.cl0 + aero.cl_alpha_per_rad * math.radians(alpha)
        cd: float = aero.cd0 + k * cl**2
        rows.append({"alpha (deg)": alpha, "CL": round(cl, 3), "CD": round(cd, 4), "L/D": round(cl / cd, 1)})
    return rows


def requirement_rows(req: OperationalRequirements) -> list[tuple[str, str]]:
    values: dict[str, Any] = presenters.form_defaults(req)
    return [(label, str(values[key])) for key, label in presenters.FIELD_LABELS.items()]


def facts(inputs: ReportInputs) -> str:
    """Compact JSON of the computed results, given to the LLM as the only source of numbers."""
    data: dict[str, Any] = {"requirements": inputs.requirements.model_dump(mode="json"),
                            "converged": inputs.converged, "iterations": inputs.iterations,
                            "gates": list(inputs.final_gates), "assumptions": list(inputs.assumptions),
                            "warnings": list(inputs.warnings)}
    if inputs.sizing:
        data["sizing"] = dict(presenters.sizing_summary(inputs.sizing))
    if inputs.bom:
        data["bom_total_mass_kg"] = round(inputs.bom.total_hardware_mass_kg, 3)
    if inputs.aero:
        data["aero"] = {r["Symbol"]: r["Value"] for r in presenters.aero_rows(inputs.aero)}
    if inputs.cad:
        data["cad"] = dict(presenters.cad_rows(inputs.cad))
    if inputs.fcs:
        data["sitl"] = {r["Check"]: f"{r['Result']} ({'pass' if r['Pass'] == 'yes' else 'fail'})"
                        for r in presenters.fcs_rows(inputs.fcs)}
    return json.dumps(data, indent=1)


def fallback_narrative(inputs: ReportInputs, reason: str) -> ReportNarrative:
    status: str = "meets all critic gates" if inputs.converged else "did not pass every critic gate"
    return ReportNarrative(
        executive_summary=f"ISR-Forge designed '{inputs.requirements.mission_name}'; the design {status} after "
                          f"{inputs.iterations} critic iteration(s). (Narrative generated without an LLM: {reason}.)",
        trade_offs="See the tables below; no LLM narrative was generated for this run.",
        risks_and_next_steps="Review the warnings section and the SITL report before building hardware.",
    )


def _sections(inputs: ReportInputs) -> list[str]:
    s: list[str] = []
    s += ["## Operational requirements\n", pairs(requirement_rows(inputs.requirements))]
    s += ["### Assumptions\n", "\n".join(f"- {inline(a)}" for a in inputs.assumptions) or "_none_", "\n"]
    s += ["## Design compliance (critic gates)\n", table(list(inputs.final_gates))]
    s += ["### Critic iterations\n", table([{"Iteration": c["iteration"], "Failed gates": ", ".join(c["failed"]) or "-",
                                             "Changes": "; ".join(c["changes"]) or "-"} for c in inputs.critic_log])]
    if inputs.sizing:
        s += ["## Configuration and sizing\n", pairs(presenters.sizing_summary(inputs.sizing))]
        s += ["## Mission energy budget\n", table(presenters.phase_rows(inputs.sizing))]
    if inputs.bom:
        s += ["## Bill of materials\n", table(presenters.bom_rows(inputs.bom))]
    if inputs.aero and inputs.sizing:
        s += ["## Aerodynamics and stability (VSPAERO)\n", table(presenters.aero_rows(inputs.aero))]
        s += ["### Linearised cruise polar\n", table(polar_rows(inputs.aero, inputs.sizing))]
    if inputs.cad:
        s += ["## CAD, mass properties and packaging\n", pairs(presenters.cad_rows(inputs.cad))]
        s += [f"Files: `{Path(inputs.cad.oml_step_path).name}`, `{Path(inputs.cad.assembly_step_path).name}`, "
              f"`{Path(inputs.cad.oml_vsp3_path).name}`\n"]
    if inputs.fcs:
        s += ["## Flight control and SITL validation\n", table(presenters.fcs_rows(inputs.fcs))]
        s += [f"Parameters: `{Path(inputs.fcs.param_path).name}`; telemetry: `{Path(inputs.fcs.sitl_report_path).name}`\n"]
    return s


def render(inputs: ReportInputs, narrative: ReportNarrative) -> str:
    status: str = "CONVERGED" if inputs.converged else "NOT CONVERGED"
    stamp: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head: list[str] = [f"# Design report: {inline(inputs.requirements.mission_name)}\n",
                       f"Run `{inputs.run_id}` · {stamp} · status **{status}** after {inputs.iterations} critic "
                       f"iteration(s)\n", "## Executive summary\n",
                       "_Narrative written by the LLM from the computed results; the tables are authoritative._\n",
                       defang(narrative.executive_summary) + "\n"]
    tail: list[str] = ["## Trade-offs\n", defang(narrative.trade_offs) + "\n",
                       "## Risks and next steps\n", defang(narrative.risks_and_next_steps) + "\n",
                       "## Warnings\n", "\n".join(f"- {inline(w)}" for w in inputs.warnings) or "_none_", "\n"]
    return "\n".join(head + _sections(inputs) + tail)


def write_report(inputs: ReportInputs, narrative: ReportNarrative) -> Path:
    stray: list[str] = unsupported_numbers(narrative, facts(inputs))
    if stray:
        inputs = replace(inputs, warnings=(*inputs.warnings, "The LLM narrative mentions figures not in the computed "
                                           f"results (treat as unverified): {', '.join(stray)}"))
    path: Path = inputs.out_dir / "design_report.md"
    path.write_text(render(inputs, narrative))
    return path
