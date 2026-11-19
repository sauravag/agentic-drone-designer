"""Export a design: every file of its run folder plus `design.json`, in one zip.

`design.json` is a versioned `DesignRecord` built at download time from the in-memory `AgentRunView`, so any
design on screen can be exported, including runs made before this existed. Paths in it are relative to the run
folder (no absolute paths from this machine), and it exists only inside the zip, never in the run folder.
Opening a zip in the app is parked (docs/superpowers/specs/2026-11-19-share-designs-design.md); the record is
what that opener will validate. Kept free of Streamlit so it is unit-testable.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict

from src.schemas.assumptions import DesignAssumptions
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, RunLineage, SizingOutput
from src.ui.agent_service import AgentRunView

DESIGN_FORMAT: Final[str] = "isr-forge-design"
FORMAT_VERSION: Final[int] = 1
RECORD_FILE: Final[str] = "design.json"
_CAD_PATHS: Final[tuple[str, ...]] = ("oml_vsp3_path", "oml_step_path", "assembly_step_path", "viewer_model_path")
_FCS_PATHS: Final[tuple[str, ...]] = ("param_path", "sitl_report_path")


class DesignRecord(BaseModel):
    """Everything the results view shows for one design; `out_dir` is omitted and paths are run-folder relative."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["isr-forge-design"] = "isr-forge-design"
    format_version: Literal[1] = 1
    created_utc: datetime
    status: str
    message: Optional[str] = None
    run_id: str
    requirements: Optional[OperationalRequirements] = None
    sizing: Optional[SizingOutput] = None
    bom: Optional[SelectedBOM] = None
    aero: Optional[AeroStabilityOutput] = None
    cad: Optional[CADArtifacts] = None
    fcs: Optional[FCSArtifacts] = None
    iterations: int = 0
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    gates: tuple[dict[str, Any], ...] = ()
    critic_log: tuple[dict[str, Any], ...] = ()
    report_path: Optional[str] = None
    used_llm: bool = False
    design_assumptions: DesignAssumptions
    explorer_mode: str = "none"
    options: tuple[dict[str, Any], ...] = ()
    options_explanation: Optional[str] = None
    explorer_note: Optional[str] = None
    lineage: Optional[RunLineage] = None


def _relative(path: str | Path, root: Path) -> str:
    """`path` as a POSIX path relative to the run folder; ValueError if it lies outside it."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{Path(path).name} is outside the run folder") from None


def _relative_paths(model: Optional[BaseModel], fields: tuple[str, ...], root: Path) -> Optional[BaseModel]:
    """A copy of an artifact model with its (set) path fields made run-folder relative."""
    if model is None:
        return None
    update = {name: _relative(value, root) for name in fields if (value := getattr(model, name)) is not None}
    return model.model_copy(update=update)


def design_record(view: AgentRunView, now: Optional[datetime] = None) -> DesignRecord:
    """The design.json record for `view`. Raises ValueError for a view without a run folder, or one whose report
    or artifact paths lie outside it."""
    if view.run_id is None or view.out_dir is None:
        raise ValueError("this design has no run folder to export")
    root: Path = Path(view.out_dir).resolve()
    return DesignRecord(
        created_utc=now or datetime.now(timezone.utc), status=view.status, message=view.message,
        run_id=view.run_id, requirements=view.requirements, sizing=view.sizing, bom=view.bom, aero=view.aero,
        cad=_relative_paths(view.cad, _CAD_PATHS, root), fcs=_relative_paths(view.fcs, _FCS_PATHS, root),
        iterations=view.iterations, warnings=view.warnings, assumptions=view.assumptions, gates=view.gates,
        critic_log=view.critic_log, report_path=None if view.report_path is None else _relative(view.report_path, root),
        used_llm=view.used_llm, design_assumptions=view.design_assumptions, explorer_mode=view.explorer_mode,
        options=view.options, options_explanation=view.options_explanation, explorer_note=view.explorer_note,
        lineage=view.lineage,
    )
