"""End-to-end design run used by the web form (and the M5 CLI): sizing -> COTS BOM -> aero/CAD -> SITL -> files.

Every engineering number comes from the deterministic tools; this module only sequences them, writes
the artifacts to `outputs/<run_id>/`, and turns cross-checks into human-readable warnings.
"""

import importlib.util
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final, Optional

from pydantic import BaseModel

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, SizingOutput
from src.tools.aero_cad.cad_packer import SM_MAX, SM_MIN, CADPackingError
from src.tools.aero_cad.stability import AeroAnalysisError
from src.tools.database.matcher import ComponentNotFoundError, select_bom
from src.tools.database.seed_data import DEFAULT_DB_PATH, seed_database
from src.tools.flight_control.flight_model import FlightModelError
from src.tools.flight_control.param_generator import ParamGenerationError
from src.tools.flight_control.runtime import (MAX_OVERSHOOT, MAX_SETTLING_S, MAX_TRANSITION_ALT_LOSS_M,
                                              SitlValidationError, sitl_available)
from src.tools.sizing.mission_solver import SizingDivergenceError, component_needs, size_aircraft
from src.tools.sizing.topology import GroupClassificationError

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT: Final[Path] = Path(__file__).resolve().parents[1] / "outputs"
MAX_SLUG_LENGTH: Final[int] = 30  # leaves room for a de-duplication suffix within 40 chars
MASS_CONVERGENCE_TOLERANCE: Final[float] = 0.03  # SRS 4.2: as-built vs sized MTOW within +/- 3 %
LD_DEVIATION_TOLERANCE: Final[float] = 0.15

# OpenVSP is a process-wide singleton: serialise runs from concurrent web sessions. The catalog is
# seeded at most once per process even if several sessions start their first run together.
_OPENVSP_LOCK: Final = threading.Lock()
_SEED_LOCK: Final = threading.Lock()

Progress = Callable[[str], None]


class PipelineError(RuntimeError):
    """The run cannot proceed as requested (e.g. aero/CAD asked for without OpenVSP)."""


# Errors that mean "this mission/design is infeasible" - shown to the user verbatim. Anything else is a
# bug and propagates with its traceback.
DESIGN_ERRORS: Final[tuple[type[Exception], ...]] = (
    PipelineError, SizingDivergenceError, GroupClassificationError, ComponentNotFoundError,
    CADPackingError, AeroAnalysisError, FlightModelError, ParamGenerationError, SitlValidationError,
)


@dataclass(frozen=True)
class DesignRun:
    run_id: str
    out_dir: Path
    requirements: OperationalRequirements
    sizing: SizingOutput
    bom: SelectedBOM
    aero: Optional[AeroStabilityOutput]
    cad: Optional[CADArtifacts]
    warnings: tuple[str, ...]
    fcs: Optional[FCSArtifacts] = None


def aero_cad_available() -> bool:
    """OpenVSP's compiled Python API exists only in the Docker image."""
    return importlib.util.find_spec("openvsp") is not None


def new_run_id(mission_name: str) -> str:
    """`<UTC timestamp>-<slug>`: safe as a single path component whatever the user typed."""
    stamp: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug: str = re.sub(r"[^a-z0-9]+", "-", mission_name.lower()).strip("-")[:MAX_SLUG_LENGTH].strip("-")
    return f"{stamp}-{slug or 'design'}"


def create_run_dir(outputs_root: Path, run_id: str) -> tuple[str, Path]:
    """Atomically claim a fresh folder (mkdir fails if another session took the name first)."""
    outputs_root.mkdir(parents=True, exist_ok=True)
    candidate, n = run_id, 1
    while True:
        try:
            (outputs_root / candidate).mkdir()
            return candidate, outputs_root / candidate
        except FileExistsError:
            n += 1
            candidate = f"{run_id}-{n}"


def _write_json(path: Path, payload: BaseModel | dict) -> None:
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    path.write_text(json.dumps(data, indent=2))


def design_warnings(sizing: SizingOutput, aero: Optional[AeroStabilityOutput],
                    cad: Optional[CADArtifacts], fcs: Optional[FCSArtifacts] = None) -> tuple[str, ...]:
    """Cross-checks between sizing, VSPAERO, the as-built CAD and SITL that the critic (M5) will act on."""
    notes: list[str] = []
    if fcs is not None and not fcs.passed:
        notes.append(f"SITL doublet test failed after {fcs.attempts} attempt(s): roll overshoot "
                     f"{fcs.roll_overshoot:.0%} / settling {fcs.roll_settling_s:.1f} s, pitch overshoot "
                     f"{fcs.pitch_overshoot:.0%} / settling {fcs.pitch_settling_s:.1f} s, transition altitude loss "
                     f"{fcs.max_transition_altitude_loss_m:.1f} m (limits {MAX_OVERSHOOT:.0%}, {MAX_SETTLING_S:g} s, "
                     f"{MAX_TRANSITION_ALT_LOSS_M:g} m).")
    if aero is not None:
        dev: float = aero.ld_cruise / sizing.assumed_ld_cruise - 1
        if abs(dev) > LD_DEVIATION_TOLERANCE:
            notes.append(f"VSPAERO cruise L/D {aero.ld_cruise:.1f} vs sizing assumption {sizing.assumed_ld_cruise:.1f} "
                         f"({dev:+.0%}); the VLM drag build-up omits lift-rotor and nacelle drag.")
    if cad is not None:
        growth: float = cad.as_built_mass_kg / sizing.mtow_kg - 1
        if abs(growth) > MASS_CONVERGENCE_TOLERANCE:
            notes.append(f"As-built mass {cad.as_built_mass_kg:.2f} kg is {growth:+.0%} vs sized MTOW "
                         f"{sizing.mtow_kg:.2f} kg (outside the ±3% convergence band; catalog parts or the "
                         f"enlarged pod differ from the sizing estimate).")
        if not SM_MIN <= cad.static_margin <= SM_MAX:
            notes.append(f"Final static margin {cad.static_margin:.3f} is outside {SM_MIN}-{SM_MAX}.")
        if cad.collision_count:
            notes.append(f"{cad.collision_count} interference(s) between parts, booms or the skin.")
    return tuple(notes)


def _noop(_: str) -> None:
    return None


def ensure_catalog(db_path: Path) -> None:
    with _SEED_LOCK:
        if not db_path.is_file():
            seed_database(db_path)


def run_aero_cad(sizing: SizingOutput, bom: SelectedBOM, req: OperationalRequirements, out_dir: Path,
              progress: Progress) -> tuple[AeroStabilityOutput, CADArtifacts]:
    from src.tools.aero_cad.build import build_aero_cad  # imports OpenVSP

    if not _OPENVSP_LOCK.acquire(blocking=False):
        progress("Waiting for another design run to release OpenVSP")
        _OPENVSP_LOCK.acquire()
    try:
        progress("Building the OpenVSP OML, running VSPAERO and packing the CAD (this can take a few minutes)")
        return build_aero_cad(sizing, bom, req.service_ceiling_m, out_dir)
    finally:
        _OPENVSP_LOCK.release()


def run_sitl(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput,
          cad: CADArtifacts, out_dir: Path, progress: Progress) -> FCSArtifacts:
    from src.tools.flight_control.sitl_validator import generate_and_verify_fcs  # pymavlink + JSBSim

    progress("Generating ArduPilot parameters and flying the SITL take-off and doublet test")
    return generate_and_verify_fcs(req, sizing, bom, aero, cad, out_dir)


def _summary(run_id: str, stages: dict[str, bool], status: str, **extra: object) -> dict[str, object]:
    return {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **stages, "status": status, **extra}


def _execute(req: OperationalRequirements, run_id: str, out_dir: Path, db_path: Path, stages: dict[str, bool],
             progress: Progress) -> DesignRun:
    _write_json(out_dir / "requirements.json", req)
    progress("Sizing the aircraft (time-stepped mission simulation + MTOW convergence)")
    sizing: SizingOutput = size_aircraft(req)
    _write_json(out_dir / "sizing.json", sizing)
    progress("Selecting COTS components from the catalog")
    ensure_catalog(db_path)
    bom: SelectedBOM = select_bom(db_path, component_needs(sizing, req))
    _write_json(out_dir / "bom.json", bom)
    aero, cad = run_aero_cad(sizing, bom, req, out_dir, progress) if stages["aero_cad"] else (None, None)
    fcs: Optional[FCSArtifacts] = None
    if aero is not None and cad is not None:
        _write_json(out_dir / "aero.json", aero)
        _write_json(out_dir / "cad.json", cad)
        if stages["sitl"]:
            fcs = run_sitl(req, sizing, bom, aero, cad, out_dir, progress)
            _write_json(out_dir / "fcs.json", fcs)
    warnings = design_warnings(sizing, aero, cad, fcs)
    _write_json(out_dir / "run_summary.json", _summary(run_id, stages, "ok", warnings=list(warnings)))
    progress("Done")
    return DesignRun(run_id, out_dir, req, sizing, bom, aero, cad, warnings, fcs)


def resolve_stages(with_aero_cad: Optional[bool], with_sitl: Optional[bool]) -> dict[str, bool]:
    """None means "run it when the toolchain is installed". SITL flies the aero/CAD model, so it needs both."""
    aero_ok: bool = aero_cad_available()
    do_aero: bool = aero_ok if with_aero_cad is None else with_aero_cad
    if do_aero and not aero_ok:
        raise PipelineError("Aero/CAD needs OpenVSP, which is only installed in the Docker image "
                            "(run the form with `docker compose up ui`).")
    do_sitl: bool = (do_aero and sitl_available()) if with_sitl is None else with_sitl
    if do_sitl and not do_aero:
        raise PipelineError("The SITL flight test flies the aero/CAD model: enable aero/CAD too.")
    if do_sitl and not sitl_available():
        raise PipelineError("The SITL flight test needs ArduPilot's arduplane, which is only installed in the "
                            "Docker image (run the form with `docker compose up ui`).")
    return {"aero_cad": do_aero, "sitl": do_sitl}


def run_design(req: OperationalRequirements, outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
               db_path: Path = DEFAULT_DB_PATH, with_aero_cad: Optional[bool] = None,
               progress: Progress = _noop, with_sitl: Optional[bool] = None) -> DesignRun:
    """Run the design pipeline for `req`. `with_aero_cad` / `with_sitl` = None run OpenVSP/CAD and the
    ArduPilot SITL flight test when installed. Raises one of DESIGN_ERRORS when the mission cannot be
    designed; a failed run's folder keeps a `run_summary.json` with status "failed"."""
    stages: dict[str, bool] = resolve_stages(with_aero_cad, with_sitl)
    run_id, out_dir = create_run_dir(outputs_root, new_run_id(req.mission_name))
    logger.info("[Tool] run_design %s -> %s (%s)", run_id, out_dir, stages)
    try:
        return _execute(req, run_id, out_dir, db_path, stages, progress)
    except Exception as exc:
        _write_json(out_dir / "run_summary.json",
                    _summary(run_id, stages, "failed", error=f"{type(exc).__name__}: {exc}"))
        raise
