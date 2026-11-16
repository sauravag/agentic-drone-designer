"""ISR-Forge CLI: design an ISR UAV from a natural-language request (or a requirements JSON file).

    python -m src.main "Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal,
                        6-hour endurance, and 80 km datalink radius"
    python -m src.main --spec requirements.json --no-sitl

Natural-language input needs an LLM configured in .env: OPENROUTER_API_KEY (any OpenRouter model) or
ANTHROPIC_API_KEY, with ISR_FORGE_LLM_PROVIDER/ISR_FORGE_MODEL; --spec runs without an LLM (rule-based
critic, template narrative). Aero/CAD and SITL run when their toolchains are installed
(the Docker image). Exit codes: 0 converged (and SITL passed if run), 1 not converged or SITL failed,
2 clarification needed, 3 error.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final, Optional, Sequence

from pydantic import ValidationError

from src import pipeline
from src.agent.graph import run_agent
from src.agent.llm import StructuredLLM
from src.agent.nodes import MAX_ITERATIONS, AgentContext, run_status
from src.agent.providers import LLMConfigError, make_llm
from src.schemas.assumptions import DesignAssumptions
from src.schemas.requirements import OperationalRequirements
from src.tools.database.seed_data import DEFAULT_DB_PATH

EXIT_CONVERGED: Final[int] = 0
EXIT_NOT_CONVERGED: Final[int] = 1
EXIT_CLARIFY: Final[int] = 2
EXIT_ERROR: Final[int] = 3
ITERATION_CEILING: Final[int] = 50


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="isr-forge", description="Autonomous ISR UAV design agent")
    p.add_argument("prompt", nargs="?", default="", help="natural-language design request")
    p.add_argument("--spec", type=Path, help="OperationalRequirements JSON (skips the LLM parser)")
    p.add_argument("--outputs", type=Path, default=pipeline.DEFAULT_OUTPUTS_ROOT, help="run folders root")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="COTS catalog path")
    p.add_argument("--no-aero", action="store_true", help="skip OpenVSP/CAD (and therefore SITL)")
    p.add_argument("--no-sitl", action="store_true", help="skip the ArduPilot SITL flight test")
    p.add_argument("--max-iterations", type=_iterations, default=MAX_ITERATIONS,
                   help=f"critic loop cap, 0-{ITERATION_CEILING} (SRS: {MAX_ITERATIONS})")
    p.add_argument("--quiet", action="store_true")
    return p


def _iterations(text: str) -> int:
    value: int = int(text)
    if not 0 <= value <= ITERATION_CEILING:
        raise argparse.ArgumentTypeError(f"must be between 0 and {ITERATION_CEILING}")
    return value


def _load_spec(path: Path) -> tuple[OperationalRequirements, DesignAssumptions]:
    """Requirements plus the optional top-level "assumptions" object (unknown fields are rejected)."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the spec must be a JSON object")
    assumptions = DesignAssumptions.model_validate(data.pop("assumptions", {}))
    return OperationalRequirements.model_validate(data), assumptions


def _context(args: argparse.Namespace) -> AgentContext:
    stages = pipeline.resolve_stages(False if args.no_aero else None, False if args.no_sitl or args.no_aero else None)
    llm: Optional[StructuredLLM] = None if args.spec else make_llm()
    return AgentContext(llm=llm, outputs_root=args.outputs, db_path=args.db, with_aero_cad=stages["aero_cad"],
                        with_sitl=stages["sitl"], max_iterations=args.max_iterations,
                        progress=lambda message: None if args.quiet else print(f"[Tool] {message}", flush=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    if bool(args.prompt) == bool(args.spec):
        print("Give either a design request or --spec requirements.json (not both)", file=sys.stderr)
        return EXIT_ERROR
    try:
        requirements, assumptions = _load_spec(args.spec) if args.spec else (None, None)
        state = run_agent(_context(args), args.prompt, requirements, design_assumptions=assumptions)
    except (OSError, ValueError, ValidationError, LLMConfigError, pipeline.PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if state.get("fatal_error"):
        print(f"error: {state['fatal_error']}", file=sys.stderr)
        return EXIT_ERROR
    if state.get("clarification_needed"):
        print(f"Clarification needed: {state['clarification_needed']}")
        return EXIT_CLARIFY
    status: str = run_status(state)
    print(f"Run {state['run_id']}: {status.replace('_', ' ')} after {state['iteration_count']} critic "
          f"iteration(s). Report: {state['final_report_path']}")
    return EXIT_CONVERGED if status == "converged" else EXIT_NOT_CONVERGED


if __name__ == "__main__":
    sys.exit(main())
