"""OpenVSP OML generation, watertight STEP export, VSPAERO run matrix and ParasiteDrag (SRS 4.3 step 1).

Run matrix (VLM on the lifting surfaces): alpha in {-4, 0, 4, 8, 12} deg; beta = 5 deg; aileron,
elevator and rudder at 10 deg (a V-tail: each ruddervator panel alone at 10 deg instead of elevator and rudder).
Fuselage, booms and rotor discs are excluded from VLM (see stability.py for the handbook fuselage corrections);
booms and fuselage are in the OML/drag set.

Empirical conventions verified against OpenVSP 3.56 / VSPAERO 7.2.2 (and guarded at runtime by
stability._check_control_conventions):
  * equal gains on a mirrored control surface deflect it ANTISYMMETRICALLY, so the elevator's mirror
    gain is -1 and the ailerons keep +1;
  * a mirrored surface's gains are "<surf>_0_Gain" for the main (+y, right) panel and "<surf>_1_Gain" for the planar
    mirror (left), so gains (1, 0) deflect the right panel alone and (0, 1) the left one (a VSPAERO probe of the
    G1 V-tail: (1, 0) yaws nose-right and rolls left for trailing edge down, as only the right panel can);
  * plain EXPORT_STEP assumes feet and writes untrimmed patches, so the OML goes through the
    SurfaceIntersection analysis with CADLenUnit = metres, which yields one trimmed watertight solid.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import aerosandbox as asb
import openvsp as vsp

from src.schemas.state import AeroStabilityOutput, SizingOutput
from src.tools.aero_cad.airfoil_geometry import load_airfoil, write_selig_dat
from src.tools.aero_cad.layout import (
    AILERON_CHORD_FRACTION,
    AILERON_ETA,
    FUSELAGE_PROFILE,
    TAIL_CONTROL_CHORD_FRACTION,
    TAIL_CONTROL_ETA,
    AircraftLayout,
    Boom,
    SurfacePlanform,
)
from src.tools.aero_cad.stability import (
    AeroAnalysisError,
    AeroCoefficients,
    ReferenceGeometry,
    VspaeroCases,
    derive_stability,
)
from src.tools.sizing.topology import surface_laminar_fraction

logger = logging.getLogger(__name__)

ALPHA_SWEEP_DEG: Final[tuple[float, ...]] = (-4.0, 0.0, 4.0, 8.0, 12.0)
BETA_DEG: Final[float] = 5.0
DEFLECTION_DEG: Final[float] = 10.0
WAKE_ITERATIONS: Final[int] = 3
MIRROR_GAIN_SYMMETRIC: Final[float] = -1.0
ELEVATOR_GAINS: Final[tuple[float, float]] = (1.0, MIRROR_GAIN_SYMMETRIC)  # (main, mirror): symmetric deflection
RIGHT_PANEL_GAINS: Final[tuple[float, float]] = (1.0, 0.0)  # the main (+y) panel alone
LEFT_PANEL_GAINS: Final[tuple[float, float]] = (0.0, 1.0)  # the mirrored (-y) panel alone
DEFAULT_GAIN: Final[float] = 1.0  # OpenVSP's auto-grouped gain, restored after a per-panel case

ROTOR_AXIS_ROTATION_DEG: Final[float] = -90.0  # PROP disc normal from +x to +z (lift rotor)
MAX_PLAUSIBLE_CD0: Final[float] = 0.2
PERCENT: Final[float] = 100.0  # OpenVSP's PercLam is a percentage

LIFT_SET: Final[int] = vsp.SET_FIRST_USER
OML_SET: Final[int] = vsp.SET_FIRST_USER + 1
SWEEP: Final[str] = "VSPAEROSweep"
COMPUTE_GEOMETRY: Final[str] = "VSPAEROComputeGeometry"
HISTORY: Final[str] = "VSPAERO_History"


# Component names OpenVSP shows for the tail surfaces (kept stable for stored models).
VSP_TAIL_NAMES: Final[dict[str, str]] = {"htail": "HTail", "vtail": "VTail", "vtail_pair": "VTailPair"}
# Tail control sets the run matrix knows how to deflect.
SUPPORTED_TAIL_CONTROLS: Final[tuple[frozenset[str], ...]] = (frozenset({"elevator", "rudder"}),
                                                               frozenset({"ruddervator"}))


@dataclass(frozen=True)
class VspModel:
    wing: str
    tail_controls: tuple[tuple[str, str], ...]  # (control kind, OpenVSP component name) of each tail surface

    def tail_named(self, control: str) -> str:
        """The OpenVSP component carrying `control`."""
        return next(name for kind, name in self.tail_controls if kind == control)


@dataclass(frozen=True)
class FlightCondition:
    altitude_m: float
    vinf_ms: float
    rho: float
    mach: float
    re_cref: float
    x_ref_m: float
    z_ref_m: float


# ------------------------------------------------------------------------------------ geometry ---
def _set_subsurf(ss_id: str, name: str, value: float) -> None:
    pid: Optional[str] = next((p for p in vsp.GetSubSurfParmIDs(ss_id) if vsp.GetParmName(p) == name), None)
    if pid is None:
        raise AeroAnalysisError(f"OpenVSP sub-surface has no parameter {name!r}")
    vsp.SetParmVal(pid, value)


def _add_control(geom: str, eta: tuple[float, float], chord_fraction: float) -> None:
    ss: str = vsp.AddSubSurf(geom, vsp.SS_CONTROL)
    for name, value in (("EtaFlag", 1.0), ("EtaStart", eta[0]), ("EtaEnd", eta[1]),
                        ("Length_C_Start", chord_fraction), ("Length_C_End", chord_fraction)):
        _set_subsurf(ss, name, value)


def _add_surface(name: str, p: SurfacePlanform, airfoil_dat: Optional[Path], vertical: bool = False) -> str:
    gid: str = vsp.AddGeom("WING")
    vsp.SetGeomName(gid, name)
    vsp.SetDriverGroup(gid, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    for parm, value in (("Span", p.semi_span_m), ("Root_Chord", p.root_chord_m), ("Tip_Chord", p.tip_chord_m),
                        ("Sweep", 0.0), ("Sweep_Location", 0.0), ("Dihedral", p.dihedral_deg)):
        vsp.SetParmVal(gid, parm, "XSec_1", value)
    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", p.x_le_root_m)
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", p.z_root_m)
    if vertical:
        vsp.SetParmVal(gid, "X_Rel_Rotation", "XForm", 90.0)
        vsp.SetParmVal(gid, "Sym_Planar_Flag", "Sym", 0.0)
    if airfoil_dat is not None:
        surf = vsp.GetXSecSurf(gid, 0)
        for i in range(vsp.GetNumXSec(surf)):
            vsp.ChangeXSecShape(surf, i, vsp.XS_FILE_AIRFOIL)
            vsp.ReadFileAirfoil(vsp.GetXSec(surf, i), str(airfoil_dat))
    vsp.Update()
    return gid


def _add_fuselage(layout: AircraftLayout) -> str:
    fid: str = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fid, "Fuselage")
    vsp.SetParmVal(fid, "Length", "Design", layout.fuselage_length_m)
    surf = vsp.GetXSecSurf(fid, 0)
    while vsp.GetNumXSec(surf) - 2 < len(FUSELAGE_PROFILE):  # keep the nose and tail points closed
        vsp.InsertXSec(fid, 1, vsp.XS_CIRCLE)
    if vsp.GetNumXSec(surf) - 2 != len(FUSELAGE_PROFILE):
        raise AeroAnalysisError(f"fuselage has {vsp.GetNumXSec(surf) - 2} interior sections, profile needs {len(FUSELAGE_PROFILE)}")
    for i, (x_frac, d_frac) in enumerate(FUSELAGE_PROFILE, start=1):
        vsp.ChangeXSecShape(surf, i, vsp.XS_CIRCLE)
        xsec = vsp.GetXSec(surf, i)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "Circle_Diameter"), d_frac * layout.fuselage_diameter_m)
        vsp.SetParmVal(vsp.GetXSecParm(xsec, "XLocPercent"), x_frac)
    vsp.Update()
    return fid


def _add_boom(name: str, boom: Boom) -> str:
    pid: str = vsp.AddGeom("POD")
    vsp.SetGeomName(pid, name)
    vsp.SetParmVal(pid, "Length", "Design", boom.length_m)
    vsp.SetParmVal(pid, "FineRatio", "Design", boom.length_m / boom.diameter_m)
    for axis, value in (("X", boom.x_start_m), ("Y", boom.y_m), ("Z", boom.z_m)):
        vsp.SetParmVal(pid, f"{axis}_Rel_Location", "XForm", value)
    return pid


def _add_rotor(index: int, hub: tuple[float, float, float], diameter: float) -> str:
    rid: str = vsp.AddGeom("PROP")
    vsp.SetGeomName(rid, f"LiftRotor{index}")
    vsp.SetParmVal(rid, "PropMode", "Design", float(vsp.PROP_DISK))
    vsp.SetParmVal(rid, "Diameter", "Design", diameter)
    for axis, value in zip("XYZ", hub):
        vsp.SetParmVal(rid, f"{axis}_Rel_Location", "XForm", value)
    vsp.SetParmVal(rid, "Y_Rel_Rotation", "XForm", ROTOR_AXIS_ROTATION_DEG)
    return rid


def _set_laminar_fraction(surfaces: tuple[str, ...], laminar_fraction: Optional[float]) -> None:
    """ParasiteDrag's laminar share on the lifting surfaces (bodies and booms stay fully turbulent)."""
    if laminar_fraction is None:
        return
    for geom in surfaces:
        vsp.SetParmVal(vsp.GetParm(geom, "PercLam", "ParasiteDragProps"), PERCENT * laminar_fraction)


def build_model(layout: AircraftLayout, out_dir: Path, laminar_fraction: Optional[float] = None) -> VspModel:
    """Fresh OpenVSP model of `layout`, with lifting surfaces in LIFT_SET and the OML in OML_SET. The lifting
    surfaces carry `laminar_fraction` for ParasiteDrag (`topology.surface_laminar_fraction`; None keeps OpenVSP's
    fully turbulent default), so the exported .vsp3 reproduces the analysis."""
    vsp.VSPRenew()
    dat: Path = write_selig_dat(load_airfoil(layout.airfoil), out_dir / "airfoils" / f"{layout.airfoil}.dat")
    fuselage = _add_fuselage(layout)
    wing = _add_surface("Wing", layout.wing, dat)
    controls: frozenset[str] = frozenset(t.control for t in layout.tail_surfaces)
    if controls not in SUPPORTED_TAIL_CONTROLS or len(layout.tail_surfaces) != len(controls):
        raise ValueError(f"no VSPAERO run matrix for tail controls {sorted(t.control for t in layout.tail_surfaces)}; "
                         f"supported: {[sorted(c) for c in SUPPORTED_TAIL_CONTROLS]}")
    tail: dict[str, str] = {
        t.name: _add_surface(VSP_TAIL_NAMES.get(t.name, t.name), t.planform, None, vertical=t.is_vertical)
        for t in layout.tail_surfaces
    }
    _add_control(wing, AILERON_ETA, AILERON_CHORD_FRACTION)
    for gid in tail.values():
        _add_control(gid, TAIL_CONTROL_ETA, TAIL_CONTROL_CHORD_FRACTION)
    booms = [_add_boom("TailBoom" if i == 0 else f"TailBoom{i}", b) for i, b in enumerate(layout.tail_booms)]
    booms += [_add_boom(f"VtolBoom{i}", b) for i, b in enumerate(layout.vtol_booms)]
    for i, hub in enumerate(layout.rotor_positions):
        _add_rotor(i, hub, layout.rotor_diameter_m)  # shown in the .vsp3; not part of the OML or VLM
    for gid in (wing, *tail.values()):
        vsp.SetSetFlag(gid, LIFT_SET, True)
    for gid in (fuselage, wing, *tail.values(), *booms):
        vsp.SetSetFlag(gid, OML_SET, True)
    _set_laminar_fraction((wing, *tail.values()), laminar_fraction)
    vsp.Update()
    vsp.AutoGroupVSPAEROControlSurfaces()
    return VspModel(wing=wing, tail_controls=tuple((t.control, VSP_TAIL_NAMES.get(t.name, t.name))
                                                   for t in layout.tail_surfaces))


def export_oml(out_dir: Path) -> tuple[Path, Path]:
    """Write `oml.vsp3` and the trimmed, watertight `oml.step` (metres)."""
    vsp3: Path = out_dir / "oml.vsp3"
    step: Path = out_dir / "oml.step"
    step.unlink(missing_ok=True)
    vsp.WriteVSPFile(str(vsp3))
    si = "SurfaceIntersection"
    vsp.SetAnalysisInputDefaults(si)
    vsp.SetIntAnalysisInput(si, "SelectedSetIndex", [OML_SET])
    vsp.SetIntAnalysisInput(si, "STEPFileFlag", [1])
    vsp.SetIntAnalysisInput(si, "IGESFileFlag", [0])
    vsp.SetIntAnalysisInput(si, "CADLenUnit", [vsp.LEN_M])
    vsp.SetIntAnalysisInput(si, "STEPRepresentation", [vsp.STEP_BREP])
    vsp.SetStringAnalysisInput(si, "STEPFileName", [str(step)])
    vsp.ExecAnalysis(si)
    if not step.is_file() or step.stat().st_size == 0:
        raise AeroAnalysisError(f"OpenVSP SurfaceIntersection did not write {step}")
    return vsp3, step


# ------------------------------------------------------------------------------------ analysis ---
def flight_condition(layout: AircraftLayout, sizing: SizingOutput, altitude_m: float,
                     temperature_offset_c: float = 0.0) -> FlightCondition:
    """Loiter (max-L/D) speed at the service ceiling on the design day (ISA + temperature_offset_c): sets the VSPAERO
    Reynolds and Mach numbers. The L/D the critic compares is taken from the polar at the sizing CLs
    (`stability.ld_deviations`), not at this speed."""
    atm = asb.Atmosphere(altitude=altitude_m, temperature_deviation=temperature_offset_c)
    rho, mu, a = float(atm.density()), float(atm.dynamic_viscosity()), float(atm.speed_of_sound())
    v: float = sizing.v_loiter_ms
    return FlightCondition(altitude_m=altitude_m, vinf_ms=v, rho=rho, mach=v / a,
                           re_cref=rho * v * layout.wing.mac_m / mu,
                           x_ref_m=layout.wing.x_ac_m, z_ref_m=layout.wing.z_root_m)


def _control_group(geom_name: str) -> tuple[str, list[str]]:
    """(deflection parm, gain parms) of the auto-generated VSPAERO control group for a geom; the gains in panel order
    (main +y panel first, then its mirror), by their "_<index>_Gain" names."""
    for gid in vsp.GetControlSurfaceGroupIDVec():
        if vsp.GetVSPAEROControlGroupNameByID(gid).startswith(f"{geom_name}_"):
            pids = list(vsp.FindContainerParmIDs(gid))
            deflection = next(p for p in pids if vsp.GetParmName(p) == "DeflectionAngle")
            return deflection, sorted((p for p in pids if vsp.GetParmName(p).endswith("_Gain")), key=vsp.GetParmName)
    raise AeroAnalysisError(f"no VSPAERO control-surface group for {geom_name}")


def _sweep_inputs(model: VspModel, cond: FlightCondition, alphas: tuple[float, ...], beta: float, log: Path) -> None:
    vsp.SetAnalysisInputDefaults(SWEEP)
    ints = {"GeomSet": vsp.SET_NONE, "ThinGeomSet": LIFT_SET, "AlphaNpts": len(alphas), "BetaNpts": 1,
            "MachNpts": 1, "ReCrefNpts": 1, "RefFlag": vsp.COMPONENT_REF, "MACFlag": 1,
            "WakeNumIter": WAKE_ITERATIONS, "UnsteadyType": vsp.STABILITY_OFF}
    doubles = {"AlphaStart": alphas[0], "AlphaEnd": alphas[-1], "BetaStart": beta, "MachStart": cond.mach,
               "ReCref": cond.re_cref, "Vinf": cond.vinf_ms, "Rho": cond.rho,
               "Xcg": cond.x_ref_m, "Ycg": 0.0, "Zcg": cond.z_ref_m}
    for key, value in ints.items():
        vsp.SetIntAnalysisInput(SWEEP, key, [int(value)])
    for key, value in doubles.items():
        vsp.SetDoubleAnalysisInput(SWEEP, key, [float(value)])
    vsp.SetStringAnalysisInput(SWEEP, "WingID", [model.wing])
    vsp.SetStringAnalysisInput(SWEEP, "RedirectFile", [str(log)])


def _last(rid: str, key: str) -> float:
    return float(vsp.GetDoubleResults(rid, key)[-1])


def _run_sweep(model: VspModel, cond: FlightCondition, alphas: tuple[float, ...], beta: float,
               log: Path) -> tuple[tuple[AeroCoefficients, ...], ReferenceGeometry]:
    _sweep_inputs(model, cond, alphas, beta, log)
    rid: str = vsp.ExecAnalysis(SWEEP)
    histories = [r for r in vsp.GetStringResults(rid, "ResultsVec") if vsp.GetResultsName(r) == HISTORY] if rid else []
    if len(histories) != len(alphas):
        raise AeroAnalysisError(f"VSPAERO returned {len(histories)} cases for {len(alphas)} requested (see {log})")
    cases = tuple(
        AeroCoefficients(alpha_deg=_last(h, "Alpha"), beta_deg=_last(h, "Beta"), cl=_last(h, "CLtot"),
                         cdi=_last(h, "CDi"), cmx=_last(h, "CMxtot"), cmy=_last(h, "CMytot"), cmz=_last(h, "CMztot"))
        for h in histories
    )
    h = histories[0]
    ref = ReferenceGeometry(sref_m2=_last(h, "FC_Sref_"), cref_m=_last(h, "FC_Cref_"),
                            bref_m=_last(h, "FC_Bref_"), x_ref_m=cond.x_ref_m)
    return cases, ref


def _deflected_case(model: VspModel, cond: FlightCondition, geom_name: str, log: Path,
                    panel_gains: Optional[tuple[float, float]] = None) -> AeroCoefficients:
    """One DEFLECTION_DEG case of a geom's control group. `panel_gains` (main, mirror) for a mirrored surface; None
    keeps OpenVSP's equal gains (antisymmetric on a mirrored surface). The gains are restored afterwards."""
    deflection, gains = _control_group(geom_name)
    if panel_gains is not None:
        if len(gains) != len(panel_gains):
            raise AeroAnalysisError(f"{geom_name} has {len(gains)} control gains, expected a mirrored pair")
        for pid, gain in zip(gains, panel_gains):
            vsp.SetParmVal(pid, gain)
    vsp.SetParmVal(deflection, DEFLECTION_DEG)
    vsp.Update()
    try:
        (case,), _ = _run_sweep(model, cond, (0.0,), 0.0, log)
    finally:
        vsp.SetParmVal(deflection, 0.0)
        for pid in gains:
            vsp.SetParmVal(pid, DEFAULT_GAIN)
        vsp.Update()
    return case


def _tail_cases(model: VspModel, cond: FlightCondition, log: Path) -> dict[str, AeroCoefficients]:
    """The tail deflection cases by VspaeroCases field: elevator and rudder, or each ruddervator panel alone."""
    controls: frozenset[str] = frozenset(kind for kind, _ in model.tail_controls)
    if "ruddervator" in controls:
        pair: str = model.tail_named("ruddervator")
        return {"ruddervator_left": _deflected_case(model, cond, pair, log, LEFT_PANEL_GAINS),
                "ruddervator_right": _deflected_case(model, cond, pair, log, RIGHT_PANEL_GAINS)}
    return {"elevator": _deflected_case(model, cond, model.tail_named("elevator"), log, ELEVATOR_GAINS),
            "rudder": _deflected_case(model, cond, model.tail_named("rudder"), log)}


def run_vspaero(model: VspModel, cond: FlightCondition, work_dir: Path) -> tuple[VspaeroCases, ReferenceGeometry]:
    """The SRS run matrix: alpha sweep, beta = 5 deg, and 10 deg aileron / elevator / rudder (or each ruddervator)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    model_file: Path = work_dir / "model.vsp3"
    vsp.WriteVSPFile(str(model_file))
    vsp.SetVSP3FileName(str(model_file))  # VSPAERO writes its working files next to this
    log: Path = work_dir / "vspaero.log"
    vsp.SetAnalysisInputDefaults(COMPUTE_GEOMETRY)
    vsp.SetIntAnalysisInput(COMPUTE_GEOMETRY, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(COMPUTE_GEOMETRY, "ThinGeomSet", [LIFT_SET])
    vsp.ExecAnalysis(COMPUTE_GEOMETRY)
    sweep, ref = _run_sweep(model, cond, ALPHA_SWEEP_DEG, 0.0, log)
    (beta,), _ = _run_sweep(model, cond, (0.0,), BETA_DEG, log)
    cases = VspaeroCases(sweep=sweep, beta=beta, aileron=_deflected_case(model, cond, "Wing", log),
                         deflection_deg=DEFLECTION_DEG, **_tail_cases(model, cond, log))
    return cases, ref


def parasite_cd0(model: VspModel, cond: FlightCondition) -> float:
    """Component build-up zero-lift drag of the OML set, referenced to the wing area (with the laminar fraction the
    model was built with)."""
    pd = "ParasiteDrag"
    vsp.SetAnalysisInputDefaults(pd)
    for key, value in (("GeomSet", OML_SET), ("RefFlag", vsp.COMPONENT_REF), ("LengthUnit", vsp.LEN_M),
                       ("VelocityUnit", vsp.V_UNIT_M_S), ("AltLengthUnit", vsp.LEN_M),
                       ("FreestreamPropChoice", vsp.ATMOS_TYPE_US_STANDARD_1976)):
        vsp.SetIntAnalysisInput(pd, key, [int(value)])
    vsp.SetDoubleAnalysisInput(pd, "Altitude", [cond.altitude_m])
    vsp.SetDoubleAnalysisInput(pd, "Vinf", [cond.vinf_ms])
    vsp.SetStringAnalysisInput(pd, "WingID", [model.wing])
    rid: str = vsp.ExecAnalysis(pd)
    if not rid:
        raise AeroAnalysisError("OpenVSP ParasiteDrag analysis failed to run")
    cd0: float = float(vsp.GetDoubleResults(rid, "Total_CD_Total")[0])
    if not 0.0 < cd0 < MAX_PLAUSIBLE_CD0:
        raise AeroAnalysisError(f"OpenVSP ParasiteDrag returned non-physical CD0 = {cd0:.4g}")
    return cd0


def analyse_layout(layout: AircraftLayout, sizing: SizingOutput, cruise_altitude_m: float,
                   out_dir: Path, temperature_offset_c: float = 0.0) -> tuple[AeroStabilityOutput, Path, Path]:
    """Build the OML, export vsp3/STEP, run VSPAERO + ParasiteDrag and derive the stability set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[Tool] openvsp_builder: wing station %.3f", layout.wing_station)
    model: VspModel = build_model(layout, out_dir, surface_laminar_fraction(sizing.launch_method))
    vsp3, step = export_oml(out_dir)
    cond: FlightCondition = flight_condition(layout, sizing, cruise_altitude_m, temperature_offset_c)
    cases, ref = run_vspaero(model, cond, out_dir / "vspaero")
    cd0: float = parasite_cd0(model, cond)
    if sizing.cl_cruise is None:
        raise AeroAnalysisError("the sizing record predates the same-CL L/D check; re-run the sizing")
    return derive_stability(cases, ref, layout, cd0, sizing.cl_cruise), vsp3, step  # L/D at the sizing CL
