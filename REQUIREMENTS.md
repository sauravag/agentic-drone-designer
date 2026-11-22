# CLAUDE.md / REQUIREMENTS.md — Autonomous ISR Drone Design Agent ("ISR-Forge")

## 0. Claude Execution Directives (Read First)
You are acting as a Principal Aerospace Systems Engineer and AI Software Architect. Your task is to build **ISR-Forge**, an end-to-end neuro-symbolic AI engineering agent that designs **DoD Group 1, 2, or 3 Intelligence, Surveillance, and Reconnaissance (ISR) UAVs** from operational requirements.

### Strict Operating Rules for Claude:
1. **Milestone-Gated Execution:** Implement this project strictly in the 6 sequential milestones defined in Section 7. Do **not** skip ahead. After completing each milestone, write and run `pytest` tests to verify everything works before moving to the next milestone.
2. **Zero LLM Math Rule:** Never allow the LLM agent nodes to estimate weights, drag coefficients, battery capacities, PID gains, or CAD coordinates in prompts. Every engineering calculation must be performed by deterministic Python functions registered as tools.
3. **Strict SI Units Everywhere:** All internal data structures, schemas, physics calculations, and CAD generators must operate strictly in base SI units (`kg`, `m`, `s`, `W`, `J`, `N`, `Pa`, `rad`). Perform unit conversions (e.g., `knots` to `m/s`, `cm/s` for ArduPilot parameters, `inches` for propeller diameters) only at system I/O boundaries.
4. **Reproducible Environment First:** Because `OpenVSP` (Python API), `JSBSim`, and `ArduPilot SITL` require C++ binaries and system libraries, create the `Dockerfile` and `docker-compose.yml` in **Milestone 1** so all physics and SITL tests execute cleanly in a headless Ubuntu environment without host OS conflicts.

---

## 1. Project Overview & Deliverables
Given natural-language or structured operational requirements (e.g., *"Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, 6-hour endurance, and 80 km datalink radius"*), the system must autonomously generate:
1. **Converged Aircraft Sizing & Mass/Power Budget:** Supporting pure battery-electric and series hybrid-electric powertrains across DoD Group 1 (`< 9.07 kg`), Group 2 (`9.07–24.95 kg`), and Group 3 (`24.95–599 kg`).
2. **COTS Bill of Materials (`bom.json`):** Mapped to real-world Commercial Off-The-Shelf (COTS) motors, engines, generators, batteries, propellers, and payloads.
3. **Manufacturing-Ready 3D CAD (`oml.step` & `assembly.step`):** Watertight Outer Mold Line (OML) generated via OpenVSP, plus internal structural wing ribs, main carbon spars, and collision-free internal component packaging solved for Static Margin ($8\%\text{–}14\%$) via `Build123d`.
4. **Flight Control Parameters, checked in simulation (`ardupilot.param`):** ArduPilot `Plane` / `QuadPlane` parameter file derived from stability/control derivatives and 3D inertia tensors, checked via a headless `ArduPilot SITL + JSBSim` automated doublet test flight against two extreme corners of the roll/pitch aerodynamics: ±30 % aileron/elevator power and roll/pitch damping, ±20 % inertia, and both static-margin limits (0.08 and 0.14). Yaw/rudder terms and the VTOL rotor terms are not perturbed (the hover sees the inertia change but no CG imbalance), the transition altitude-loss gate uses the nominal flight, and mixed corners are not flown.
5. **Engineering Design Report (`design_report.md`):** Complete trade-off summary, mass breakdown, aerodynamic polars, stability report, and SITL per-plant telemetry metrics.

---

## 2. System Architecture & Tech Stack

### 2.1 Technology Stack
* **Language & Runtime:** Python 3.11+
* **Agent Orchestration:** `LangGraph` (cyclic state machine with conditional routing) + `Pydantic v2` (strict I/O validation).
* **Conceptual Sizing & MDO:** `AeroSandbox` (v4.2.10+, CasADi automatic differentiation backend + built-in `NeuralFoil`), `SciPy`, `NumPy`.
* **3D Aerodynamics & OML CAD:** `OpenVSP` (v3.56+ Python API `openvsp` + `VSPAERO` headless solver).
* **Internal Structural CAD & Inertia:** `Build123d` (OpenCascade Python B-Rep CAD) for solid modeling, rib/spar slicing, bounding-box placement, and exact 3D inertia tensor calculation.
* **Component Database:** `SQLite` with `SQLAlchemy` (or `sqlite3` + Pydantic models).
* **Flight Control & SITL Verification:** `ArduPilot SITL` (`arduplane`), `JSBSim` (Python `jsbsim` package + headless flight dynamics engine), and `pymavlink`.

### 2.2 Directory Structure
```text
isr-forge/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── REQUIREMENTS.md
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI entrypoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requirements.py          # OperationalRequirements schema
│   │   ├── components.py            # COTS hardware schemas
│   │   └── state.py                 # LangGraph DesignState & tool output schemas
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph StateGraph & conditional edges
│   │   ├── nodes.py                 # LLM parser, critic, and report compiler nodes
│   │   └── prompts.py               # System prompts for requirement parsing & critique
│   └── tools/
│       ├── sizing/
│       │   ├── __init__.py
│       │   ├── topology.py          # Group 1-3 & electric/hybrid selector rules
│       │   └── mission_solver.py    # AeroSandbox time-stepped electric & hybrid MDO solver
│       ├── database/
│       │   ├── __init__.py
│       │   ├── seed_data.py         # Realistic COTS catalog generator
│       │   └── matcher.py           # SQLite queries & hardware snapping logic
│       ├── aero_cad/
│       │   ├── __init__.py
│       │   ├── openvsp_builder.py   # Parametric OML generation & VSPAERO runner
│       │   └── cad_packer.py        # Build123d ribs/spars, CG optimizer & inertia calculator
│       └── flight_control/
│           ├── __init__.py
│           ├── param_generator.py   # Aero derivative + inertia -> ArduPilot .param mapper
│           ├── jsbsim_exporter.py   # Exports OpenVSP aero + CAD inertia to JSBSim XML
│           └── sitl_validator.py    # Headless ArduPilot SITL + pymavlink doublet test
├── data/
│   ├── cots_catalog.db              # Generated SQLite database
│   └── airfoils/                    # Standard airfoil .dat files
├── tests/
│   ├── test_database.py
│   ├── test_sizing.py
│   ├── test_aero_cad.py
│   ├── test_flight_control.py
│   └── test_agent_e2e.py
└── outputs/                         # Run artifacts grouped by timestamped run_id
```

---

## 3. Data Schemas (`src/schemas/`)

### 3.1 `OperationalRequirements`
```python
class OperationalRequirements(BaseModel):
    mission_name: str
    group_constraint: Optional[Literal[1, 2, 3]] = None  # Auto-assigned if None
    propulsion_type: Literal["electric", "series_hybrid", "auto"] = "auto"
    airframe_topology: Literal["fixed_wing", "quadplane_vtol", "auto"] = "auto"
    payload_mass_kg: float = Field(..., gt=0, le=200.0)
    payload_power_w: float = Field(..., ge=0, le=5000.0)  # Continuous hotel load (gimbal + compute + comms)
    endurance_hours: float = Field(..., gt=0.1, le=48.0)
    combat_radius_km: float = Field(..., gt=1.0, le=1000.0)
    cruise_speed_ms: float = Field(..., gt=8.0, le=120.0)
    service_ceiling_m: float = Field(default=3000.0, gt=100.0, le=10000.0)
    max_stall_speed_ms: float = Field(default=16.0, gt=5.0, le=45.0)  # True airspeed at the take-off site
    design_wind_ms: float = Field(default=0.0, ge=0.0, le=40.0)  # Headwind on ingress and egress; < cruise_speed_ms
    takeoff_altitude_m: float = Field(default=0.0, ge=0.0, le=5000.0)  # AMSL, < service_ceiling_m (also AMSL)
    isa_temperature_offset_c: float = Field(default=0.0, ge=-40.0, le=40.0)  # Applied to the whole atmosphere
    runway_available: bool = False
```

### 3.2 `DesignState` (LangGraph State)
```python
class DesignState(TypedDict):
    user_prompt: str
    requirements: Optional[OperationalRequirements]
    clarification_needed: Optional[str]
    iteration_count: int                  # Hard cap: 8 iterations
    sizing_overrides: dict[str, float]    # Adjustments from critic (e.g., aspect_ratio, tail_volume_h)
    sizing_results: Optional[SizingOutput]
    selected_bom: Optional[SelectedBOM]
    aero_stability: Optional[AeroStabilityOutput]
    cad_artifacts: Optional[CADArtifacts]
    fcs_artifacts: Optional[FCSArtifacts]
    validation_errors: list[str]
    final_report_path: Optional[str]
```

---

## 4. Detailed Module Specifications

### 4.1 Module A: Multi-Class & Hybrid-Electric Sizing Engine (`src/tools/sizing/`)
1. **DoD Group & Topology Classification (`topology.py`):**
   * Classify MTOW dynamically:
     * **Group 1:** $\text{MTOW} < 9.07\text{ kg}$ ($20\text{ lbs}$)
     * **Group 2:** $9.07\text{ kg} \le \text{MTOW} < 24.95\text{ kg}$ ($21\text{–}55\text{ lbs}$)
     * **Group 3:** $24.95\text{ kg} \le \text{MTOW} < 599.0\text{ kg}$ ($< 1,320\text{ lbs}$)
   * If `airframe_topology == "auto"`: select `"quadplane_vtol"` when `runway_available == False`, and `"fixed_wing"` when `runway_available == True`.
   * If `propulsion_type == "auto"`: select `"electric"` if estimated $\text{MTOW} < 15\text{ kg}$ and `endurance_hours <= 3.0`; otherwise select `"series_hybrid"`.
2. **Time-Stepped Mission Solver (`mission_solver.py` using `AeroSandbox`):**
   * Discretize the mission into 6 sequential phases:
     1. `VTOL_Takeoff` (if QuadPlane: 60s hover/climb at $T/W = 1.45$, induced power calculated via momentum theory + figure of merit $FM = 0.72$, at the take-off site density).
     2. `Climb` (climb at best rate-of-climb speed from `takeoff_altitude_m` to `service_ceiling_m`).
     3. `Ingress` (`combat_radius_km` at `cruise_speed_ms` into the design headwind: duration $R / (V_{\text{cruise}} - w)$).
     4. `Loiter` (`endurance_hours` at maximum $L/D$ speed $V_{\text{loiter}}$).
     5. `Egress` (`combat_radius_km` at `cruise_speed_ms`, again into the headwind).
     6. `VTOL_Land` (if QuadPlane: 60s descent/hover, at the take-off site density).
   * **Design day:** every density is ISA shifted by `isa_temperature_offset_c` (temperature only, at the same pressure). Hover, the VTOL phases and the stall constraint use the take-off site density; the stall constraint holds `max_stall_speed_ms` as a true airspeed there, so hot-and-high sites grow the wing. `SizingOutput.v_stall_ms` stays the sea-level equivalent airspeed (ArduPilot's airspeed parameters are equivalent airspeeds) and `v_stall_takeoff_ms` records the true stall speed at the site.
   * **Electric Powertrain Model:**
     * Pack-level specific energy: $E_{\text{spec}} = 200\text{ Wh/kg}$ (Li-Ion 21700), drivetrain efficiency $\eta_{\text{prop}} = 0.68$, minimum end-of-mission reserve $\text{SoC} \ge 20\%$.
     * Integrate total energy: $E_{\text{total}} = \sum (P_{\text{prop}, i} + P_{\text{payload}} + P_{\text{avionics}}) \cdot \Delta t_i$.
   * **Series Hybrid-Electric Powertrain Model:**
     * Size an Internal Combustion Engine (ICE) + starter-generator to deliver continuous power covering cruise propulsion + `payload_power_w` + battery recharge margin ($1.15 \times P_{\text{cruise, total}}$).
     * Size a high-discharge buffer LiPo battery ($E_{\text{spec}} = 155\text{ Wh/kg}$, $25\text{C}$ discharge) to cover the peak power deficit during VTOL hover and transition ($P_{\text{hover}} - P_{\text{gen}}$) for 180 seconds plus emergency abort reserve.
     * Compute fuel mass burned per segment using $\text{BSFC} = 0.50\text{ kg/kWh}$ (small EFI/heavy-fuel UAV engines) plus $10\%$ reserve fuel.
   * **Airfoil & Structural Scaling:**
     * Assign Reynolds-matched airfoils (`SD7037` for $Re < 3\times 10^5$; `MH 32` for $3\times 10^5 \le Re \le 10^6$; `NACA 4415` for $Re > 10^6$) and evaluate polar curves via `NeuralFoil`.
     * Enforce stall constraint: $S \ge \frac{2 \cdot \text{MTOW} \cdot g}{\rho \cdot V_{\text{stall}}^2 \cdot C_{L,\max}}$.
     * Output converged `SizingOutput`: `mtow_kg`, `empty_mass_kg`, `fuel_mass_kg`, `battery_mass_kg`, `wing_area_m2`, `wingspan_m`, `mean_chord_m`, `aspect_ratio`, `fuselage_length_m`, `fuselage_diameter_m`, `cruise_power_w`, `hover_power_w`, `v_stall_ms`, `v_loiter_ms`.

### 4.2 Module B: COTS Component Database & Matcher (`src/tools/database/`)
* Create `seed_data.py` to populate `data/cots_catalog.db` with **at least 15 realistic entries per category** spanning Group 1 (`2 kg`) through Group 3 (`500 kg`) scales:
  1. `cruise_motors`: `id, name, mass_kg, max_power_w, continuous_power_w, kv, voltage_v, diameter_m, length_m`
  2. `vtol_motors`: `id, name, mass_kg, max_thrust_n, max_power_w, voltage_v, diameter_m, length_m`
  3. `ice_generators`: `id, name, mass_kg, continuous_power_w, bsfc_kg_kwh, fuel_type, length_m, width_m, height_m`
  4. `batteries`: `id, name, chemistry, capacity_wh, mass_kg, max_c_rate, voltage_v, length_m, width_m, height_m`
  5. `propellers`: `id, name, diameter_m, pitch_m, mass_kg, max_power_w, application` (`cruise` or `vtol`)
  6. `payloads`: `id, name, sensor_type, mass_kg, power_w, length_m, width_m, height_m`
* **`matcher.py` Snapping Algorithm:**
  * Query components that meet or exceed required power/thrust/capacity with a **15% safety margin** while minimizing mass.
  * Sum actual discrete hardware masses (`selected_bom.total_hardware_mass_kg`) and verify MTOW convergence within $\pm 3\%$.

### 4.3 Module C: 3D CAD & Aerodynamic Validation (`src/tools/aero_cad/`)
1. **OpenVSP OML & VSPAERO (`openvsp_builder.py`):**
   * Use the `openvsp` Python API to instantiate:
     * Streamlined fuselage pod (`FuselageGeom`) sized to house internal payload, fuel tank, and avionics.
     * Tapered main wing (`WingGeom`) with trailing-edge aileron control surfaces (`SubSurface`).
     * Conventional or inverted-V tail empennage (`WingGeom`) with elevator and rudder control surfaces.
     * If `quadplane_vtol`: dual longitudinal twin booms (`PodGeom` / `WingGeom`) carrying 4 vertical lift rotor discs (`DiskGeom` / `PropGeom`).
   * Run `VSPAERO` (Vortex Lattice Method) across $\alpha \in [-4^\circ, 0^\circ, 4^\circ, 8^\circ, 12^\circ]$, sideslip $\beta = 5^\circ$, and control surface deflections ($\delta_a, \delta_e, \delta_r = 10^\circ$).
   * Extract and return `AeroStabilityOutput`: $C_{L_0}, C_{L_\alpha}, C_{D_0}, L/D_{\text{cruise}}, C_{m_\alpha}, C_{n_\beta}, C_{l_\beta}, C_{l_{\delta a}}, C_{m_{\delta e}}, C_{n_{\delta r}}$, and aerodynamic Neutral Point $X_{np}$ (meters from nose datum).
   * Export `oml.vsp3` and watertight `oml.step`.
2. **Build123d Internal Structure & CG Packaging (`cad_packer.py`):**
   * Load `oml.step` into `Build123d`.
   * Build the internal structure as a laser-cut kit for Group 1–2 designs (`src/tools/fabrication/`): ribs, formers, crutch, wing-mount plates and tail parts as flat plywood outlines, plus stock carbon tubes for the spars, tail boom and lift booms. Per-design `FabricationSettings` (bed, kerf, sheet and tube stock, allowables) set every input.
   * Select each spar as the smallest-OD stock tube from the tube list that passes the flight bending check (root moment of one panel at $n_{\text{limit}}$, elliptic lift, $\sigma \le \sigma_{\text{allow}}/FS$) and, for `quadplane_vtol`, the hover bending check (rotor thrust acting at the boom station), combined with torsion by $(\sigma/\sigma_{\text{allow}})^2 + (\tau/\tau_{\text{allow}})^2 \le 1/FS^2$, at the joint, at every spar step and at the boom clamp. The tube must also fit inside the local airfoil thickness less covering and a minimum rib web; at equal OD, lift booms take the lighter wall. The spar is not a fixed fraction of airfoil thickness.
   * Check the wing's stiffness as well as its strength (R43-R45): tip deflection at 1 g of at most 5 % of the semi-span (tube bending only), and an aileron reversal speed above 1.15 × V_NE with V_NE = 1.4 × cruise speed. A wing whose tube passes both needs nothing more. If only the reversal check fails, a 1.5 mm balsa leading-edge D-box (hand-wrapped sheet and a spar shear web, with the rib noses inset to seat it) is added and counted for torsion only, never for bending; its area and mass appear in the covering and the kit mass. If the tip deflection fails, or the reversal still fails with the D-box, the kit is unavailable. The build sheet's "Wing stiffness" section (after "Covering") gives both results against their limits, whether a D-box is fitted, and its build note.
   * Join the two fuselage socket tubes (`SK-R`/`SK-L`) at the centreline with a glued mitre and a wet-laid carbon overwrap over it, sized for the socket moment (smallest layer count within the allowed maximum, with the lap sized for epoxy lap shear). The spar roots slide into the sockets unglued, so the wings stay detachable. Also check the tail boom (bending and torsion), socket bearing in the wing-mount plates (adding a boss doubler when needed) and crutch rail width. If a check cannot be met by any stock tube or doubler, or a part cannot fit the bed even when split, the kit is unavailable.
   * Size the fuselage pod kit-aware (wide enough for the crutch rails, battery seated low), and feed the kit masses into the as-built mass and CG in place of the shell-model lump. Without a kit (Group 3 designs or an unavailable kit) the shell model is used and the run reports a warning; the run status is unaffected.
   * Write the kit outputs to `kit/`: nested DXF sheets per stock (`kit/sheets/`, millimetres, layers `CUT_OUTER`, `CUT_INNER`, `ENGRAVE`, kerf-compensated), `kit/build_sheet.md` and `kit/parts.csv`. The kit parts and tubes are also included in `assembly.step` and the viewer model.
   * Create parametric solid bounding boxes for every item in `SelectedBOM` plus a cylindrical fuel tank (if hybrid).
   * **1D Longitudinal CG Optimization:** Place the EO/IR gimbal in the forward nose/chin bay and fuel tank centroid directly at $X_{np}$ (so fuel burn causes zero CG shift). Use `scipy.optimize.minimize_scalar` to slide the battery pack and avionics tray along the fuselage $X$-axis until Static Margin:
     $$SM = \frac{X_{np} - X_{cg}}{\bar{c}} \in [0.08, 0.14]$$
     while enforcing zero 3D bounding-box intersections (`shape1.intersect(shape2).volume == 0`).
   * Compute full 3D inertia tensor ($I_{xx}, I_{yy}, I_{zz}, I_{xz}$) combining composite shell mass ($\rho = 1600\text{ kg/m}^3$) and discrete component point/box masses, and export `assembly.step`.

### 4.4 Module D: Flight Control Parameter Generation & SITL Gate (`src/tools/flight_control/`)
1. **ArduPilot Parameter Translator (`param_generator.py`):**
   * Generate a complete, valid `ardupilot.param` file for **ArduPlane / QuadPlane**:
     * **Airspeed & Envelope:**
       * `ARSPD_FBW_MIN` = $\text{round}(1.25 \cdot V_{\text{stall}}, 1)$
       * `ARSPD_FBW_MAX` = $\text{round}(0.85 \cdot V_{NE}, 1)$
       * `TRIM_ARSPD_CM` = $\text{int}(100 \cdot V_{\text{cruise}})$
     * **Fixed-Wing Attitude & Rate Loops:**
       * Compute dimensional control effectiveness $L_{\delta a} = \frac{q S b C_{l_{\delta a}}}{I_{xx}}$ and $M_{\delta e} = \frac{q S \bar{c} C_{m_{\delta e}}}{I_{yy}}$.
       * Set feed-forward gains `RLL2SRV_FF` $= \text{clamp}(K_{ff,\phi} / L_{\delta a}, 0.15, 1.5)$ and `PTCH2SRV_FF` $= \text{clamp}(K_{ff,\theta} / M_{\delta e}, 0.15, 1.5)$, and scale `RLL2SRV_P/I/D` and `PTCH2SRV_P/I/D` inversely with inertia-to-control-power ratios.
     * **QuadPlane VTOL Parameters (when `airframe_topology == "quadplane_vtol"`):**
       * `Q_ENABLE = 1`, `Q_FRAME_CLASS = 1` (Quad), `Q_FRAME_TYPE = 1` (X-frame)
       * `Q_ASSIST_SPEED` = $\text{round}(1.15 \cdot V_{\text{stall}}, 1)$
       * `Q_TRANSITION_MS` = computed from cruise motor acceleration time from $0 \to 1.2 \cdot V_{\text{stall}}$
       * Scale `Q_A_RAT_RLL_P` and `Q_A_RAT_PIT_P` proportional to $\frac{I_{xx}}{L_{\text{arm}} \cdot T_{\max}}$ and $\frac{I_{yy}}{L_{\text{arm}} \cdot T_{\max}}$.
     * **Energy & Failsafes:** Set `BATT_CAPACITY`, `BATT_LOW_MAH`, and `BATT_CRT_MAH` directly from `SelectedBOM`.
2. **Headless SITL Robustness Check (`sitl_validator.py`):**
   * `jsbsim_exporter.py`: Write a `JSBSim` aircraft configuration XML using the OpenVSP aerodynamic derivatives, wing geometry, propulsion thrust tables, and Build123d inertia tensor.
   * Launch `arduplane` SITL connected to the JSBSim dynamics model headlessly.
   * Connect via `pymavlink` (UDP `127.0.0.1:14550`), arm vehicle, switch to `AUTO` / `FBWA`, execute takeoff, transition to fixed-wing cruise (if QuadPlane), and inject $+15^\circ$ roll and $+10^\circ$ pitch step commands.
   * Analyze telemetry attitude logs: verify **overshoot $< 20\%$**, **settling time $< 3.0\text{ s}$**, and **no altitude loss $> 15\text{ m}$** during transition. Return `FCSArtifacts` with pass/fail status and damping metrics.

---

## 5. LangGraph Agent Workflow (`src/agent/graph.py`)

Build a cyclical `StateGraph(DesignState)` with the following nodes and routing rules:

1. **`parse_requirements` (LLM Node):** Parses `user_prompt` into `OperationalRequirements` using structured output. If required fields (`payload_mass_kg` or `endurance_hours`) are absent or ambiguous, sets `clarification_needed` and halts cleanly for user input.
2. **`run_sizing` (Tool Node):** Executes Module A (`mission_solver.py`) applying any `sizing_overrides`.
3. **`match_cots` (Tool Node):** Executes Module B (`matcher.py`) to select real COTS components.
4. **`build_aero_cad` (Tool Node):** Executes Module C (`openvsp_builder.py` $\rightarrow$ `cad_packer.py`).
5. **`design_critic` (LLM + Deterministic Check Node):**
   * Evaluates 4 hard engineering gates:
     1. Static Margin $0.08 \le SM \le 0.14$
     2. VSPAERO $L/D$ within $\pm 15\%$ of the sizing $L/D$ at the sizing cruise **and** loiter $C_L$ (both polars evaluated at the same $C_L$; VSPAERO's inviscid $k$ gets Kroo's viscous lift-dependent drag, $k = k_{\text{VLM}} + 0.38\,C_{D_0}$). Only a low VSPAERO $L/D$ triggers the aspect-ratio rule fix; a high one has no automatic fix.
     3. Zero internal CAD component collisions and MTOW within DoD Group limit
     4. Sufficient battery/fuel reserve margin ($\ge 15\%$)
   * **Conditional Edge:**
     * If all gates pass $\rightarrow$ route to `generate_and_verify_fcs`.
     * If any gate fails and `iteration_count < 8` $\rightarrow$ LLM reviews `validation_errors`, updates `sizing_overrides` (e.g., increases `aspect_ratio` if $L/D$ is too low, increases `tail_volume_h` or `fuselage_length_factor` if $SM$ is out of bounds), increments `iteration_count`, and routes back to `run_sizing`.
     * If `iteration_count >= 8` $\rightarrow$ route to `compile_report` with explicit non-convergence warnings.
6. **`generate_and_verify_fcs` (Tool Node):** Executes Module D (`param_generator.py` + `sitl_validator.py`). If the SITL check fails, automatically retries up to 3 times with the rate gains scaled by $0.75\times$ after an overshoot failure or a crashed plant, or by $1/0.75\times$ after a settling-only failure (the fixed-wing rate D gain is never raised above its model value: a raised D drove the high-control-power plant into a ~13 Hz rate-loop oscillation). A step that would reverse the previous one stops the retries: no gain scale on the 0.75 step grid satisfies both plant corners. Raising the gains is capped at $1/0.75^2 \approx 1.78\times$; an up-step past it also stops the retries, and the quadplane rate P stays within its own clamp. When no attempt passes, the least severe attempt (largest limit ratio of the worst plant; a crash counts as infinite) is shipped, ties going to the lower gain scale.
7. **`compile_report` (LLM Node):** Generates `outputs/<run_id>/design_report.md` summarizing operational compliance, mass/power budget, BOM table, aerodynamic stability derivatives, CAD file paths, and SITL flight test metrics.

---

## 6. Docker & Environment Configuration (Milestone 1 Deliverable)
Provide a `Dockerfile` based on `ubuntu:22.04` that installs:
* Python 3.11, `pip`, `git`, `cmake`, `build-essential`, `libgl1-mesa-glx`, `libglu1-mesa`, `xvfb`.
* `OpenVSP` (pre-compiled Ubuntu 22.04 release with Python 3.11 `openvsp` wheel installed into site-packages + `vspaero` on `PATH`).
* `ArduPilot` (`arduplane` SITL binary compiled via `./waf configure --board sitl && ./waf plane`).
* Python dependencies via `pyproject.toml`: `langgraph`, `langchain-anthropic` (or `langchain-openai`), `pydantic>=2.0`, `aerosandbox`, `neuralfoil`, `casadi`, `scipy`, `numpy`, `build123d`, `jsbsim`, `pymavlink`, `pytest`.

---

## 7. Sequential Implementation Milestones & Acceptance Tests

Implement and verify one milestone at a time:

* **Milestone 1 — Project Scaffold, Docker & COTS Database:**
  * Create repository structure, `Dockerfile`, `pyproject.toml`, Pydantic schemas (`src/schemas/`), and seeded SQLite DB (`data/cots_catalog.db`).
  * *Acceptance Test:* `pytest tests/test_database.py` passes (verifies $\ge 15$ items per table, valid SI units, and query matching across Group 1–3 power scales).
* **Milestone 2 — Multi-Class Electric & Series-Hybrid Sizing Engine:**
  * Implement `src/tools/sizing/topology.py` and `mission_solver.py`.
  * *Acceptance Test:* `pytest tests/test_sizing.py` passes for 3 reference cases:
    1. Group 1 Electric Fixed-Wing (`1.0 kg` payload, `2.0 hr` endurance $\rightarrow$ $\text{MTOW} < 9.07\text{ kg}$).
    2. Group 2 Electric QuadPlane VTOL (`3.0 kg` payload, `3.0 hr` endurance $\rightarrow$ $9.07 \le \text{MTOW} < 24.95\text{ kg}$).
    3. Group 3 Series-Hybrid QuadPlane VTOL (`20.0 kg` payload, `12.0 hr` endurance $\rightarrow$ $24.95 \le \text{MTOW} < 599\text{ kg}$).
* **Milestone 3 — OpenVSP OML, VSPAERO & Build123d Internal CAD:**
  * Implement `src/tools/aero_cad/openvsp_builder.py` and `cad_packer.py`.
  * *Acceptance Test:* `pytest tests/test_aero_cad.py` generates valid `oml.step` and `assembly.step` files, extracts non-zero VSPAERO stability/control derivatives, achieves $0.08 \le SM \le 0.14$ with zero component collisions, and outputs a positive-definite inertia tensor.
* **Milestone 4 — ArduPilot `.param` Generator & Headless SITL Validator:**
  * Implement `param_generator.py`, `jsbsim_exporter.py`, and `sitl_validator.py`.
  * *Acceptance Test:* `pytest tests/test_flight_control.py` generates `ardupilot.param`, runs headless `arduplane` + `JSBSim` SITL through takeoff and roll/pitch doublet maneuvers, and passes overshoot ($< 20\%$) and settling time ($< 3.0\text{ s}$) assertions.
* **Milestone 5 — LangGraph Agent Orchestration & Critic Loop:**
  * Implement `src/agent/` (`prompts.py`, `nodes.py`, `graph.py`) and `src/main.py` CLI.
  * *Acceptance Test:* `pytest tests/test_agent_e2e.py` executes a full end-to-end run from a natural language prompt (including forcing at least 1 critic loop iteration via an initially tight constraint) and writes `oml.step`, `assembly.step`, `bom.json`, `ardupilot.param`, `sitl_report.json`, and `design_report.md` to `outputs/<run_id>/`.