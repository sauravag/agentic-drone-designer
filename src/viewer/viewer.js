// Aircraft 3D viewer: the packed aircraft in Shape / X-ray modes and, when a SITL trace exists, its replay.
// Run data comes only from the page's JSON data block (#viewer-data) and reaches the DOM via textContent only.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DEG = Math.PI / 180;
const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
  '#469990', '#9a6324', '#800000', '#808000', '#000075', '#fabed4', '#ffd8b1', '#aaffc3'];
const SKIN_COLOR = '#9fb3c8';
const STRUCTURE_COLOR = '#8a8f96';  // shell-model ribs and spars
const KIT_STRUCTURE_COLORS = Object.freeze({ 'structure (ply)': '#c9a66b', 'structure (carbon)': '#2e3238' });
const TRAIL_COLOR = '#f59e0b';
const SCALE_BAR_COLOR = '#f97316';
const XRAY_SKIN_OPACITY = 0.2;
const HIGHLIGHT_EMISSIVE = 0x444444;
const CG_SPHERE_FRACTION = 0.025; // CG sphere radius as a fraction of the aircraft's bounding radius
const CG_AXIS_FRACTION = 0.3;     // CG axis length as a fraction of the bounding radius
const MIN_RADIUS_M = 0.1;
const GROUND_CLEARANCE_M = 0.01;
const SCALE_BAR_OFFSET_RADII = 1.2;
const FINE_GRID_M = 40;           // 1 m cells around the start point
const COARSE_GRID_M = 4000;       // 10 m cells out to the flight path
const COARSE_GRID_CELL_M = 10;
const CHASE_DISTANCE_RADII = 3.5;
const CHASE_HEIGHT_RADII = 1.0;
const START_VIEW = new THREE.Vector3(1.6, 1.0, -2.0); // from the aircraft, in bounding radii: front-right-above
const CAMERA_FOV_DEG = 45;
const CAMERA_NEAR_M = 0.02;
const CAMERA_FAR_M = 20000;
const MAX_FRAME_DT_S = 0.1;       // clamp so a background tab does not jump the replay

// ---------------------------------------------------------------------------------------------------------
// Frames. Every conversion between conventions lives here.
//
//   CAD   (viewer_model.json): metres, x aft from the nose, y right, z up; origin at the nose.
//   Body  (aircraft):          x forward, y right, z down; origin at the CG (the JSBSim aero reference point).
//           x_b = -(x - x_cg),  y_b = y - y_cg,  z_b = -(z - z_cg)   (a 180° turn about y: winding kept)
//   NED   (trace):             x north, y east, z down. Attitude is Z-Y-X Euler: v_ned = Rz(ψ)·Ry(θ)·Rx(φ)·v_body
//           with φ = roll_deg (right wing down > 0), θ = pitch_deg (nose up > 0), ψ = heading_deg (0 = north,
//           90 = east). Position: north_m, east_m; altitude = agl_m.
//   Scene (Three.js):          right-handed, y up:  X = east,  Y = -down (= altitude),  Z = -north.
// The aircraft's geometry is stored in body coordinates; its Object3D carries (NED→scene)·(body→NED).
// ---------------------------------------------------------------------------------------------------------

/** CAD-frame vertex positions (flat xyz) -> body-frame positions about the CG, as a new array. */
function cadToBody(cadPositions, cg) {
  const body = new Float32Array(cadPositions.length);
  for (let i = 0; i < cadPositions.length; i += 3) {
    body[i] = -(cadPositions[i] - cg[0]);
    body[i + 1] = cadPositions[i + 1] - cg[1];
    body[i + 2] = -(cadPositions[i + 2] - cg[2]);
  }
  return body;
}

// Rows of the matrix are scene X, Y, Z in NED components: X = east, Y = -down, Z = -north.
const NED_TO_SCENE = new THREE.Matrix4().set(
  0, 1, 0, 0,
  0, 0, -1, 0,
  -1, 0, 0, 0,
  0, 0, 0, 1);
const NED_TO_SCENE_Q = new THREE.Quaternion().setFromRotationMatrix(NED_TO_SCENE);

/** Body -> NED rotation. Three.js Euler order 'ZYX' is the matrix Rz(z)·Ry(y)·Rx(x), i.e. Z-Y-X Euler. */
function bodyToNed(rollDeg, pitchDeg, headingDeg) {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(rollDeg * DEG, pitchDeg * DEG, headingDeg * DEG, 'ZYX'));
}

/** Body -> scene rotation for an attitude: (NED -> scene)·(body -> NED). */
function attitudeToScene(rollDeg, pitchDeg, headingDeg) {
  return NED_TO_SCENE_Q.clone().multiply(bodyToNed(rollDeg, pitchDeg, headingDeg));
}

/** NED position (north, east) and altitude above ground -> scene position. */
function nedToScene(northM, eastM, altitudeM) {
  return new THREE.Vector3(eastM, altitudeM, -northM);
}

/** Horizontal forward direction in the scene for a heading (NED (cos ψ, sin ψ, 0) -> scene). */
function headingForward(headingDeg) {
  return new THREE.Vector3(Math.sin(headingDeg * DEG), 0, -Math.cos(headingDeg * DEG));
}

// ---------------------------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------------------------

function readData() {
  return JSON.parse(document.getElementById('viewer-data').textContent);
}

/** Base64 -> typed array. The payload is little-endian, as are the typed arrays of every browser platform. */
function decodeBase64(b64, ArrayType) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  if (bytes.length % ArrayType.BYTES_PER_ELEMENT !== 0) throw new Error('mesh payload has a partial element');
  return new ArrayType(bytes.buffer);
}

function partGeometry(part, cg) {
  if (!part.positions_b64 || !part.indices_b64) return null;
  const positions = decodeBase64(part.positions_b64, Float32Array);
  const indices = decodeBase64(part.indices_b64, Uint32Array);
  if (positions.length % 3 !== 0 || indices.length % 3 !== 0) throw new Error(`part "${part.name}" is malformed`);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(cadToBody(positions, cg), 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function formatMass(massKg) {
  return typeof massKg === 'number' ? `${massKg.toFixed(2)} kg` : '—';
}

// ---------------------------------------------------------------------------------------------------------
// Scene: aircraft, CG marker, ground, scale bar
// ---------------------------------------------------------------------------------------------------------

function partMaterial(group, color) {
  const base = group === 'skin' ? SKIN_COLOR : color;
  return new THREE.MeshStandardMaterial({ color: base, side: THREE.DoubleSide, metalness: 0.1, roughness: 0.7 });
}

function structureColor(name) {
  return Object.hasOwn(KIT_STRUCTURE_COLORS, name) ? KIT_STRUCTURE_COLORS[name] : STRUCTURE_COLOR;
}

function buildAircraft(model) {
  const group = new THREE.Group();
  const entries = [];
  let componentIndex = 0;
  for (const part of model.parts) {
    let color = null;
    if (part.group === 'component') color = PALETTE[componentIndex++ % PALETTE.length];
    else if (part.group === 'structure') color = structureColor(part.name);
    const geometry = partGeometry(part, model.cg_m);
    const mesh = geometry ? new THREE.Mesh(geometry, partMaterial(part.group, color)) : null;
    if (mesh) {
      mesh.userData.part = part;
      group.add(mesh);
    }
    entries.push({ part, mesh, color });
  }
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
  const radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, MIN_RADIUS_M);
  return { group, entries, radius, lowestBodyZ: box.max.z }; // body z is down: max z is the lowest point
}

function axisLine(direction, length, color) {
  const points = [new THREE.Vector3(), direction.clone().multiplyScalar(length)];
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, depthTest: false }));
}

/** CG sphere and body axes (x forward red, y right green, z down blue), drawn on top of the skin. */
function buildCgMarker(radius) {
  const marker = new THREE.Group();
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius * CG_SPHERE_FRACTION, 16, 12),
    new THREE.MeshBasicMaterial({ color: '#111111', depthTest: false }));
  const length = radius * CG_AXIS_FRACTION;
  marker.add(sphere, axisLine(new THREE.Vector3(1, 0, 0), length, '#e53935'),
    axisLine(new THREE.Vector3(0, 1, 0), length, '#43a047'), axisLine(new THREE.Vector3(0, 0, 1), length, '#1e88e5'));
  marker.traverse((object) => { object.renderOrder = 10; });
  return marker;
}

function themeColors() {
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  return dark ? { background: '#14181d', fine: '#4b5563', coarse: '#2b323b' }
    : { background: '#f4f5f7', fine: '#9ca3af', coarse: '#d1d5db' };
}

function buildGround(colors) {
  const ground = new THREE.Group();
  const coarse = new THREE.GridHelper(COARSE_GRID_M, COARSE_GRID_M / COARSE_GRID_CELL_M, colors.coarse, colors.coarse);
  const fine = new THREE.GridHelper(FINE_GRID_M, FINE_GRID_M, colors.fine, colors.fine);
  fine.position.y = GROUND_CLEARANCE_M / 2; // above the coarse grid so the two never z-fight
  ground.add(coarse, fine);
  return ground;
}

/** A 1 m bar lying on the ground along scene X (east), `offsetM` south of the start point. */
function buildScaleBar(offsetM) {
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.01, 0.04), new THREE.MeshBasicMaterial({ color: SCALE_BAR_COLOR }));
  bar.position.set(0.5, GROUND_CLEARANCE_M, offsetM);
  return bar;
}

function createStage(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 2.0));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(3, 10, 4);
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(CAMERA_FOV_DEG, 1, CAMERA_NEAR_M, CAMERA_FAR_M);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const resize = () => {
    const { clientWidth: width, clientHeight: height } = container;
    renderer.setSize(width, height);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', resize);
  resize();
  return { renderer, scene, camera, controls };
}

// ---------------------------------------------------------------------------------------------------------
// Modes, legend, hover tooltip, CG label
// ---------------------------------------------------------------------------------------------------------

function applyMode(aircraft, cgMarker, xray) {
  for (const { part, mesh } of aircraft.entries) {
    if (!mesh) continue;
    if (part.group !== 'skin') {
      mesh.visible = xray;
      continue;
    }
    mesh.material.transparent = xray;
    mesh.material.opacity = xray ? XRAY_SKIN_OPACITY : 1;
    mesh.material.depthWrite = !xray;
    mesh.material.needsUpdate = true;
  }
  cgMarker.visible = xray;
  document.getElementById('legend').hidden = !xray;
  document.getElementById('cg-label').hidden = !xray;
  document.getElementById('mode-shape').setAttribute('aria-pressed', String(!xray));
  document.getElementById('mode-xray').setAttribute('aria-pressed', String(xray));
  if (!xray) document.getElementById('tooltip').hidden = true;
}

function fillLegend(aircraft) {
  const list = document.getElementById('legend-list');
  for (const { part, color } of aircraft.entries) {
    if (part.group === 'skin') continue; // structure (ply / carbon, or the shell model's) and every component
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = color;
    const name = document.createElement('span');
    name.textContent = part.name;
    const mass = document.createElement('span');
    mass.className = 'mass';
    mass.textContent = formatMass(part.mass_kg);
    const item = document.createElement('li');
    item.append(swatch, name, mass);
    list.append(item);
  }
}

function setupHover(stage, aircraft, isXray) {
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const tooltip = document.getElementById('tooltip');
  const targets = aircraft.entries.filter((e) => e.mesh && e.part.group === 'component').map((e) => e.mesh);
  let highlighted = null;
  const highlight = (mesh) => {
    if (highlighted === mesh) return;
    if (highlighted) highlighted.material.emissive.setHex(0x000000);
    if (mesh) mesh.material.emissive.setHex(HIGHLIGHT_EMISSIVE);
    highlighted = mesh;
  };
  stage.renderer.domElement.addEventListener('pointermove', (event) => {
    const rect = stage.renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, stage.camera);
    const hit = isXray() ? raycaster.intersectObjects(targets, false)[0] : undefined;
    highlight(hit ? hit.object : null);
    tooltip.hidden = !hit;
    if (!hit) return;
    const part = hit.object.userData.part;
    tooltip.textContent = `${part.name} — ${formatMass(part.mass_kg)}`;
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  });
  stage.renderer.domElement.addEventListener('pointerleave', () => {
    highlight(null);
    tooltip.hidden = true;
  });
}

function cgLabelText(model) {
  const [x, y, z] = model.cg_m.map((v) => v.toFixed(3));
  return `CG x ${x} (aft of nose), y ${y}, z ${z} m · ${formatMass(model.mass_kg)}`;
}

function placeCgLabel(label, aircraftGroup, camera, canvas) {
  if (label.hidden) return;
  const ndc = aircraftGroup.getWorldPosition(new THREE.Vector3()).project(camera);
  label.style.left = `${((ndc.x + 1) / 2) * canvas.clientWidth + 12}px`;
  label.style.top = `${((1 - ndc.y) / 2) * canvas.clientHeight + 12}px`;
}

// ---------------------------------------------------------------------------------------------------------
// Replay: sample interpolation, controls, camera follow
// ---------------------------------------------------------------------------------------------------------

function prepareTrace(trace) {
  const hasHeading = trace.every((s) => typeof s.heading_deg === 'number');
  const hasPosition = trace.every((s) => typeof s.north_m === 'number' && typeof s.east_m === 'number');
  const samples = trace.map((s) => ({
    raw: s,
    heading: hasHeading ? s.heading_deg : 0,
    position: hasPosition ? nedToScene(s.north_m, s.east_m, s.agl_m) : nedToScene(0, 0, s.agl_m),
    quaternion: attitudeToScene(s.roll_deg, s.pitch_deg, hasHeading ? s.heading_deg : 0),
  }));
  return { samples, hasPose: hasHeading && hasPosition, start: trace[0].t_s, end: trace[trace.length - 1].t_s };
}

/** Index of the last sample with t_s <= t (binary search; samples are in time order). */
function sampleIndex(samples, t) {
  let low = 0;
  let high = samples.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (samples[mid].raw.t_s <= t) low = mid; else high = mid - 1;
  }
  return low;
}

/** The interpolated state at time t: position linear, attitude by slerp, readouts linear. */
function stateAt(prepared, t) {
  const { samples } = prepared;
  const i = sampleIndex(samples, t);
  const a = samples[i];
  const b = samples[Math.min(i + 1, samples.length - 1)];
  const span = b.raw.t_s - a.raw.t_s;
  const f = span > 0 ? THREE.MathUtils.clamp((t - a.raw.t_s) / span, 0, 1) : 0;
  const lerp = (key) => a.raw[key] + (b.raw[key] - a.raw[key]) * f;
  const headingStep = ((b.heading - a.heading + 540) % 360) - 180; // shortest way round
  return {
    index: i,
    position: a.position.clone().lerp(b.position, f),
    quaternion: a.quaternion.clone().slerp(b.quaternion, f),
    t, altitude: lerp('agl_m'), airspeed: lerp('airspeed_ms'), roll: lerp('roll_deg'), pitch: lerp('pitch_deg'),
    heading: (((a.heading + headingStep * f) % 360) + 360) % 360,
  };
}

function buildTrail(prepared) {
  const geometry = new THREE.BufferGeometry().setFromPoints(prepared.samples.map((s) => s.position));
  geometry.setDrawRange(0, 1);
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: TRAIL_COLOR }));
}

function showReadout(state, hasPose) {
  const set = (id, text) => { document.getElementById(id).textContent = text; };
  set('r-t', `${state.t.toFixed(1)} s`);
  set('r-alt', `${state.altitude.toFixed(1)} m`);
  set('r-speed', `${state.airspeed.toFixed(1)} m/s`);
  set('r-roll', `${state.roll.toFixed(1)}°`);
  set('r-pitch', `${state.pitch.toFixed(1)}°`);
  set('r-heading', hasPose ? `${Math.round(state.heading) % 360}°` : '—'); // 359.6 reads 0, not 360
}

/** Moves the camera with the aircraft: orbit keeps the user's offset, chase sits behind along the heading. */
function followCamera(stage, cameraMode, previous, state, radius) {
  const { camera, controls } = stage;
  if (cameraMode === 'chase') {
    const behind = headingForward(state.heading).multiplyScalar(-CHASE_DISTANCE_RADII * radius);
    camera.position.copy(state.position).add(behind).add(new THREE.Vector3(0, CHASE_HEIGHT_RADII * radius, 0));
    controls.target.copy(state.position);
    camera.lookAt(state.position);
    return;
  }
  const delta = state.position.clone().sub(previous);
  camera.position.add(delta);
  controls.target.add(delta);
}

function wireReplayControls(replay, prepared, seek, setPlaying, stage) {
  document.getElementById('play').addEventListener('click', () => {
    if (!replay.playing && replay.t >= prepared.end) seek(prepared.start);
    setPlaying(!replay.playing);
  });
  const slider = document.getElementById('time');
  slider.addEventListener('input', () => seek(Number(slider.value)));
  document.getElementById('speed').addEventListener('change', (e) => { replay.speed = Number(e.target.value); });
  document.getElementById('camera').addEventListener('change', (e) => {
    replay.cameraMode = e.target.value;
    stage.controls.enabled = replay.cameraMode === 'orbit';
    seek(replay.t);
  });
}

function setupReplay(stage, aircraft, trace, sitlPassed) {
  const prepared = prepareTrace(trace);
  const trail = buildTrail(prepared);
  stage.scene.add(trail);
  document.body.classList.add('has-replay');
  document.getElementById('replay').hidden = false;
  document.getElementById('note').hidden = prepared.hasPose;
  document.getElementById('sitl-badge').hidden = sitlPassed !== false;
  const slider = document.getElementById('time');
  slider.min = String(prepared.start);
  slider.max = String(prepared.end);
  const play = document.getElementById('play');
  const replay = { t: prepared.start, playing: false, speed: 1, cameraMode: 'orbit',
    position: stateAt(prepared, prepared.start).position };

  const seek = (t) => {
    const state = stateAt(prepared, THREE.MathUtils.clamp(t, prepared.start, prepared.end));
    aircraft.group.position.copy(state.position);
    aircraft.group.quaternion.copy(state.quaternion);
    followCamera(stage, replay.cameraMode, replay.position, state, aircraft.radius);
    replay.position = state.position;
    replay.t = state.t;
    trail.geometry.setDrawRange(0, state.index + 1);
    slider.value = String(state.t);
    showReadout(state, prepared.hasPose);
  };
  const setPlaying = (playing) => {
    replay.playing = playing;
    play.textContent = playing ? 'Pause' : 'Play';
  };
  wireReplayControls(replay, prepared, seek, setPlaying, stage);
  seek(prepared.start);
  return {
    seek,
    tick(dt) {
      if (!replay.playing) return;
      seek(replay.t + dt * replay.speed);
      if (replay.t >= prepared.end) setPlaying(false);
    },
  };
}

// ---------------------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------------------

function placeStatic(aircraft) {
  aircraft.group.quaternion.copy(attitudeToScene(0, 0, 0));
  aircraft.group.position.set(0, aircraft.lowestBodyZ + GROUND_CLEARANCE_M, 0); // resting on the grid
}

function frameCamera(stage, aircraft) {
  const target = aircraft.group.position;
  stage.camera.position.copy(target).addScaledVector(START_VIEW, aircraft.radius);
  stage.controls.target.copy(target);
  stage.camera.lookAt(target);
}

function setupTheme(stage, groundHolder) {
  const apply = () => {
    const colors = themeColors();
    stage.scene.background = new THREE.Color(colors.background);
    groundHolder.clear();
    groundHolder.add(buildGround(colors));
  };
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
  apply();
}

function setupModes(stage, aircraft, cgMarker, model) {
  let xray = false;
  const setXray = (value) => { xray = value; applyMode(aircraft, cgMarker, xray); };
  document.getElementById('mode-shape').addEventListener('click', () => setXray(false));
  document.getElementById('mode-xray').addEventListener('click', () => setXray(true));
  fillLegend(aircraft);
  document.getElementById('cg-label').textContent = cgLabelText(model);
  setupHover(stage, aircraft, () => xray);
  setXray(false);
}

function main() {
  const data = readData();
  const stage = createStage(document.getElementById('stage'));
  const aircraft = buildAircraft(data.model);
  const cgMarker = buildCgMarker(aircraft.radius);
  aircraft.group.add(cgMarker);
  const groundHolder = new THREE.Group();
  stage.scene.add(aircraft.group, groundHolder, buildScaleBar(aircraft.radius * SCALE_BAR_OFFSET_RADII));
  setupTheme(stage, groundHolder);
  setupModes(stage, aircraft, cgMarker, data.model);

  placeStatic(aircraft);
  const replay = data.trace ? setupReplay(stage, aircraft, data.trace, data.sitl_passed) : null;
  frameCamera(stage, aircraft);
  // Read-only handle for debugging and the browser check: the aircraft object and a seek(t) for replays.
  window.viewer = Object.freeze({ aircraft: aircraft.group, seek: replay ? replay.seek : null });

  const cgLabel = document.getElementById('cg-label');
  let last = performance.now();
  stage.renderer.setAnimationLoop((now) => {
    const dt = Math.min((now - last) / 1000, MAX_FRAME_DT_S);
    last = now;
    if (replay) replay.tick(dt);
    stage.controls.update();
    stage.renderer.render(stage.scene, stage.camera);
    placeCgLabel(cgLabel, aircraft.group, stage.camera, stage.renderer.domElement);
  });
}

try {
  main();
} catch (error) {
  console.error('viewer failed to start', error);
  const box = document.getElementById('error');
  box.textContent = `The 3D viewer could not start: ${error.message}`;
  box.hidden = false;
}
