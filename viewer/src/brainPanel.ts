import * as THREE from "three";

export interface BrainData {
  nNeurons: number;
  positions: Float32Array; // [N,2] layout_x, layout_y (already normalized, roughly [-4,4])
  groupIds: Uint8Array; // [N] index into groupColors
  groupColors: THREE.Color[];
}

export async function loadBrainData(baseUrl = "/data"): Promise<BrainData> {
  const [meta, xyBuf, groupBuf] = await Promise.all([
    fetch(`${baseUrl}/meta.json`).then((r) => r.json()),
    fetch(`${baseUrl}/soma_xy.bin`).then((r) => r.arrayBuffer()),
    fetch(`${baseUrl}/group_ids.bin`).then((r) => r.arrayBuffer()),
  ]);

  const positions = new Float32Array(xyBuf);
  const groupIds = new Uint8Array(groupBuf);
  const groupColors: THREE.Color[] = meta.group_colors.map((hex: string) => new THREE.Color(hex));

  return { nNeurons: meta.n_neurons, positions, groupIds, groupColors, ...meta };
}

const VERTEX_SHADER = `
  attribute float heat;
  attribute vec3 baseColor;
  uniform float uPixelRatio;
  uniform float uBaseSize;
  varying float vHeat;
  varying vec3 vColor;
  void main() {
    vHeat = heat;
    vColor = baseColor;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = uPixelRatio * (uBaseSize + heat * 6.0);
  }
`;

// Halo pass: the points drawn again, larger and faint, only where a neuron is active
// (bloom without post-processing, so it works with per-panel scissor viewports).
const HALO_VERTEX_SHADER = `
  attribute float heat;
  attribute vec3 baseColor;
  uniform float uPixelRatio;
  varying float vHeat;
  varying vec3 vColor;
  void main() {
    vHeat = heat;
    vColor = baseColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = heat > 0.06 ? uPixelRatio * (4.0 + heat * 22.0) : 0.0;
  }
`;

const HALO_FRAGMENT_SHADER = `
  varying float vHeat;
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord.xy - 0.5);
    if (d > 0.5) discard;
    float f = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vColor * 1.2, f * f * vHeat * 0.16);
  }
`;

const FRAGMENT_SHADER = `
  uniform float uRestAlpha;
  varying float vHeat;
  varying vec3 vColor;
  void main() {
    vec2 uv = gl_PointCoord.xy - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float falloff = smoothstep(0.5, 0.0, d);
    vec3 baseGrey = vec3(0.75, 0.76, 0.8);
    vec3 color = mix(baseGrey, vColor, clamp(vHeat * 1.4, 0.0, 1.0));
    // Resting points are faint so that, with additive blending of ~165k points, active neurons
    // still stand out; uRestAlpha scales with panel area (see render()).
    float alpha = falloff * (uRestAlpha + vHeat * 0.9);
    gl_FragColor = vec4(color * (0.35 + vHeat * 1.3), alpha);
  }
`;

/** One fly's brain panel: shared point geometry, plus its own per-neuron `heat` buffer that drives the glow. */
export class BrainPanel {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  points: THREE.Points;
  heat: Float32Array;
  heatAttr: THREE.BufferAttribute;
  decayPerMs: number;

  constructor(
    private data: BrainData,
    sharedPositionAttr: THREE.BufferAttribute,
    sharedColorAttr: THREE.BufferAttribute,
    decayTauMs = 300,
  ) {
    this.decayPerMs = 1 / decayTauMs;
    this.heat = new Float32Array(data.nNeurons);
    this.heatAttr = new THREE.BufferAttribute(this.heat, 1);
    this.heatAttr.setUsage(THREE.DynamicDrawUsage);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", sharedPositionAttr);
    geometry.setAttribute("baseColor", sharedColorAttr);
    geometry.setAttribute("heat", this.heatAttr);

    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
        uBaseSize: { value: 1.6 },
        uRestAlpha: { value: 0.035 },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, material);
    const halo = new THREE.Points(geometry, new THREE.ShaderMaterial({
      vertexShader: HALO_VERTEX_SHADER,
      fragmentShader: HALO_FRAGMENT_SHADER,
      uniforms: { uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) } },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    this.scene = new THREE.Scene();
    this.scene.add(halo);
    this.scene.add(this.points);

    // Layout coords are roughly N(0,1)-normalized; frame with a fixed orthographic box.
    const half = 3.5;
    this.camera = new THREE.OrthographicCamera(-half, half, half * 1.3, -half * 1.3, 0.1, 10);
    this.camera.position.z = 1;
  }

  /** Bump heat for the given neuron indices (this frame's spikes), then decay all. */
  update(spikeIndices: Uint32Array | number[], dtMs: number) {
    const decay = Math.exp(-dtMs * this.decayPerMs);
    for (let i = 0; i < this.heat.length; i++) {
      this.heat[i] *= decay;
    }
    for (let i = 0; i < spikeIndices.length; i++) {
      const idx = spikeIndices[i];
      this.heat[idx] = Math.min(1.0, this.heat[idx] + 0.9);
    }
    this.heatAttr.needsUpdate = true;
  }

  /** Decay all heat by elapsed time (call once per frame). */
  decay(dtMs: number) {
    const k = Math.exp(-dtMs * this.decayPerMs);
    for (let i = 0; i < this.heat.length; i++) this.heat[i] *= k;
    this.heatAttr.needsUpdate = true;
  }

  /** Add heat for neurons that spiked (may be called several times per frame). */
  bump(spikeIndices: Uint32Array | number[]) {
    for (let i = 0; i < spikeIndices.length; i++) {
      const idx = spikeIndices[i];
      if (idx < this.heat.length) this.heat[idx] = Math.min(1.0, this.heat[idx] + 0.9);
    }
    this.heatAttr.needsUpdate = true;
  }

  clear() {
    this.heat.fill(0);
    this.heatAttr.needsUpdate = true;
  }

  /** x, y, w, h in CSS pixels with y measured from the TOP of the canvas. */
  render(renderer: THREE.WebGLRenderer, x: number, y: number, w: number, h: number) {
    if (w < 2 || h < 2) return;
    // WebGL viewports are measured from the bottom edge; flip so layout rects line up with
    // overlays. three.js applies the pixel ratio itself, so these stay in CSS pixels.
    const yGl = renderer.domElement.clientHeight - y - h;
    // resting alpha scales with the square root of panel area, so small panels stay visible
    const area = Math.min(w, h * 0.77) * h;
    (this.points.material as THREE.ShaderMaterial).uniforms.uRestAlpha.value =
      Math.min(0.035, Math.max(0.006, 0.035 * Math.sqrt(area / 560000)));
    renderer.setViewport(x, yGl, w, h);
    renderer.setScissor(x, yGl, w, h);
    renderer.setScissorTest(true);
    // orthographic: adjust left/right to preserve aspect without distortion
    const half = 3.5;
    if (w / h > 1) {
      this.camera.top = half * 1.3;
      this.camera.bottom = -half * 1.3;
      this.camera.left = -half * (w / h) * 1.0;
      this.camera.right = half * (w / h) * 1.0;
    } else {
      this.camera.left = -half;
      this.camera.right = half;
      this.camera.top = half * (h / w);
      this.camera.bottom = -half * (h / w);
    }
    this.camera.updateProjectionMatrix();
    renderer.render(this.scene, this.camera);
  }
}

export function buildSharedAttributes(data: BrainData) {
  // ShaderMaterial declares `attribute vec3 position`, so pad the stored [x,y] pairs to [x,y,0].
  const positions3 = new Float32Array(data.nNeurons * 3);
  for (let i = 0; i < data.nNeurons; i++) {
    positions3[i * 3] = data.positions[i * 2];
    positions3[i * 3 + 1] = data.positions[i * 2 + 1];
    positions3[i * 3 + 2] = 0;
  }
  const positionAttr = new THREE.BufferAttribute(positions3, 3);
  const colorArr = new Float32Array(data.nNeurons * 3);
  for (let i = 0; i < data.nNeurons; i++) {
    const c = data.groupColors[data.groupIds[i]];
    colorArr[i * 3] = c.r;
    colorArr[i * 3 + 1] = c.g;
    colorArr[i * 3 + 2] = c.b;
  }
  const colorAttr = new THREE.BufferAttribute(colorArr, 3);
  return { positionAttr, colorAttr };
}
