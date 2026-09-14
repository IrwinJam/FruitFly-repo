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

const FRAGMENT_SHADER = `
  varying float vHeat;
  varying vec3 vColor;
  void main() {
    vec2 uv = gl_PointCoord.xy - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float falloff = smoothstep(0.5, 0.0, d);
    vec3 baseGrey = vec3(0.75, 0.76, 0.8);
    vec3 color = mix(baseGrey, vColor, clamp(vHeat * 1.4, 0.0, 1.0));
    // Low resting alpha matters a lot here: with additive blending and ~165k
    // overlapping points, even a modest per-point alpha saturates to solid white
    // once dozens of points overlap in a screen pixel. Keep resting points faint
    // so colored activity glows still read as distinct against the point cloud.
    float alpha = falloff * (0.035 + vHeat * 0.9);
    gl_FragColor = vec4(color * (0.35 + vHeat * 1.3), alpha);
  }
`;

/**
 * One fly's brain panel: shares the (read-only) position/group-color geometry
 * attributes across all panels, but owns its own `heat` buffer -- the per-neuron
 * "how recently did this spike" value that drives the glow. This is what lets N
 * panels reuse one static point cloud while showing independent live activity
 * (PROJECT_PLAN.md section 4.7).
 */
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
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, material);
    this.scene = new THREE.Scene();
    this.scene.add(this.points);

    // Layout coords are roughly N(0,1)-normalized; frame with a fixed orthographic box.
    const half = 4.2;
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

  render(renderer: THREE.WebGLRenderer, x: number, y: number, w: number, h: number) {
    const pr = renderer.getPixelRatio();
    renderer.setViewport(x * pr, y * pr, w * pr, h * pr);
    renderer.setScissor(x * pr, y * pr, w * pr, h * pr);
    renderer.setScissorTest(true);
    // orthographic: adjust left/right to preserve aspect without distortion
    const half = 4.2;
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
  // Three.js's ShaderMaterial always auto-declares `attribute vec3 position` --
  // expand our on-disk [x,y] pairs to [x,y,0] so the bound buffer matches.
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
