import * as THREE from "three";
import { BrainPanel, buildSharedAttributes, loadBrainData, type BrainData } from "./brainPanel";
import { computeLayout, computeMatchLayout, type PanelRect } from "./layout";
import { AGENT_COLORS, MapView } from "./mapView";
import { Replay, type ReplayIndexEntry } from "./replay";

const app = document.getElementById("app")!;
const CONTROL_H = 64;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, style: Partial<CSSStyleDeclaration> = {}, parent?: HTMLElement) {
  const e = document.createElement(tag);
  Object.assign(e.style, style);
  if (parent) parent.appendChild(e);
  return e;
}

const glCanvas = el("canvas", { position: "absolute", inset: "0" }, app);
const overlay = el("div", { position: "absolute", inset: "0", pointerEvents: "none", fontFamily: "system-ui, sans-serif", color: "#e5e7eb" }, app);

function panelHeader(rect: PanelRect, parent: HTMLElement) {
  const root = el("div", { position: "absolute", pointerEvents: "none" }, parent);
  const header = el("div", { position: "absolute", top: "8px", left: "10px", right: "10px", fontSize: "12px", fontWeight: "600", textShadow: "0 1px 3px #000", lineHeight: "1.35" }, root);
  return { root, header };
}

function placeRoot(root: HTMLElement, r: { x: number; y: number; w: number; h: number }) {
  Object.assign(root.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
}

async function main() {
  const data = await loadBrainData();
  const { positionAttr, colorAttr } = buildSharedAttributes(data);
  const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 1);

  let index: ReplayIndexEntry[] = [];
  try {
    const r = await fetch("/replays/index.json");
    if (r.ok) index = await r.json();
  } catch {
    /* no replays yet */
  }
  const requested = new URLSearchParams(location.search).get("replay");
  const name = requested ?? index[0]?.name;
  if (name) {
    await runReplay(data, positionAttr, colorAttr, renderer, name, index);
  } else {
    runDemo(data, positionAttr, colorAttr, renderer);
  }
}

// ------------------------------------------------------------------ replay mode
async function runReplay(
  data: BrainData, positionAttr: THREE.BufferAttribute, colorAttr: THREE.BufferAttribute,
  renderer: THREE.WebGLRenderer, name: string, index: ReplayIndexEntry[],
) {
  const rp = await Replay.load(name);
  const m = rp.meta;
  const nHiders = m.roles.filter((r) => r === "hider").length;

  const mapCanvas = el("canvas", { position: "absolute", pointerEvents: "none" }, app);
  const map = new MapView(mapCanvas, rp);

  const panels = new Map<number, BrainPanel>();
  const headers = new Map<number, ReturnType<typeof panelHeader>>();
  let layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
  for (const p of layout.panels) {
    panels.set(p.agent, new BrainPanel(data, positionAttr, colorAttr));
    headers.set(p.agent, panelHeader(p, overlay));
  }

  // ---- controls
  const bar = el("div", {
    position: "absolute", left: "0", right: "0", bottom: "0", height: `${CONTROL_H}px`, background: "#0b0f10",
    borderTop: "1px solid #1f2a2e", display: "flex", flexDirection: "column", justifyContent: "center", gap: "4px",
    padding: "0 12px", boxSizing: "border-box", fontFamily: "system-ui, sans-serif", color: "#cbd5e1", fontSize: "12px",
  }, app);
  const row = el("div", { display: "flex", alignItems: "center", gap: "10px" }, bar);
  const play = el("button", { background: "#1f2a2e", color: "#e5e7eb", border: "1px solid #334155", borderRadius: "4px", padding: "3px 10px", cursor: "pointer" }, row);
  play.id = "play-toggle";
  const speed = el("select", { background: "#1f2a2e", color: "#e5e7eb", border: "1px solid #334155", borderRadius: "4px" }, row);
  speed.id = "speed";
  for (const s of [0.5, 1, 2, 4, 8]) {
    const o = el("option", {}, speed);
    o.value = String(s);
    o.textContent = `${s}x`;
    if (s === 1) o.selected = true;
  }
  const scrub = el("input", { flex: "1" }, row);
  scrub.id = "scrub";
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(m.n_ticks - 1);
  scrub.value = "0";
  const picker = el("select", { background: "#1f2a2e", color: "#e5e7eb", border: "1px solid #334155", borderRadius: "4px", maxWidth: "260px" }, row);
  picker.id = "replay-picker";
  for (const e of index.length ? index : [{ name, brain_agents: m.brain_agents, graph: m.graph, seconds: 0, winner: null }]) {
    const o = el("option", {}, picker);
    o.value = e.name;
    o.textContent = `${e.name} (${e.brain_agents.length ? `${e.brain_agents.length} brains, ${e.graph}` : "scripted only"}${e.winner ? `, ${e.winner} won` : ""})`;
    if (e.name === name) o.selected = true;
  }
  picker.onchange = () => { location.search = `?replay=${encodeURIComponent(picker.value)}`; };
  const note = el("div", { color: "#94a3b8", fontSize: "11px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, bar);
  note.textContent = `${m.disclaimer} Engineered: ${m.engineered.join("; ")}.${m.brain_agents.length && !m.odor_channels_enabled ? " Odor channels off on this graph." : ""}`;

  let playing = true;
  let tickF = 0;
  let lastTick = -1;
  play.textContent = "Pause";
  play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
  scrub.oninput = () => { tickF = Number(scrub.value); for (const p of panels.values()) p.clear(); lastTick = tickF - 1; };
  window.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); play.click(); } });

  function relayout() {
    layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    map.resize(layout.map.x, layout.map.y, layout.map.w, layout.map.h);
    for (const p of layout.panels) placeRoot(headers.get(p.agent)!.root, p);
  }
  window.addEventListener("resize", relayout);
  relayout();

  const brainSet = new Set(m.brain_agents);
  let prev = performance.now();
  function frame(now: number) {
    const dt = now - prev;
    prev = now;
    if (playing) tickF = Math.min(m.n_ticks - 1, tickF + (dt / m.tick_ms) * Number(speed.value));
    const tick = Math.floor(tickF);

    for (const p of panels.values()) p.decay(dt);
    if (tick !== lastTick) {
      const from = tick > lastTick && tick - lastTick < 50 ? lastTick + 1 : tick;
      for (let t = from; t <= tick; t++) {
        for (const a of brainSet) panels.get(a)?.bump(rp.spikes(t, a));
      }
      lastTick = tick;
      scrub.value = String(tick);
    }

    renderer.setScissorTest(false);
    renderer.clear();
    for (const p of layout.panels) {
      panels.get(p.agent)!.render(renderer, p.x, p.y, p.w, p.h);
      const alive = rp.field(tick, p.agent, "alive") > 0.5;
      const kind = brainSet.has(p.agent) ? `brain · ${m.graph} (${m.neurons_simulated.toLocaleString()} neurons simulated)` : "scripted · no brain";
      const role = p.role === "seeker" ? "SEEKER" : `HIDER ${p.agent}`;
      headers.get(p.agent)!.header.innerHTML =
        `<span style="color:${AGENT_COLORS[p.agent % AGENT_COLORS.length]}">●</span> ${role}${alive ? "" : " <span style='color:#f87171'>caught</span>"}` +
        `<br><span style="font-weight:400;opacity:.7;font-size:11px">${kind}</span>`;
    }
    renderer.setScissorTest(false);
    map.draw(tick);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ demo mode (no replays)
function runDemo(data: BrainData, positionAttr: THREE.BufferAttribute, colorAttr: THREE.BufferAttribute, renderer: THREE.WebGLRenderer) {
  const nHiders = 5;
  const note = el("div", { position: "absolute", bottom: "0", left: "0", right: "0", padding: "4px 10px", fontSize: "11px", color: "#fca5a5", background: "rgba(0,0,0,.65)" }, overlay);
  note.textContent = "DEMO MODE: synthetic activity. Record and export a match (flyseek.world.match, flyseek.world.export_replay) to see real spikes.";
  const groupIndex: Record<string, number[]> = {};
  for (let i = 0; i < data.groupIds.length; i++) {
    const g = (data as any).group_names[data.groupIds[i]];
    (groupIndex[g] ??= []).push(i);
  }
  let rects = computeLayout(window.innerWidth, window.innerHeight - 20, nHiders);
  const panels = rects.map(() => new BrainPanel(data, positionAttr, colorAttr));
  const heads = rects.map((r) => panelHeader(r, overlay));
  const resize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    rects = computeLayout(window.innerWidth, window.innerHeight - 20, nHiders);
    rects.forEach((r, i) => {
      placeRoot(heads[i].root, r);
      heads[i].header.textContent = `${r.role === "seeker" ? "SEEKER" : "HIDER"} · demo`;
    });
  };
  window.addEventListener("resize", resize);
  resize();
  const pick = (pool: number[], n: number) => Array.from({ length: n }, () => pool[(Math.random() * pool.length) | 0]);
  let prev = performance.now();
  function frame(now: number) {
    const dt = now - prev;
    prev = now;
    renderer.setScissorTest(false);
    renderer.clear();
    rects.forEach((r, i) => {
      const env = 0.5 + 0.5 * Math.sin(now / 770 + i);
      const groups = r.role === "seeker" ? ["pursuit", "steering"] : ["looming_escape", "locomotion"];
      panels[i].decay(dt);
      panels[i].bump(pick(groupIndex["other"] ?? [], 40));
      for (const g of groups) panels[i].bump(pick(groupIndex[g] ?? [], Math.round((groupIndex[g]?.length ?? 0) * 0.15 * env)));
      panels[i].render(renderer, r.x, r.y, r.w, r.h);
    });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f87171;padding:20px;font-family:monospace">Viewer failed to start:\n${err}\n\nIf brain data is missing, run: python -m flyseek.connectome.export_viewer_data</pre>`;
  console.error(err);
});
