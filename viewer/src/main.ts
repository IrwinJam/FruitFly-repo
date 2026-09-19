import * as THREE from "three";
import { BrainPanel, buildSharedAttributes, loadBrainData, type BrainData } from "./brainPanel";
import { computeLayout, computeMatchLayout, type PanelRect } from "./layout";
import { CircuitMeter } from "./circuits";
import { AGENT_COLORS, MapView } from "./mapView";
import { Overlay } from "./overlay";
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
  const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true, preserveDrawingBuffer: true });
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
  const params = new URLSearchParams(location.search);
  const layoutN = (data as unknown as { n_neurons: number }).n_neurons ?? data.nNeurons;

  // optional per-replay stats lines for the end card (viewer/public/replays/showcase_stats.json)
  let statLines: string[] = [];
  try {
    const r = await fetch("/replays/showcase_stats.json");
    if (r.ok) {
      const s = await r.json();
      const key = Object.keys(s).find((k) => name.startsWith(k));
      if (key) statLines = s[key];
    }
  } catch { /* optional */ }

  const mapCanvas = el("canvas", { position: "absolute", pointerEvents: "none" }, app);
  const map = new MapView(mapCanvas, rp);
  const ovCanvas = el("canvas", { position: "absolute", left: "0", top: "0", pointerEvents: "none" }, app);
  const ov = new Overlay(ovCanvas);

  const panels = new Map<number, BrainPanel>();
  const meters = new Map<number, CircuitMeter>();
  let layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
  for (const p of layout.panels) {
    panels.set(p.agent, new BrainPanel(data, positionAttr, colorAttr));
    meters.set(p.agent, new CircuitMeter(data));
  }

  // ---- controls (HTML; deliberately not part of recordings)
  const bar = el("div", {
    position: "absolute", left: "0", right: "0", bottom: "0", height: `${CONTROL_H}px`, background: "#0b0f10",
    borderTop: "1px solid #1f2a2e", display: "flex", flexDirection: "column", justifyContent: "center", gap: "4px",
    padding: "0 12px", boxSizing: "border-box", fontFamily: "system-ui, sans-serif", color: "#cbd5e1", fontSize: "12px",
  }, app);
  const row = el("div", { display: "flex", alignItems: "center", gap: "10px" }, bar);
  const btn = { background: "#1f2a2e", color: "#e5e7eb", border: "1px solid #334155", borderRadius: "4px", padding: "3px 10px", cursor: "pointer" };
  const play = el("button", btn, row);
  play.id = "play-toggle";
  const speed = el("select", { background: "#1f2a2e", color: "#e5e7eb", border: "1px solid #334155", borderRadius: "4px" }, row);
  speed.id = "speed";
  const startSpeed = Number(params.get("speed") ?? 1);
  for (const s of [0.5, 1, 2, 4, 8]) {
    const o = el("option", {}, speed);
    o.value = String(s);
    o.textContent = `${s}x`;
    if (s === startSpeed) o.selected = true;
  }
  const scrub = el("input", { flex: "1" }, row);
  scrub.id = "scrub";
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(m.n_ticks - 1);
  scrub.value = "0";
  const rec = el("button", { ...btn, color: "#fca5a5" }, row);
  rec.id = "record";
  rec.textContent = "● Record";
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

  const tickS = m.tick_ms / 1000;
  let playing = params.get("paused") === null;
  let tickF = Math.min(m.n_ticks - 1, Number(params.get("start") ?? 0) / tickS);
  const endTick = params.get("end") ? Math.min(m.n_ticks - 1, Number(params.get("end")) / tickS) : m.n_ticks - 1;
  let lastTick = Math.floor(tickF) - 1;
  const showTitle = params.get("title") !== "0" && tickF === 0;
  let titleT = 0; // ms since start, for the title card
  let endCardMs = 0;
  let holdMs = 0; // time spent parked at the end tick (recording auto-stop)
  let selected: number | null = null;
  play.textContent = playing ? "Pause" : "Play";
  play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
  const resetActivity = () => { for (const p of panels.values()) p.clear(); for (const mt of meters.values()) mt.reset(); };
  scrub.oninput = () => { tickF = Number(scrub.value); resetActivity(); lastTick = tickF - 1; endCardMs = 0; };
  window.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); play.click(); } });
  app.addEventListener("click", (e) => {
    const hit = layout.panels.find((p) => e.clientX >= p.x && e.clientX < p.x + p.w && e.clientY >= p.y && e.clientY < p.y + p.h);
    selected = hit ? (selected === hit.agent ? null : hit.agent) : selected;
    map.selected = selected;
  });

  // ---- recorder: composite WebGL + map + overlay into one canvas, save as WebM
  const comp = document.createElement("canvas");
  const compCtx = comp.getContext("2d")!;
  let recorder: MediaRecorder | null = null;
  let chunks: Blob[] = [];
  const startRec = () => {
    comp.width = glCanvas.width;
    comp.height = glCanvas.height - Math.round(CONTROL_H * Math.min(window.devicePixelRatio, 2));
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
    recorder = new MediaRecorder(comp.captureStream(30), { mimeType: mime, videoBitsPerSecond: 10_000_000 });
    chunks = [];
    recorder.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
    recorder.onstop = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      a.download = `${name}.webm`;
      a.click();
    };
    recorder.start(1000);
    rec.textContent = "■ Stop";
  };
  const stopRec = () => { recorder?.stop(); recorder = null; rec.textContent = "● Record"; };
  rec.onclick = () => (recorder ? stopRec() : startRec());
  if (params.get("rec") === "1") startRec();

  function relayout() {
    layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    map.resize(layout.map.x, layout.map.y, layout.map.w, layout.map.h);
    ov.resize(window.innerWidth, window.innerHeight - CONTROL_H);
  }
  window.addEventListener("resize", relayout);
  relayout();

  const brainSet = new Set(m.brain_agents);
  const agentName = (a: number) => (m.roles[a] === "seeker" ? "Seeker" : `Hider ${a}`);
  const over = m.events.find((e) => e.kind === "over");
  let prev = performance.now();
  function frame(now: number) {
    const dt = now - prev;
    prev = now;
    titleT += dt;
    const titleOn = showTitle && titleT < 4200;
    if (playing && !titleOn) tickF = Math.min(endTick, tickF + (dt / m.tick_ms) * Number(speed.value));
    const tick = Math.floor(tickF);

    for (const p of panels.values()) p.decay(dt);
    if (tick !== lastTick) {
      const from = tick > lastTick && tick - lastTick < 50 ? lastTick + 1 : tick;
      for (let t = from; t <= tick; t++) {
        for (const a of brainSet) {
          const sp = rp.spikes(t, a);
          panels.get(a)?.bump(sp);
          meters.get(a)?.addTick(sp, m.tick_ms);
        }
      }
      lastTick = tick;
      scrub.value = String(tick);
    }

    renderer.setScissorTest(false);
    renderer.clear();
    ov.clear();
    const sx = rp.field(tick, 0, "x");
    const sy = rp.field(tick, 0, "y");
    for (const p of layout.panels) {
      panels.get(p.agent)!.render(renderer, p.x, p.y, p.w, p.h);
      const alive = rp.field(tick, p.agent, "alive") > 0.5;
      const isBrain = brainSet.has(p.agent);
      const d = Math.hypot(rp.field(tick, p.agent, "x") - sx, rp.field(tick, p.agent, "y") - sy);
      ov.panel(p, {
        title: p.role === "seeker" ? "SEEKER" : `HIDER ${p.agent}`,
        color: AGENT_COLORS[p.agent % AGENT_COLORS.length],
        subtitle: isBrain
          ? `${layoutN.toLocaleString()} neurons in layout · ${m.neurons_simulated.toLocaleString()} simulated (${m.graph})`
          : "scripted · no brain",
        alive, selected: selected === p.agent,
        circuits: isBrain ? meters.get(p.agent)!.levels() : null,
        danger: p.role === "hider" && alive ? Math.max(0, Math.min(1, 1 - d / m.danger_range)) : null,
        activityMs: 300,
      });
    }
    renderer.setScissorTest(false);
    map.draw(tick);
    ov.eventFeed(layout.map, rp, tick, agentName);

    const W = window.innerWidth;
    const H = window.innerHeight - CONTROL_H;
    if (showTitle) {
      const a = titleT < 3400 ? 1 : Math.max(0, 1 - (titleT - 3400) / 800);
      ov.card(W, H, a, [
        { text: "FlySeek", size: 40, weight: 800 },
        { text: "Among Us Hide n Seek, played by fruit-fly connectome brains", size: 18 },
        { text: `${m.brain_agents.length} flies · each one a simulated ${m.graph} brain (${m.neurons_simulated.toLocaleString()} spiking neurons) wired from the MaleCNS connectome`, size: 13, color: "#cbd5e1" },
        { text: "Connectome-constrained model with engineered senses, route planning and motor readout. Not a real fly.", size: 12, color: "#94a3b8" },
        { text: "Data: MaleCNS v1.0 (CC-BY 4.0) · fan project, not affiliated with Innersloth", size: 11, color: "#64748b" },
      ]);
    }
    if (over && tick >= over.tick) {
      endCardMs += dt;
      const hiders = m.roles.map((r, i) => (r === "hider" ? i : -1)).filter((i) => i >= 0);
      const alive = hiders.filter((i) => rp.field(tick, i, "alive") > 0.5).length;
      const t = over.tick * tickS;
      const head = over.winner === "seeker"
        ? `Seeker wins · all ${hiders.length} hiders caught in ${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`
        : `Hiders win · ${alive} of ${hiders.length} survived`;
      ov.card(W, H, Math.min(1, Math.max(0, (endCardMs - 800) / 1000)), [
        { text: head, size: 26, weight: 700, color: over.winner === "seeker" ? "#fca5a5" : "#86efac" },
        ...statLines.map((s) => ({ text: s, size: 13, color: "#cbd5e1" })),
        { text: `Seed ${m.seed} · ${m.preset} rules · replay ${name}`, size: 11, color: "#64748b" },
      ]);
    }

    if (recorder) {
      const dpr = Math.min(window.devicePixelRatio, 2);
      compCtx.fillStyle = "#000";
      compCtx.fillRect(0, 0, comp.width, comp.height);
      compCtx.drawImage(glCanvas, 0, 0);
      compCtx.drawImage(mapCanvas, Math.round(layout.map.x * dpr), Math.round(layout.map.y * dpr));
      compCtx.drawImage(ovCanvas, 0, 0);
      holdMs = tick >= endTick ? holdMs + dt : 0;
      if (holdMs > (over && tick >= over.tick ? 4500 : 800)) stopRec();
    }
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
