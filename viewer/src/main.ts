import * as THREE from "three";
import { BrainPanel, buildSharedAttributes, loadBrainData, type BrainData } from "./brainPanel";
import { computeLayout, computeMatchLayout, panelAreas, type PanelRect } from "./layout";
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

const sliderCss = document.createElement("style");
sliderCss.textContent = `
  #scrub { -webkit-appearance: none; appearance: none; height: 2px; background: #2a2e35; border-radius: 1px; outline: none; }
  #scrub::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 50%; background: #c9ccd2; cursor: pointer; }
  #scrub::-moz-range-thumb { width: 10px; height: 10px; border: none; border-radius: 50%; background: #c9ccd2; cursor: pointer; }
  button:hover, select:hover { border-color: #3a3f47 !important; }
`;
document.head.appendChild(sliderCss);

const glCanvas = el("canvas", { position: "absolute", inset: "0" }, app);
const overlay = el("div", { position: "absolute", inset: "0", pointerEvents: "none", fontFamily: "system-ui, sans-serif", color: "#e5e7eb" }, app);

/** Next animation frame; a ~30 fps timer while the page is hidden, so replays and recordings keep going in the background. */
function nextFrame(cb: (now: number) => void) {
  // whichever fires first wins: a frame queued while visible never fires if the page is hidden meanwhile
  let fired = false;
  const run = (now: number) => { if (!fired) { fired = true; cb(now); } };
  requestAnimationFrame(run);
  setTimeout(() => run(performance.now()), document.hidden ? 33 : 100);
}

// ------------------------------------------------------------------ recording (shared across reel segments)
/**
 * Hand a finished still or recording to the user: a browser download by default, or with ?save=server
 * a POST to the dev server, which writes it into the project (viewer/vite.config.ts).
 */
function deliver(blob: Blob, fileName: string, kind: "still" | "video") {
  if (new URLSearchParams(location.search).get("save") === "server") {
    fetch(`/__save/${kind}/${encodeURIComponent(fileName)}`, { method: "POST", body: blob })
      .then((r) => r.text()).then((t) => console.info(t, blob.size))
      .catch((e) => console.error("save failed", e));
    return;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;
  a.click();
}

/** Composites the WebGL canvas, the map and the overlay into one canvas and records it as WebM. */
class Compositor {
  comp = document.createElement("canvas");
  ctx = this.comp.getContext("2d")!;
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  get active() { return this.rec !== null; }
  start(fileName: string) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.comp.width = glCanvas.width;
    this.comp.height = glCanvas.height - Math.round(CONTROL_H * dpr);
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
    this.rec = new MediaRecorder(this.comp.captureStream(30), { mimeType: mime, videoBitsPerSecond: 10_000_000 });
    this.chunks = [];
    this.rec.ondataavailable = (ev) => { if (ev.data.size) this.chunks.push(ev.data); };
    this.rec.onstop = () => deliver(new Blob(this.chunks, { type: "video/webm" }), `${fileName}.webm`, "video");
    this.rec.start(1000);
  }
  stop() { this.rec?.stop(); this.rec = null; }
  /** Save the current composite as a PNG. */
  still(fileName: string, mapCanvas: HTMLCanvasElement, mapX: number, mapY: number, ovCanvas: HTMLCanvasElement) {
    if (!this.rec) {
      const dpr = Math.min(window.devicePixelRatio, 2);
      this.comp.width = glCanvas.width;
      this.comp.height = glCanvas.height - Math.round(CONTROL_H * dpr);
    }
    this.draw(mapCanvas, mapX, mapY, ovCanvas);
    this.comp.toBlob((b) => { if (b) deliver(b, `${fileName}.png`, "still"); }, "image/png");
  }
  draw(mapCanvas: HTMLCanvasElement, mapX: number, mapY: number, ovCanvas: HTMLCanvasElement) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.comp.width, this.comp.height);
    this.ctx.drawImage(glCanvas, 0, 0);
    this.ctx.drawImage(mapCanvas, Math.round(mapX * dpr), Math.round(mapY * dpr));
    this.ctx.drawImage(ovCanvas, 0, 0);
  }
}

/** One clip of a reel: a time window of one replay, with an optional caption. Times are match seconds. */
interface Segment {
  replay: string;
  start?: number;
  end?: number;
  speed?: number;
  caption?: string[];
}

interface SegmentOpts extends Partial<Segment> {
  replayObj?: Replay;
  recorder?: Compositor;
  reel?: boolean;
  keep?: boolean; // last reel segment: keep rendering its end card after the reel is done
  onStart?: () => void;
}

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
  const q = new URLSearchParams(location.search);
  const reelUrl = q.get("reel");
  if (reelUrl) {
    await runReel(data, positionAttr, colorAttr, renderer, reelUrl, index, q.get("rec") === "1");
    return;
  }
  const requested = q.get("replay");
  const name = requested ?? index[0]?.name;
  if (name) {
    await runReplay(data, positionAttr, colorAttr, renderer, name, index);
  } else {
    runDemo(data, positionAttr, colorAttr, renderer);
  }
}

// ------------------------------------------------------------------ reel mode
/**
 * Plays a list of clips from several replays back to back and records them as one WebM.
 * The reel file is written by amongusfly.experiments.plan_reel; ?reel=/replays/reel.json&rec=1 records it.
 */
async function runReel(
  data: BrainData, positionAttr: THREE.BufferAttribute, colorAttr: THREE.BufferAttribute,
  renderer: THREE.WebGLRenderer, reelUrl: string, index: ReplayIndexEntry[], record: boolean,
) {
  const reel: { name: string; segments: Segment[] } = await (await fetch(reelUrl)).json();
  const names = [...new Set(reel.segments.map((s) => s.replay))];
  const loaded = new Map(await Promise.all(names.map(async (n) => [n, await Replay.load(n)] as const)));
  const recorder = new Compositor();
  let started = false;
  for (const [i, seg] of reel.segments.entries()) {
    console.info(`reel ${i + 1}/${reel.segments.length}: ${seg.replay} ${seg.start ?? 0}-${seg.end ?? "end"} s`, performance.now() | 0);
    await runReplay(data, positionAttr, colorAttr, renderer, seg.replay, index, {
      ...seg, replayObj: loaded.get(seg.replay), recorder, reel: true, keep: i === reel.segments.length - 1,
      onStart: () => { if (record && !started) { recorder.start(reel.name); started = true; } },
    });
  }
  if (started) recorder.stop();
  console.info("reel done", performance.now() | 0);
}

// ------------------------------------------------------------------ replay mode
async function runReplay(
  data: BrainData, positionAttr: THREE.BufferAttribute, colorAttr: THREE.BufferAttribute,
  renderer: THREE.WebGLRenderer, name: string, index: ReplayIndexEntry[], opts: SegmentOpts = {},
): Promise<void> {
  const rp = opts.replayObj ?? await Replay.load(name);
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
  if (opts.reel) map.showResult = false;
  const ovCanvas = el("canvas", { position: "absolute", left: "0", top: "0", pointerEvents: "none" }, app);
  const ov = new Overlay(ovCanvas);

  const panels = new Map<number, BrainPanel>();
  const meters = new Map<number, CircuitMeter>();
  let layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
  for (const p of layout.panels) {
    panels.set(p.agent, new BrainPanel(data, positionAttr, colorAttr));
    meters.set(p.agent, new CircuitMeter(data));
  }

  // ---- controls (HTML, not recorded)
  const bar = el("div", {
    position: "absolute", left: "0", right: "0", bottom: "0", height: `${CONTROL_H}px`, background: "#07080a",
    borderTop: "1px solid #16191e", display: "flex", flexDirection: "column", justifyContent: "center", gap: "4px",
    padding: "0 12px", boxSizing: "border-box", fontFamily: "system-ui, sans-serif", color: "#cbd5e1", fontSize: "12px",
  }, app);
  const row = el("div", { display: "flex", alignItems: "center", gap: "10px" }, bar);
  const btn = { background: "transparent", color: "#c9ccd2", border: "1px solid #2a2e35", borderRadius: "3px", padding: "3px 12px",
    cursor: "pointer", font: "400 12px system-ui, -apple-system, 'Segoe UI', sans-serif" };
  const play = el("button", btn, row);
  play.id = "play-toggle";
  const speed = el("select", { background: "#07080a", color: "#c9ccd2", border: "1px solid #2a2e35", borderRadius: "3px" }, row);
  speed.id = "speed";
  const startSpeed = opts.speed ?? Number(params.get("speed") ?? 1);
  for (const s of [0.5, 1, 2, 4, 8]) {
    const o = el("option", {}, speed);
    o.value = String(s);
    o.textContent = `${s}x`;
    if (s === startSpeed) o.selected = true;
  }
  // seek bar with clickable event markers (catches, vents, pings, phase changes)
  const seekWrap = el("div", { flex: "1", position: "relative", height: "28px", display: "flex", alignItems: "center" }, row);
  const scrub = el("input", { width: "100%", position: "relative", zIndex: "1" }, seekWrap);
  scrub.id = "scrub";
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(m.n_ticks - 1);
  scrub.value = "0";
  const markerColor: Record<string, string> = { kill: "#c9899f", vent_enter: "#5d626b", ping: "#cbb173", phase: "#9a9fa8" };
  const markerLabel = (e: { kind: string; [k: string]: any }) =>
    e.kind === "kill" ? `Hider ${e.victim} caught` : e.kind === "vent_enter" ? `Hider ${e.agent} vented`
      : e.kind === "ping" ? "Ping: hiders revealed" : e.phase === "seek" ? "Seeker released" : e.phase === "final_hide" ? "Final Hide" : e.phase;
  for (const e of m.events) {
    if (!(e.kind in markerColor) || (e.kind === "phase" && e.phase === "hide")) continue;
    const mk = el("div", {
      position: "absolute", top: e.kind === "kill" ? "0" : "3px", width: e.kind === "kill" ? "4px" : "2px",
      height: e.kind === "kill" ? "8px" : "5px", marginLeft: "-1px", borderRadius: "1px", cursor: "pointer", zIndex: "2",
      left: `calc(8px + (100% - 16px) * ${e.tick / Math.max(1, m.n_ticks - 1)})`, background: markerColor[e.kind],
    }, seekWrap);
    mk.title = `${Math.floor((e.tick * m.tick_ms) / 60000)}:${Math.floor(((e.tick * m.tick_ms) / 1000) % 60).toString().padStart(2, "0")} ${markerLabel(e)}`;
    mk.onclick = (ev) => { ev.stopPropagation(); jumpTo(Math.max(0, e.tick - Math.round(3000 / m.tick_ms))); };
  }
  const rec = el("button", btn, row);
  rec.id = "record";
  rec.textContent = "Record";
  const picker = el("select", { background: "#07080a", color: "#c9ccd2", border: "1px solid #2a2e35", borderRadius: "3px", maxWidth: "260px" }, row);
  picker.id = "replay-picker";
  for (const e of index.length ? index : [{ name, brain_agents: m.brain_agents, graph: m.graph, seconds: 0, winner: null }]) {
    const o = el("option", {}, picker);
    o.value = e.name;
    o.textContent = `${e.name} (${e.brain_agents.length ? `${e.brain_agents.length} brains, ${e.graph}` : "scripted only"}${e.winner ? `, ${e.winner} won` : ""})`;
    if (e.name === name) o.selected = true;
  }
  picker.onchange = () => { location.search = `?replay=${encodeURIComponent(picker.value)}`; };
  const note = el("div", { color: "#6b7079", fontSize: "11px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", cursor: "help" }, bar);
  note.textContent = "Connectome-constrained model, not a real fly. Glow: neurons that just spiked; bars: mean firing rate of each pathway. S saves a still. Hover for what is engineered.";
  note.title = `${m.disclaimer}

Engineered: ${m.engineered.join("; ")}.${m.brain_agents.length && !m.odor_channels_enabled ? " Odor channels off on this graph." : ""}`;

  const tickS = m.tick_ms / 1000;
  let playing = opts.reel || params.get("paused") === null;
  const startS = opts.start ?? Number(params.get("start") ?? 0);
  const endS = opts.end ?? (params.get("end") ? Number(params.get("end")) : null);
  let tickF = Math.min(m.n_ticks - 1, startS / tickS);
  const endTick = endS !== null ? Math.min(m.n_ticks - 1, endS / tickS) : m.n_ticks - 1;
  let lastTick = Math.floor(tickF) - 1;
  let segMs = 0; // ms since this segment started (reel fades and captions)
  let stopped = false;
  let finish: () => void = () => {};
  const done = new Promise<void>((res) => { finish = res; });
  let endCardMs = 0;
  let holdMs = 0; // time spent parked at the end tick (recording auto-stop)
  let selected: number | null = null;
  play.textContent = playing ? "Pause" : "Play";
  play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
  const resetActivity = () => { for (const p of panels.values()) p.clear(); for (const mt of meters.values()) mt.reset(); };
  const jumpTo = (t: number) => { tickF = t; scrub.value = String(t); resetActivity(); lastTick = t - 1; endCardMs = 0; };
  scrub.oninput = () => jumpTo(Number(scrub.value));
  // S saves a PNG of the current frame; ?start=T-3&end=T&still=1 plays into T (so the glow builds up) and saves it there
  const stillName = params.get("still"); // "1" or a file name, e.g. fig6a_hide
  let stillAtEnd = !opts.reel && stillName !== null;
  let stillQueued = false;
  const onKey = (e: KeyboardEvent) => {
    if (e.code === "Space") { e.preventDefault(); play.click(); }
    if (e.code === "KeyS") stillQueued = true;
  };
  window.addEventListener("keydown", onKey);
  const onClick = (e: MouseEvent) => {
    const hit = layout.panels.find((p) => e.clientX >= p.x && e.clientX < p.x + p.w && e.clientY >= p.y && e.clientY < p.y + p.h);
    selected = hit ? (selected === hit.agent ? null : hit.agent) : selected;
    map.selected = selected;
  };
  app.addEventListener("click", onClick);

  // ---- recorder: composite WebGL + map + overlay into one canvas, save as WebM
  const recorder = opts.recorder ?? new Compositor();
  const startRec = () => { recorder.start(name); rec.textContent = "Stop recording"; rec.style.color = "#d99a8a"; };
  const stopRec = () => { recorder.stop(); rec.textContent = "Record"; rec.style.color = "#c9ccd2"; };
  rec.onclick = () => (recorder.active ? stopRec() : startRec());
  if (opts.reel) rec.disabled = true;
  if (!opts.reel && params.get("rec") === "1") startRec();

  function relayout() {
    layout = computeMatchLayout(window.innerWidth, window.innerHeight - CONTROL_H, nHiders);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    map.resize(layout.map.x, layout.map.y, layout.map.w, layout.map.h);
    ov.resize(window.innerWidth, window.innerHeight - CONTROL_H);
  }
  window.addEventListener("resize", relayout);
  relayout();
  opts.onStart?.();

  const brainSet = new Set(m.brain_agents);
  const agentName = (a: number) => (m.roles[a] === "seeker" ? "Seeker" : `Hider ${a}`);
  const clock = (t: number) => `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`;
  const over = m.events.find((e) => e.kind === "over");
  // the "over" event is logged one tick after the last recorded frame, so clamp it to a frame that exists
  const overTick = over ? Math.min(over.tick, m.n_ticks - 1) : Infinity;
  let prev = performance.now();
  function frame(now: number) {
    if (stopped) return;
    const dt = now - prev;
    prev = now;
    segMs += dt;
    if (playing) tickF = Math.min(endTick, tickF + (dt / m.tick_ms) * Number(speed.value));
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
      const b = panelAreas(p).brain;
      panels.get(p.agent)!.render(renderer, b.x, b.y, b.w, b.h);
      const alive = rp.alive(tick, p.agent);
      const isBrain = brainSet.has(p.agent);
      const d = Math.hypot(rp.field(tick, p.agent, "x") - sx, rp.field(tick, p.agent, "y") - sy);
      const kill = m.events.find((e) => e.kind === "kill" && e.victim === p.agent && (e.tick <= tick || tick >= m.n_ticks - 1));
      ov.panel(p, {
        title: p.role === "seeker" ? "Seeker" : `Hider ${p.agent}`,
        color: AGENT_COLORS[p.agent % AGENT_COLORS.length],
        subtitle: isBrain
          ? `${layoutN.toLocaleString()} neurons · ${m.neurons_simulated.toLocaleString()} simulated`
          : "scripted, no brain",
        alive, caughtAt: kill ? clock(kill.tick * tickS) : null, selected: selected === p.agent,
        circuits: isBrain ? meters.get(p.agent)!.levels() : null,
        danger: p.role === "hider" && alive ? Math.max(0, Math.min(1, 1 - d / m.danger_range)) : null,
      });
    }
    renderer.setScissorTest(false);
    map.draw(tick);
    ov.eventFeed(layout.map, rp, tick, agentName);

    const W = window.innerWidth;
    const H = window.innerHeight - CONTROL_H;
    ov.brand(layout.title);
    if (!opts.reel && over && tick >= overTick) {  // end card (interactive viewer only; the reel has none)
      endCardMs += dt;
      const hiders = m.roles.map((r, i) => (r === "hider" ? i : -1)).filter((i) => i >= 0);
      const alive = hiders.filter((i) => rp.alive(tick, i)).length;
      const t = over.tick * tickS;
      const head = over.winner === "seeker"
        ? `Seeker wins · all ${hiders.length} hiders caught in ${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`
        : `Hiders win · ${alive} of ${hiders.length} survived`;
      ov.card(W, H, Math.min(1, Math.max(0, (endCardMs - 800) / 1000)), [
        { text: head, size: 24, weight: 500, color: over.winner === "seeker" ? "#d98c5f" : "#79b3a2" },
        ...statLines.map((s) => ({ text: s, size: 13, color: "#cbd5e1" })),
        { text: `Seed ${m.seed} · ${m.preset} rules · replay ${name}`, size: 11, color: "#64748b" },
        { text: "Data: MaleCNS v1.0 (CC-BY 4.0) · fan project, not affiliated with Innersloth", size: 11, color: "#64748b" },
      ]);
    }

    if (opts.reel) {
      // caption in the band between the HUD and the top of the map
      const band = { x: layout.map.x, y: layout.map.y + 52, w: layout.map.w, h: Math.max(60, map.floorTop - 60) };
      if (opts.caption) ov.caption(band, opts.caption, Math.min(1, segMs / 400));
      const fadeIn = Math.max(0, 1 - segMs / 350);
      if (fadeIn > 0) ov.card(W, H, fadeIn, []);
    }

    const urlStill = stillAtEnd && tick >= endTick && holdMs > 300;
    if (stillQueued || urlStill) {
      const auto = `${name}_${clock(tick * tickS).replace(":", "m")}s`;
      recorder.still(urlStill && stillName !== "1" ? stillName! : auto, mapCanvas, layout.map.x, layout.map.y, ovCanvas);
      stillQueued = false;
      if (urlStill) stillAtEnd = false;
    }
    holdMs = tick >= endTick ? holdMs + dt : 0;
    const holdLimit = tick >= overTick ? (opts.reel ? 1500 : 4500) : 800;
    if (recorder.active) recorder.draw(mapCanvas, layout.map.x, layout.map.y, ovCanvas);
    if (holdMs > holdLimit) {
      if (opts.keep) finish(); // last reel segment: stay parked on the end card
      else if (opts.reel) { cleanup(); return; }
      if (recorder.active) stopRec();
    }
    nextFrame(frame);
  }
  function cleanup() {
    stopped = true;
    window.removeEventListener("resize", relayout);
    window.removeEventListener("keydown", onKey);
    app.removeEventListener("click", onClick);
    mapCanvas.remove();
    ovCanvas.remove();
    bar.remove();
    finish();
  }
  nextFrame(frame);
  return opts.reel ? done : Promise.resolve();
}

// ------------------------------------------------------------------ demo mode (no replays)
function runDemo(data: BrainData, positionAttr: THREE.BufferAttribute, colorAttr: THREE.BufferAttribute, renderer: THREE.WebGLRenderer) {
  const nHiders = 5;
  const note = el("div", { position: "absolute", bottom: "0", left: "0", right: "0", padding: "4px 10px", fontSize: "11px", color: "#fca5a5", background: "rgba(0,0,0,.65)" }, overlay);
  note.textContent = "DEMO MODE: synthetic activity. Record and export a match (amongusfly.world.match, amongusfly.world.export_replay) to see real spikes.";
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
    nextFrame(frame);
  }
  nextFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f87171;padding:20px;font-family:monospace">Viewer failed to start:\n${err}\n\nIf brain data is missing, run: python -m amongusfly.connectome.export_viewer_data</pre>`;
  console.error(err);
});
