import * as THREE from "three";
import { BrainPanel, buildSharedAttributes, loadBrainData, type BrainData } from "./brainPanel";
import { computeLayout, type PanelRect } from "./layout";

const N_HIDERS_DEFAULT = 5;

const app = document.getElementById("app")!;

const canvas = document.createElement("canvas");
canvas.style.position = "absolute";
canvas.style.inset = "0";
app.appendChild(canvas);

const overlay = document.createElement("div");
overlay.style.position = "absolute";
overlay.style.inset = "0";
overlay.style.pointerEvents = "none";
overlay.style.fontFamily = "system-ui, -apple-system, sans-serif";
overlay.style.color = "#e5e7eb";
app.appendChild(overlay);

const disclaimer = document.createElement("div");
disclaimer.textContent =
  "DEMO MODE — synthetic activity, not connected to a live brain simulation yet. " +
  "Connectome-constrained LIF model with engineered sensory/motor mappings — not a validated fly brain.";
disclaimer.style.position = "absolute";
disclaimer.style.bottom = "0";
disclaimer.style.left = "0";
disclaimer.style.right = "0";
disclaimer.style.padding = "4px 10px";
disclaimer.style.fontSize = "11px";
disclaimer.style.color = "#fca5a5";
disclaimer.style.background = "rgba(0,0,0,0.65)";
disclaimer.style.pointerEvents = "none";
overlay.appendChild(disclaimer);

function makeHeaderEl(rect: PanelRect): { root: HTMLDivElement; header: HTMLDivElement; footer: HTMLDivElement } {
  const root = document.createElement("div");
  root.style.position = "absolute";
  root.style.pointerEvents = "none";
  overlay.appendChild(root);

  const header = document.createElement("div");
  header.style.position = "absolute";
  header.style.top = "6px";
  header.style.left = "10px";
  header.style.fontSize = rect.role === "seeker" ? "15px" : "11px";
  header.style.fontWeight = "600";
  header.style.textShadow = "0 1px 3px rgba(0,0,0,0.9)";
  root.appendChild(header);

  const footer = document.createElement("div");
  footer.style.position = "absolute";
  footer.style.bottom = rect.role === "seeker" ? "18px" : "4px";
  footer.style.left = "10px";
  footer.style.right = "10px";
  footer.style.fontSize = rect.role === "seeker" ? "11px" : "9px";
  footer.style.display = "flex";
  footer.style.gap = "6px";
  footer.style.textShadow = "0 1px 3px rgba(0,0,0,0.9)";
  root.appendChild(footer);

  return { root, header, footer };
}

function dnBar(label: string, value: number, color: string): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "2px";
  wrap.style.minWidth = "36px";

  const track = document.createElement("div");
  track.style.width = "100%";
  track.style.height = "5px";
  track.style.background = "rgba(255,255,255,0.15)";
  track.style.borderRadius = "3px";
  track.style.overflow = "hidden";

  const fill = document.createElement("div");
  fill.style.height = "100%";
  fill.style.width = `${Math.round(value * 100)}%`;
  fill.style.background = color;
  fill.style.transition = "width 80ms linear";
  track.appendChild(fill);

  const lbl = document.createElement("div");
  lbl.textContent = label;
  lbl.style.opacity = "0.75";

  wrap.appendChild(track);
  wrap.appendChild(lbl);
  (wrap as any)._fill = fill;
  return wrap;
}

async function main() {
  const data = await loadBrainData();
  const { positionAttr, colorAttr } = buildSharedAttributes(data);

  // Precompute group -> neuron index arrays for the synthetic activity driver.
  const groupIndex: Record<string, number[]> = {};
  for (let i = 0; i < data.groupIds.length; i++) {
    const g = (data as any).group_names[data.groupIds[i]];
    (groupIndex[g] ??= []).push(i);
  }

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 1);

  let nHiders = N_HIDERS_DEFAULT;
  let rects = computeLayout(window.innerWidth, window.innerHeight, nHiders);
  const panels = new Map<string, BrainPanel>();
  const headerEls = new Map<string, { root: HTMLDivElement; header: HTMLDivElement; footer: HTMLDivElement; bars: HTMLDivElement[] }>();

  function buildPanels() {
    for (const el of headerEls.values()) el.root.remove();
    headerEls.clear();
    panels.clear();

    for (const rect of rects) {
      const panel = new BrainPanel(data, positionAttr, colorAttr);
      panels.set(rect.id, panel);

      const els = makeHeaderEl(rect);
      const role = rect.role === "seeker" ? "SEEKER" : "HIDER";
      els.header.innerHTML = `${role} · fly-${rect.id}<br><span style="font-weight:400;opacity:0.7;font-size:0.8em">${data.nNeurons.toLocaleString()} neurons · ${((data as any).n_edges_full ?? 0).toLocaleString()} synapses</span>`;

      const bars = [
        dnBar("fwd", 0, "#38bdf8"),
        dnBar("turnL", 0, "#60a5fa"),
        dnBar("turnR", 0, "#60a5fa"),
        dnBar("dash", 0, "#facc15"),
      ];
      bars.forEach((b) => els.footer.appendChild(b));

      headerEls.set(rect.id, { ...els, bars });
    }
  }
  buildPanels();

  function layoutOverlay() {
    for (const rect of rects) {
      const el = headerEls.get(rect.id);
      if (!el) continue;
      el.root.style.left = `${rect.x}px`;
      el.root.style.top = `${rect.y}px`;
      el.root.style.width = `${rect.w}px`;
      el.root.style.height = `${rect.h}px`;
    }
  }

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight - 20; // leave room for the disclaimer bar
    renderer.setSize(w, window.innerHeight, false);
    rects = computeLayout(w, h, nHiders);
    layoutOverlay();
  }
  window.addEventListener("resize", resize);
  resize();

  function pickRandom(indices: number[], count: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
      out.push(indices[(Math.random() * indices.length) | 0]);
    }
    return out;
  }

  // Synthetic activity: background chatter across all groups + a role-appropriate
  // "burst" group that pulses. This stands in for the live brain<->viewer stream
  // (flyseek/server/ws_server.py, not yet built) so the panel look/feel can be
  // validated against IMG_0897/0898 before wiring in real spikes.
  const burstGroupForRole: Record<string, string[]> = {
    seeker: ["pursuit", "steering"],
    hider: ["looming_escape", "locomotion"],
  };

  let last = performance.now();
  function frame(now: number) {
    const dtMs = now - last;
    last = now;
    const t = now / 1000;

    for (const rect of rects) {
      const panel = panels.get(rect.id)!;
      const spikes: number[] = [];

      // faint background chatter
      spikes.push(...pickRandom(groupIndex["other"] ?? [], 40));

      // role-appropriate pulsing burst (sinusoidal "attention" envelope)
      const burstGroups = burstGroupForRole[rect.role];
      const phase = rect.id.length; // cheap per-panel phase offset
      const envelope = 0.5 + 0.5 * Math.sin(t * 1.3 + phase);
      for (const g of burstGroups) {
        const pool = groupIndex[g] ?? [];
        const n = Math.round(pool.length * 0.15 * envelope);
        spikes.push(...pickRandom(pool, n));
      }
      spikes.push(...pickRandom(groupIndex["vision"] ?? [], 15));

      panel.update(spikes, dtMs);
      panel.render(renderer, rect.x, rect.y, rect.w, rect.h);

      const els = headerEls.get(rect.id)!;
      const fwd = 0.4 + 0.3 * Math.sin(t * 0.9 + phase);
      const turnL = Math.max(0, Math.sin(t * 0.6 + phase));
      const turnR = Math.max(0, -Math.sin(t * 0.6 + phase));
      const dash = envelope > 0.85 ? envelope : 0;
      const vals = [fwd, turnL, turnR, dash];
      els.bars.forEach((b, i) => {
        ((b as any)._fill as HTMLDivElement).style.width = `${Math.round(vals[i] * 100)}%`;
      });
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f87171;padding:20px;font-family:monospace">Failed to load brain data:\n${err}\n\nDid you run:\n  python -m flyseek.connectome.export_viewer_data\nfirst?</pre>`;
  console.error(err);
});
