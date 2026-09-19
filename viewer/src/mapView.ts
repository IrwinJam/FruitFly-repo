import type { Replay } from "./replay";

export const AGENT_COLORS = ["#ef4444", "#4ade80", "#60a5fa", "#facc15", "#c084fc", "#f472b6"];

/** Top-down Skeld map drawn on a 2D canvas: floor, vents, agents, vision, events, HUD. */
export class MapView {
  private floor: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /** Agent whose panel is selected: drawn with a trail, its vision range and a ring. */
  selected: number | null = null;
  trailSeconds = 4;

  constructor(public canvas: HTMLCanvasElement, private replay: Replay) {
    this.ctx = canvas.getContext("2d")!;
    const { width, height } = replay.meta.map;
    this.floor = document.createElement("canvas");
    this.floor.width = width;
    this.floor.height = height;
    const fctx = this.floor.getContext("2d")!;
    const img = fctx.createImageData(width, height);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const walk = replay.walkable[r * width + c];
        const p = ((height - 1 - r) * width + c) * 4; // flip: row 0 is lowest y
        img.data[p] = walk ? 34 : 0;
        img.data[p + 1] = walk ? 44 : 0;
        img.data[p + 2] = walk ? 48 : 0;
        img.data[p + 3] = 255;
      }
    }
    fctx.putImageData(img, 0, 0);
  }

  resize(x: number, y: number, w: number, h: number) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    Object.assign(this.canvas.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(tick: number) {
    const rp = this.replay;
    const m = rp.meta;
    const ctx = this.ctx;
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    ctx.clearRect(0, 0, W, H);

    const pad = 12;
    const hudH = 44;
    const scale = Math.min((W - 2 * pad) / m.map.width, (H - hudH - 2 * pad) / m.map.height);
    if (!(scale > 0)) return; // window too small to draw the map
    const ox = (W - m.map.width * scale) / 2;
    const oy = hudH + pad + (H - hudH - 2 * pad - m.map.height * scale) / 2;
    const sx = (x: number) => ox + ((x - m.map.x0) / m.map.res + 0.5) * scale;
    const sy = (y: number) => oy + (m.map.height - 0.5 - (y - m.map.y0) / m.map.res) * scale;
    const units = (u: number) => (u / m.map.res) * scale;

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.floor, ox, oy, m.map.width * scale, m.map.height * scale);

    // vents
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    for (const v of m.vents) {
      const s = units(0.35);
      ctx.strokeRect(sx(v.x) - s, sy(v.y) - s, 2 * s, 2 * s);
    }

    // recent pings and kills
    const tickS = m.tick_ms / 1000;
    for (const e of m.events) {
      if (e.tick > tick) break;
      const age = (tick - e.tick) * tickS;
      if (e.kind === "ping" && age < 1.5) {
        ctx.strokeStyle = `rgba(250, 204, 21, ${1 - age / 1.5})`;
        ctx.lineWidth = 2;
        for (const [px, py] of e.positions) {
          ctx.beginPath();
          ctx.arc(sx(px), sy(py), units(0.5 + age * 1.5), 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      if (e.kind === "kill" && age < 1.0) {
        ctx.fillStyle = `rgba(239, 68, 68, ${0.6 * (1 - age)})`;
        ctx.beginPath();
        ctx.arc(sx(e.x), sy(e.y), units(0.4 + age * 2), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // trails (last few seconds of each living agent's path; brighter for the selected one)
    const trailTicks = Math.round((this.trailSeconds * 1000) / m.tick_ms);
    for (let a = 0; a < m.n_agents; a++) {
      if (rp.field(tick, a, "alive") <= 0.5) continue;
      const color = AGENT_COLORS[a % AGENT_COLORS.length];
      ctx.strokeStyle = color + (this.selected === a ? "cc" : "55");
      ctx.lineWidth = this.selected === a ? 2 : 1;
      ctx.beginPath();
      let first = true;
      for (let t = Math.max(0, tick - trailTicks); t <= tick; t += 3) {
        const px = sx(rp.field(t, a, "x"));
        const py = sy(rp.field(t, a, "y"));
        if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // agents
    for (let a = 0; a < m.n_agents; a++) {
      const x = rp.field(tick, a, "x");
      const y = rp.field(tick, a, "y");
      const hd = rp.field(tick, a, "heading");
      const alive = rp.field(tick, a, "alive") > 0.5;
      const color = AGENT_COLORS[a % AGENT_COLORS.length];
      const px = sx(x);
      const py = sy(y);
      const r = Math.max(4, units(0.3));
      if (!alive) {
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r);
        ctx.moveTo(px + r, py - r); ctx.lineTo(px - r, py + r);
        ctx.stroke();
        continue;
      }
      const hidden = rp.inVent(tick, a);
      const range = m.roles[a] === "seeker" ? m.vision_range.seeker_range_units : m.vision_range.hider_range_units;
      ctx.strokeStyle = color + (this.selected === a ? "99" : "33");
      ctx.lineWidth = this.selected === a ? 1.5 : 1;
      ctx.beginPath();
      ctx.arc(px, py, units(range), 0, Math.PI * 2);
      ctx.stroke();
      if (this.selected === a) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.globalAlpha = hidden ? 0.3 : 1;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0b0f10";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + Math.cos(hd) * r * 1.8, py - Math.sin(hd) * r * 1.8);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "600 11px system-ui, sans-serif";
      ctx.fillText(m.roles[a] === "seeker" ? "S" : `H${a}`, px + r + 3, py - r);
    }

    // HUD
    const phase = rp.phaseAt(tick);
    const t = tick * tickS;
    const mm = Math.floor(t / 60);
    const ss = Math.floor(t % 60).toString().padStart(2, "0");
    const hiders = m.roles.map((r, i) => (r === "hider" ? i : -1)).filter((i) => i >= 0);
    const aliveH = hiders.filter((i) => rp.field(tick, i, "alive") > 0.5).length;
    const labels: Record<string, string> = { hide: "HIDE", seek: "SEEK", final_hide: "FINAL HIDE", over: "MATCH OVER" };
    ctx.fillStyle = phase === "final_hide" ? "#facc15" : phase === "over" ? "#f87171" : "#e5e7eb";
    ctx.font = "700 18px system-ui, sans-serif";
    ctx.fillText(labels[phase] ?? phase, pad, 28);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "500 14px ui-monospace, Consolas, monospace";
    const hud = `${mm}:${ss} / ${Math.floor(m.timers.round_length_s / 60)}:${(m.timers.round_length_s % 60).toString().padStart(2, "0")}   hiders alive ${aliveH}/${hiders.length}`;
    ctx.fillText(hud, pad + 150, 28);
    const over = m.events.find((e) => e.kind === "over");
    if (over && tick >= over.tick) {
      ctx.fillStyle = "#f87171";
      ctx.font = "700 14px system-ui, sans-serif";
      ctx.fillText(over.winner === "seeker" ? "Seeker wins" : "Hiders win", W - 110, 28);
    }
  }
}
