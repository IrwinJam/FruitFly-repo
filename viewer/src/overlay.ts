import type { CircuitLevel } from "./circuits";
import type { PanelRect } from "./layout";
import type { Replay, ReplayEvent } from "./replay";

/**
 * Everything textual on top of the brain panels and map is drawn on one 2D canvas (not
 * HTML), so the in-viewer recorder can composite it into the video.
 */
export class Overlay {
  ctx: CanvasRenderingContext2D;
  constructor(public canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  resize(w: number, h: number) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    Object.assign(this.canvas.style, { width: `${w}px`, height: `${h}px` });
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
  }

  /** IMG_0897-style panel header, live circuit bars and (for hiders) the danger meter. */
  panel(r: PanelRect, o: {
    title: string; color: string; subtitle: string; alive: boolean; selected: boolean;
    circuits: CircuitLevel[] | null; danger: number | null; activityMs: number;
  }) {
    if (r.w < 60 || r.h < 50) return;
    const c = this.ctx;
    c.save();
    if (o.selected) {
      c.strokeStyle = o.color;
      c.lineWidth = 2;
      c.strokeRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2);
    }
    c.fillStyle = "#e5e7eb";
    c.font = "700 15px system-ui, sans-serif";
    c.shadowColor = "#000";
    c.shadowBlur = 4;
    c.fillText(o.title, r.x + 12, r.y + 22);
    const tw = c.measureText(o.title).width;
    c.fillStyle = o.color;
    c.beginPath();
    c.arc(r.x + 12 + tw + 10, r.y + 17, 5, 0, Math.PI * 2);
    c.fill();
    if (!o.alive) {
      c.fillStyle = "#f87171";
      c.font = "700 12px system-ui, sans-serif";
      c.fillText("CAUGHT", r.x + 12 + tw + 22, r.y + 21);
    }
    c.fillStyle = "#94a3b8";
    c.font = "500 11px system-ui, sans-serif";
    c.fillText(o.subtitle, r.x + 12, r.y + 38);

    // circuit bars, bottom of the panel
    if (o.circuits && r.h > 170) {
      const rows = o.circuits.filter((x) => x.everActive || x.key === "navigation" || x.key === "steering");
      const rowH = r.h > 320 ? 15 : 12;
      const barW = Math.min(90, r.w * 0.28);
      let y = r.y + r.h - 16 - rows.length * rowH - (o.danger !== null ? rowH + 4 : 0);
      c.font = `500 ${rowH > 12 ? 11 : 10}px system-ui, sans-serif`;
      c.shadowBlur = 3;
      for (const x of rows) {
        c.fillStyle = "#1f2937";
        c.fillRect(r.x + 12, y + 3, barW, rowH - 6);
        c.fillStyle = x.color;
        c.globalAlpha = 0.35 + 0.65 * x.level;
        c.fillRect(r.x + 12, y + 3, barW * x.level, rowH - 6);
        c.globalAlpha = 1;
        c.fillStyle = x.level > 0.15 ? "#e5e7eb" : "#6b7280";
        c.fillText(x.label, r.x + 18 + barW, y + rowH - 3);
        y += rowH;
      }
      if (o.danger !== null) {
        y += 4;
        c.fillStyle = "#1f2937";
        c.fillRect(r.x + 12, y + 3, barW, rowH - 6);
        c.fillStyle = "#ef4444";
        c.fillRect(r.x + 12, y + 3, barW * o.danger, rowH - 6);
        c.fillStyle = o.danger > 0 ? "#fca5a5" : "#6b7280";
        c.fillText("Danger meter (seeker nearby)", r.x + 18 + barW, y + rowH - 3);
      }
      c.fillStyle = "#64748b";
      c.font = "500 10px system-ui, sans-serif";
      c.fillText(`Activity · glow decays over ~${o.activityMs} ms · bars: mean firing rate`, r.x + 12, r.y + r.h - 5);
    }
    c.restore();
  }

  /** Recent game events, newest last, in the given rectangle's lower-left corner. */
  eventFeed(rect: { x: number; y: number; w: number; h: number }, rp: Replay, tick: number, names: (a: number) => string) {
    const c = this.ctx;
    const tickS = rp.meta.tick_ms / 1000;
    const shown: string[] = [];
    for (const e of rp.meta.events as ReplayEvent[]) {
      if (e.tick > tick) break;
      const t = e.tick * tickS;
      const ts = `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`;
      if (e.kind === "kill") shown.push(`${ts}  ${names(e.victim)} caught`);
      else if (e.kind === "vent_enter") shown.push(`${ts}  ${names(e.agent)} vented`);
      else if (e.kind === "ping") shown.push(`${ts}  ping: hiders revealed`);
      else if (e.kind === "phase" && e.phase !== "hide") shown.push(`${ts}  ${e.phase === "seek" ? "seeker released" : "FINAL HIDE"}`);
    }
    const last = shown.slice(-5);
    c.save();
    c.font = "500 12px ui-monospace, Consolas, monospace";
    c.shadowColor = "#000";
    c.shadowBlur = 3;
    last.forEach((s, i) => {
      c.fillStyle = i === last.length - 1 ? "#e5e7eb" : "#94a3b8";
      c.fillText(s, rect.x + 14, rect.y + rect.h - 14 - (last.length - 1 - i) * 16);
    });
    c.restore();
  }

  /** Full-screen card (title / end), alpha 0..1. */
  card(w: number, h: number, alpha: number, lines: { text: string; size: number; color?: string; weight?: number }[]) {
    if (alpha <= 0) return;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.fillStyle = "rgba(0,0,0,0.82)";
    c.fillRect(0, 0, w, h);
    const total = lines.reduce((s, l) => s + l.size * 1.5, 0);
    let y = (h - total) / 2;
    c.textAlign = "center";
    for (const l of lines) {
      y += l.size * 1.5;
      c.font = `${l.weight ?? 500} ${l.size}px system-ui, sans-serif`;
      c.fillStyle = l.color ?? "#e5e7eb";
      c.fillText(l.text, w / 2, y);
    }
    c.restore();
  }
}
