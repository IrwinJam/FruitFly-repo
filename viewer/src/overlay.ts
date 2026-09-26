import type { CircuitLevel } from "./circuits";
import { panelAreas, type PanelRect } from "./layout";
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

  /**
   * Panel header (name and subtitle) and circuit bars, kept clear of the brain. A caught fly is greyed out.
   */
  panel(r: PanelRect, o: {
    title: string; color: string; subtitle: string; alive: boolean; caughtAt: string | null; selected: boolean;
    circuits: CircuitLevel[] | null; danger: number | null;
  }) {
    if (r.w < 60 || r.h < 50) return;
    const c = this.ctx;
    c.save();
    if (!o.alive) {  // dim the whole panel, brain included
      c.fillStyle = "rgba(0, 0, 0, 0.55)";
      c.fillRect(r.x, r.y, r.w, r.h);
    }
    const ink = o.alive ? "#e6e7e9" : "#5d626b";
    const sub = o.alive ? "#8a8f98" : "#4a4f57";
    if (o.selected) {
      c.strokeStyle = o.color + "99";
      c.lineWidth = 1;
      c.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    }
    c.font = "600 14px system-ui, -apple-system, 'Segoe UI', sans-serif";
    c.fillStyle = ink;
    c.fillText(o.title, r.x + 14, r.y + 25);
    if (o.caughtAt) {
      const tw = c.measureText(o.title).width;
      c.font = "400 12px system-ui, -apple-system, 'Segoe UI', sans-serif";
      c.fillStyle = sub;
      c.fillText(`caught ${o.caughtAt}`, r.x + 14 + tw + 8, r.y + 25);
    }
    c.font = "400 11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    c.fillStyle = sub;
    c.fillText(o.subtitle, r.x + 14, r.y + 40);

    // circuit bars in their own area (living flies only: a caught fly's brain is no longer driven);
    // short panels get short labels so the brain beside them keeps its room
    const { bars, tall } = panelAreas(r);
    if (o.circuits && o.alive && bars.h > 40) {
      const rows = o.circuits.filter((x) => x.everActive || x.key === "navigation" || x.key === "steering");
      const n = rows.length + (o.danger !== null ? 1 : 0);
      const rowH = Math.min(r.h > 320 ? 15 : 13, Math.floor((bars.h - 8) / Math.max(1, n)));
      const barW = tall ? Math.min(72, r.w * 0.22) : 48;
      const bx = bars.x + 14;
      let y = tall ? bars.y + bars.h - 10 - n * rowH : bars.y + 2;
      c.font = "400 11px system-ui, -apple-system, 'Segoe UI', sans-serif";
      const row = (label: string, level: number, color: string, ink: string) => {
        c.fillStyle = "rgba(255,255,255,0.06)";
        c.fillRect(bx, y + rowH / 2 - 1, barW, 2);
        c.fillStyle = color;
        c.globalAlpha = 0.45 + 0.55 * level;
        c.fillRect(bx, y + rowH / 2 - 1.5, Math.max(1, barW * level), 3);
        c.globalAlpha = 1;
        c.fillStyle = ink;
        c.fillText(tall ? label : label.split(" · ")[0], bx + barW + 8, y + rowH / 2 + 4);
        y += rowH;
      };
      for (const x of rows) row(x.label, x.level, x.color, x.level > 0.15 ? "#c9ccd2" : "#62666f");
      if (o.danger !== null) row("Danger meter", o.danger, "#c98a6b", o.danger > 0 ? "#d7b3a0" : "#62666f");
    }
    c.restore();
  }

  /** Recent game events, newest last, in the given rectangle's lower-left corner. */
  eventFeed(rect: { x: number; y: number; w: number; h: number }, rp: Replay, tick: number, names: (a: number) => string) {
    const c = this.ctx;
    const tickS = rp.meta.tick_ms / 1000;
    const shown: [string, string][] = [];
    const last = tick >= rp.meta.n_ticks - 1;  // a catch logged after the last frame is shown on it
    for (const e of rp.meta.events as ReplayEvent[]) {
      if (e.tick > tick && !(last && e.kind === "kill")) continue;
      const t = e.tick * tickS;
      const ts = `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`;
      if (e.kind === "kill") shown.push([ts, `${names(e.victim)} caught`]);
      else if (e.kind === "vent_enter") shown.push([ts, `${names(e.agent)} used a vent`]);
      else if (e.kind === "ping") shown.push([ts, "Ping: hiders revealed"]);
      else if (e.kind === "phase" && e.phase !== "hide") shown.push([ts, e.phase === "seek" ? "Seeker released" : "Final Hide"]);
    }
    const recent = shown.slice(-5);
    c.save();
    c.font = "400 12px system-ui, -apple-system, 'Segoe UI', sans-serif";
    recent.forEach(([t, text], i) => {
      const y = rect.y + rect.h - 16 - (recent.length - 1 - i) * 17;
      const fade = 0.45 + 0.55 * ((i + 1) / recent.length);
      c.globalAlpha = fade;
      c.fillStyle = "#7d828b";
      c.fillText(t, rect.x + 16, y);
      c.fillStyle = "#d4d6db";
      c.fillText(text, rect.x + 56, y);
    });
    c.restore();
  }

  /** Title in the band above the seeker panel ("AM" in the accent colour). */
  brand(rect: { x: number; y: number; w: number; h: number }) {
    if (rect.h < 20) return;
    const c = this.ctx;
    c.save();
    c.textAlign = "left";
    c.font = "700 30px system-ui, -apple-system, 'Segoe UI', sans-serif";
    const x = rect.x + 14;
    const y = rect.y + rect.h / 2 + 11;
    c.fillStyle = "#d98c5f";
    c.fillText("AM", x, y);
    c.fillStyle = "#e6e7e9";
    c.fillText("ongus V1", x + c.measureText("AM").width, y);
    c.restore();
  }

  /** Reel caption: a heading and an optional second line, centred in the given band. */
  caption(rect: { x: number; y: number; w: number; h: number }, lines: string[], alpha: number) {
    if (alpha <= 0 || !lines.length) return;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.textAlign = "center";
    const sizes = lines.map((_, i) => (i === 0 ? 26 : 16));
    const total = sizes.reduce((t, z) => t + z * 1.4, 0);
    let y = rect.y + (rect.h - total) / 2;
    const cx = rect.x + rect.w / 2;
    lines.forEach((l, i) => {
      y += sizes[i] * 1.2;
      c.font = `${i === 0 ? 600 : 400} ${sizes[i]}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
      c.fillStyle = i === 0 ? "#f1f2f4" : "#a3a7ae";
      c.fillText(l, cx, y);
      y += sizes[i] * 0.2;
    });
    c.restore();
  }

  /** Full-screen card (title / end), alpha 0..1. */
  card(w: number, h: number, alpha: number, lines: { text: string; size: number; color?: string; weight?: number }[]) {
    if (alpha <= 0) return;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.fillStyle = "rgba(4, 5, 7, 0.86)";
    c.fillRect(0, 0, w, h);
    const total = lines.reduce((s, l) => s + l.size * 1.5, 0);
    let y = (h - total) / 2;
    c.textAlign = "center";
    for (const l of lines) {
      y += l.size * 1.5;
      c.font = `${l.weight ?? 400} ${l.size}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
      c.fillStyle = l.color ?? "#e6e7e9";
      c.fillText(l.text, w / 2, y);
    }
    c.restore();
  }
}
