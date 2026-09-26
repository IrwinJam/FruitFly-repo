export interface PanelRect {
  id: string;
  role: "seeker" | "hider";
  agent: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Demo layout: seeker large on the left, a column of hider panels on the right. */
export function computeLayout(width: number, height: number, nHiders: number): PanelRect[] {
  const seekerW = width * 0.55;
  const rects: PanelRect[] = [{ id: "seeker", role: "seeker", agent: 0, x: 0, y: 0, w: seekerW, h: height }];
  const colW = width - seekerW;
  const rowH = height / Math.max(1, nHiders);
  for (let i = 0; i < nHiders; i++) {
    rects.push({ id: `hider-${i}`, role: "hider", agent: i + 1, x: seekerW, y: i * rowH, w: colW, h: rowH });
  }
  return rects;
}

/** Header height at the top of every brain panel (name and subtitle). */
export const PANEL_HEAD = 48;

/**
 * Split a brain panel so the brain never sits under the circuit bars. Tall panels: bars along the
 * bottom, brain above them. Short panels: bars on the left under the header, brain on the right.
 */
export function panelAreas(r: Rect): { brain: Rect; bars: Rect; tall: boolean } {
  if (r.h > r.w * 1.2) {
    const barsH = 8 * (r.h > 320 ? 15 : 13) + 14;
    return {
      tall: true,
      bars: { x: r.x, y: r.y + r.h - barsH, w: r.w, h: barsH },
      brain: { x: r.x, y: r.y + PANEL_HEAD, w: r.w, h: Math.max(0, r.h - PANEL_HEAD - barsH) },
    };
  }
  const barsW = Math.min(200, r.w * 0.5);
  return {
    tall: false,
    bars: { x: r.x, y: r.y + PANEL_HEAD, w: barsW, h: r.h - PANEL_HEAD },
    brain: { x: r.x + barsW, y: r.y + 6, w: r.w - barsW - 6, h: r.h - 12 },
  };
}

/** Match layout: title and seeker brain | Skeld map | column of hider brains. */
export function computeMatchLayout(width: number, height: number, nHiders: number): { panels: PanelRect[]; map: Rect; title: Rect } {
  const narrow = width < 900;
  if (narrow) {
    const mapH = height * 0.5;
    const rowW = width / (nHiders + 1);
    const panels: PanelRect[] = [{ id: "seeker", role: "seeker", agent: 0, x: 0, y: mapH, w: rowW, h: height - mapH }];
    for (let i = 0; i < nHiders; i++) {
      panels.push({ id: `hider-${i}`, role: "hider", agent: i + 1, x: rowW * (i + 1), y: mapH, w: rowW, h: height - mapH });
    }
    return { panels, map: { x: 0, y: 0, w: width, h: mapH }, title: { x: 0, y: 0, w: width, h: 0 } };
  }
  const sideW = width * 0.26;
  const titleH = 64;
  const panels: PanelRect[] = [{ id: "seeker", role: "seeker", agent: 0, x: 0, y: titleH, w: sideW, h: height - titleH }];
  const rowH = height / Math.max(1, nHiders);
  for (let i = 0; i < nHiders; i++) {
    panels.push({ id: `hider-${i}`, role: "hider", agent: i + 1, x: width - sideW, y: i * rowH, w: sideW, h: rowH });
  }
  return { panels, map: { x: sideW, y: 0, w: width - 2 * sideW, h: height }, title: { x: 0, y: 0, w: sideW, h: titleH } };
}
