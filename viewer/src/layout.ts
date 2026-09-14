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

/** Demo layout: seeker large on the left, a column of hider panels on the right (IMG_0898). */
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

/** Match layout: seeker brain | Skeld map | column of hider brains. */
export function computeMatchLayout(width: number, height: number, nHiders: number): { panels: PanelRect[]; map: Rect } {
  const narrow = width < 900;
  if (narrow) {
    const mapH = height * 0.5;
    const rowW = width / (nHiders + 1);
    const panels: PanelRect[] = [{ id: "seeker", role: "seeker", agent: 0, x: 0, y: mapH, w: rowW, h: height - mapH }];
    for (let i = 0; i < nHiders; i++) {
      panels.push({ id: `hider-${i}`, role: "hider", agent: i + 1, x: rowW * (i + 1), y: mapH, w: rowW, h: height - mapH });
    }
    return { panels, map: { x: 0, y: 0, w: width, h: mapH } };
  }
  const sideW = width * 0.26;
  const panels: PanelRect[] = [{ id: "seeker", role: "seeker", agent: 0, x: 0, y: 0, w: sideW, h: height }];
  const rowH = height / Math.max(1, nHiders);
  for (let i = 0; i < nHiders; i++) {
    panels.push({ id: `hider-${i}`, role: "hider", agent: i + 1, x: width - sideW, y: i * rowH, w: sideW, h: rowH });
  }
  return { panels, map: { x: sideW, y: 0, w: width - 2 * sideW, h: height } };
}
