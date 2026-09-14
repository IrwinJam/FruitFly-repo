export interface PanelRect {
  id: string;
  role: "seeker" | "hider";
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Seeker large on the left, a column of hider panels on the right
 * (matches IMG_0898's "one big + a stack of thumbnails" composition).
 */
export function computeLayout(width: number, height: number, nHiders: number): PanelRect[] {
  const seekerW = width * 0.55;
  const rects: PanelRect[] = [
    { id: "seeker", role: "seeker", x: 0, y: 0, w: seekerW, h: height },
  ];
  const colW = width - seekerW;
  const rowH = height / Math.max(1, nHiders);
  for (let i = 0; i < nHiders; i++) {
    rects.push({
      id: `hider-${i}`,
      role: "hider",
      x: seekerW,
      y: i * rowH,
      w: colW,
      h: rowH,
    });
  }
  return rects;
}
