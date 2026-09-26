"""
Occupancy grid shared by the test arena and The Skeld: walkability, distance to the
nearest wall, and vectorized ray casting.

Coordinates are world units with y up; heading 0 = +x, positive angles counter-
clockwise (a left turn).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from amongusfly.paths import CACHE_DIR


@dataclass
class OccupancyGrid:
    walkable: np.ndarray  # [H, W] bool, row = y index, col = x index
    res: float  # world units per cell
    x0: float  # world x of column 0 center
    y0: float  # world y of row 0 center

    def __post_init__(self):
        self.dist = ndimage.distance_transform_edt(self.walkable) * self.res
        self.h, self.w = self.walkable.shape

    # ----------------------------------------------------------- constructors
    @classmethod
    def arena(cls, width: float = 10.0, height: float = 10.0, res: float = 0.05) -> "OccupancyGrid":
        """Empty rectangular room centered on the origin, one wall cell thick."""
        W, H = int(round(width / res)) + 2, int(round(height / res)) + 2
        walk = np.ones((H, W), dtype=bool)
        walk[0, :] = walk[-1, :] = walk[:, 0] = walk[:, -1] = False
        return cls(walk, res, -width / 2 - res / 2, -height / 2 - res / 2)

    @classmethod
    def skeld(cls) -> "OccupancyGrid":
        g = np.load(CACHE_DIR / "skeld_grid.npz", allow_pickle=True)
        res = float(g["grid_res"])
        walk = g["walkable"].astype(bool)
        # pad with a wall border so ray casts never index outside the array
        walk = np.pad(walk, 1, constant_values=False)
        return cls(walk, res, (float(g["x_min"]) - 1) * res, (float(g["y_min"]) - 1) * res)

    # ---------------------------------------------------------------- queries
    def _cell(self, x, y):
        cx = np.clip(np.rint((np.asarray(x) - self.x0) / self.res).astype(np.int64), 0, self.w - 1)
        cy = np.clip(np.rint((np.asarray(y) - self.y0) / self.res).astype(np.int64), 0, self.h - 1)
        return cy, cx

    def walkable_at(self, x, y) -> np.ndarray:
        cy, cx = self._cell(x, y)
        return self.walkable[cy, cx]

    def dist_at(self, x, y) -> np.ndarray:
        """Approximate distance from (x, y) to the nearest wall (0 inside walls)."""
        cy, cx = self._cell(x, y)
        return self.dist[cy, cx]

    def raycast(self, x: np.ndarray, y: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        """
        x, y: [A] ray origins; angles: [A, R] absolute ray angles.
        Returns wall distance [A, R], capped at max_range. Marches in half-cell steps.
        """
        step = self.res * 0.5
        ts = np.arange(step, max_range + step, step)  # [S]
        px = x[:, None, None] + np.cos(angles)[:, :, None] * ts  # [A, R, S]
        py = y[:, None, None] + np.sin(angles)[:, :, None] * ts
        blocked = ~self.walkable_at(px, py)
        hit = blocked.any(axis=-1)
        first = np.argmax(blocked, axis=-1)
        return np.where(hit, ts[first], max_range)

    def random_walkable_points(self, n: int, rng: np.random.Generator, min_clearance: float = 0.3) -> np.ndarray:
        cand = np.argwhere(self.walkable & (self.dist >= min_clearance))
        pick = cand[rng.integers(0, len(cand), n)]
        return np.stack([self.x0 + pick[:, 1] * self.res, self.y0 + pick[:, 0] * self.res], axis=1)
