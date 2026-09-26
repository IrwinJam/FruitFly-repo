"""
Local exploration goal policy.

Every `decide_s` seconds (or when blocked) each fly scores candidate directions by open
distance, how unvisited the ground is, alignment with its heading and consistency with its
previous goal, and passes the best as the goal direction. Weights may be per-fly arrays.
"""
from __future__ import annotations

import numpy as np

from amongusfly.world.grid import OccupancyGrid


class ExplorationGoalPolicy:
    def __init__(self, n: int, grid: OccupancyGrid, params: dict, k_dirs: int = 16, ray_range: float = 8.0,
                 bin_units: float = 1.0, novelty_reach: float = 6.0):
        self.n, self.grid, self.p = n, grid, params
        self.dirs = np.linspace(-np.pi, np.pi, k_dirs, endpoint=False)
        self.ray_range, self.bin, self.reach = ray_range, bin_units, novelty_reach
        walk = np.argwhere(grid.walkable)
        self.bx0 = grid.x0 + walk[:, 1].min() * grid.res
        self.by0 = grid.y0 + walk[:, 0].min() * grid.res
        self.bw = int(np.ceil((walk[:, 1].max() - walk[:, 1].min()) * grid.res / bin_units)) + 2
        self.bh = int(np.ceil((walk[:, 0].max() - walk[:, 0].min()) * grid.res / bin_units)) + 2
        self.visits = np.zeros((n, self.bh, self.bw), dtype=np.float32)
        self.goal = np.full(n, np.nan)
        self.next_decide = np.zeros(n)
        self.t = 0.0

    def _bins(self, x, y):
        bx = np.clip(((x - self.bx0) // self.bin).astype(int), 0, self.bw - 1)
        by = np.clip(((y - self.by0) // self.bin).astype(int), 0, self.bh - 1)
        return by, bx

    def step(self, x: np.ndarray, y: np.ndarray, heading: np.ndarray, dt: float, active: np.ndarray | None = None):
        p = self.p
        active = np.ones(self.n, bool) if active is None else active
        by, bx = self._bins(x, y)
        self.visits[np.arange(self.n)[active], by[active], bx[active]] += dt

        # blocked current goal -> decide now
        if np.isfinite(self.goal).any():
            ahead = self.grid.raycast(x, y, np.where(np.isfinite(self.goal), self.goal, 0.0)[:, None], 1.5)[:, 0]
            blocked = np.isfinite(self.goal) & (ahead < 1.0)
        else:
            blocked = np.zeros(self.n, bool)
        decide = active & ((self.t >= self.next_decide) | blocked | ~np.isfinite(self.goal))

        if decide.any():
            ids = np.flatnonzero(decide)
            ang = np.broadcast_to(self.dirs, (len(ids), len(self.dirs)))
            free = self.grid.raycast(x[ids], y[ids], ang, self.ray_range)  # [m, K]
            # novelty: mean 1/(1+visit seconds) at 1-unit samples out to min(free, reach)
            samples = np.arange(1.0, self.reach + 1e-6, 1.0)  # [S]
            px = x[ids, None, None] + np.cos(ang)[:, :, None] * samples
            py = y[ids, None, None] + np.sin(ang)[:, :, None] * samples
            sb_y, sb_x = self._bins(px, py)
            v = self.visits[ids[:, None, None], sb_y, sb_x]  # [m, K, S]
            within = samples[None, None, :] <= np.maximum(free[:, :, None], 1.0)
            novel = np.where(within, 1.0 / (1.0 + v), 0.0).sum(-1) / np.maximum(within.sum(-1), 1)
            align = np.cos(ang - heading[ids, None])
            prev = np.where(np.isfinite(self.goal[ids]), self.goal[ids], heading[ids])
            keep = np.cos(ang - prev[:, None])
            w = lambda k: np.asarray(p[k], dtype=float)[ids, None] if np.ndim(p[k]) else float(p[k])
            score = (w("w_free") * (free / self.ray_range) + w("w_novel") * novel
                     + w("w_align") * align + w("w_keep") * keep)
            score = np.where(free < 1.0, -1e9, score)  # never pick a direction blocked within 1 unit
            self.goal[ids] = self.dirs[np.argmax(score, axis=1)]
            dsec = np.asarray(p["decide_s"], dtype=float)
            self.next_decide[ids] = self.t + (dsec[ids] if dsec.ndim else float(dsec))

        self.t += dt
        return np.where(active, self.goal, np.nan)
