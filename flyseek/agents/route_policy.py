"""
Map-based route goal policy (engineered, disclosed; its weights are learned in
Phase 4 training). It replaces the local ray policy, which rarely found Cafeteria's
narrow exits (docs/phase4_diag_explore.png).

It decides WHERE to go; the connectome decides how to steer there. Each fly:
  1. every `replan_s` seconds, or when its target is reached, samples candidate
     target points and scores them:
        w_novel * (how unvisited the area around the candidate is)
      + w_room  * (candidate is in a room this fly hasn't visited)
      - w_dist  * (walkable path length from the fly / 20 units)
  2. plans the shortest walkable path to the best candidate (Dijkstra field)
  3. every tick, looks `lookahead_units` ahead along that path and passes the WORLD
     direction to that point as the FC2 goal bump. Steering comes from the connectome.
All weights may be per-fly arrays [A].
"""
from __future__ import annotations

import numpy as np

from flyseek.world.grid import OccupancyGrid
from flyseek.world.pathing import GridPaths, _NEIGH


class RouteGoalPolicy:
    def __init__(self, n: int, grid: OccupancyGrid, paths: GridPaths, rooms: np.ndarray, params: dict,
                 n_candidates: int = 24, bin_units: float = 1.0, seed: int = 0):
        self.n, self.grid, self.paths, self.rooms, self.p = n, grid, paths, rooms, params
        self.nc = n_candidates
        self.rng = np.random.default_rng(seed)
        self.bin = bin_units
        walk = np.argwhere(grid.walkable)
        self.bx0 = grid.x0 + walk[:, 1].min() * grid.res
        self.by0 = grid.y0 + walk[:, 0].min() * grid.res
        self.bw = int(np.ceil((walk[:, 1].max() - walk[:, 1].min()) * grid.res / bin_units)) + 3
        self.bh = int(np.ceil((walk[:, 0].max() - walk[:, 0].min()) * grid.res / bin_units)) + 3
        self.visits = np.zeros((n, self.bh, self.bw), dtype=np.float32)
        self.visited_rooms = [set() for _ in range(n)]
        free = paths.free
        cand = np.argwhere(free & (grid.dist >= 0.4))
        self.cand_xy = np.stack([grid.x0 + cand[:, 1] * grid.res, grid.y0 + cand[:, 0] * grid.res], axis=1)
        self.cand_room = rooms[cand[:, 0], cand[:, 1]]
        self.target = [None] * n
        self.field = [None] * n
        self.next_plan = np.zeros(n)
        self.t = 0.0

    def _param(self, key, i):
        v = self.p[key]
        return float(v[i]) if np.ndim(v) else float(v)

    def _bins(self, x, y):
        bx = np.clip(((np.asarray(x) - self.bx0) // self.bin).astype(int), 0, self.bw - 1)
        by = np.clip(((np.asarray(y) - self.by0) // self.bin).astype(int), 0, self.bh - 1)
        return by, bx

    def _plan(self, i, x, y):
        idx = self.rng.choice(len(self.cand_xy), self.nc, replace=False)
        cxy = self.cand_xy[idx]
        from_fly = self.paths.distance_field([(float(x), float(y))])
        geo = self.paths.field_at(from_fly, cxy[:, 0], cxy[:, 1])
        by, bx = self._bins(cxy[:, 0], cxy[:, 1])
        v = np.zeros(self.nc)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v += self.visits[i, np.clip(by + dy, 0, self.bh - 1), np.clip(bx + dx, 0, self.bw - 1)]
        novel = 1.0 / (1.0 + v / 9.0)
        new_room = np.array([r not in self.visited_rooms[i] and r != "" for r in self.cand_room[idx]], dtype=float)
        score = (self._param("w_novel", i) * novel + self._param("w_room", i) * new_room
                 - self._param("w_dist", i) * np.where(np.isfinite(geo), geo, 1e3) / 20.0)
        score = np.where(np.isfinite(geo) & (geo > 1.5), score, -np.inf)
        if not np.isfinite(score).any():
            return
        best = int(np.argmax(score))
        self.target[i] = (float(cxy[best, 0]), float(cxy[best, 1]))
        self.field[i] = self.paths.distance_field([self.target[i]])
        self.next_plan[i] = self.t + self._param("replan_s", i)

    def _lookahead(self, i, x, y) -> float:
        f = self.field[i]
        g = self.grid
        cy, cx = g._cell(x, y)
        steps = max(1, int(round(self._param("lookahead_units", i) / g.res)))
        for _ in range(steps):
            best, nxt = f[cy, cx], None
            for dy, dx in _NEIGH:
                r, c = cy + dy, cx + dx
                if 0 <= r < f.shape[0] and 0 <= c < f.shape[1] and f[r, c] < best:
                    best, nxt = f[r, c], (r, c)
            if nxt is None:
                break
            cy, cx = nxt
        px, py = g.x0 + cx * g.res, g.y0 + cy * g.res
        if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
            px, py = self.target[i]
        return float(np.arctan2(py - y, px - x))

    def step(self, x, y, heading, dt: float, active: np.ndarray | None = None) -> np.ndarray:
        active = np.ones(self.n, bool) if active is None else active
        by, bx = self._bins(x, y)
        cy, cx = self.grid._cell(x, y)
        goal = np.full(self.n, np.nan)
        for i in range(self.n):
            if not active[i]:
                continue
            self.visits[i, by[i], bx[i]] += dt
            room = self.rooms[cy[i], cx[i]]
            if room:
                self.visited_rooms[i].add(str(room))
            reached = self.target[i] is not None and np.hypot(self.target[i][0] - x[i], self.target[i][1] - y[i]) < 1.0
            if self.target[i] is None or reached or self.t >= self.next_plan[i]:
                self._plan(i, x[i], y[i])
            if self.field[i] is not None:
                goal[i] = self._lookahead(i, x[i], y[i])
        self.t += dt
        return goal
