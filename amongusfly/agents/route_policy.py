"""
Map-based route planner: where to go, not how to steer.

Each fly picks a destination (novel area, unvisited room, short walk), plans the shortest
walkable path to it and passes the direction to a look-ahead point on that path as its goal.
It keeps a destination while making progress, bends the goal away from nearby walls and
smooths it over time. Weights may be per-fly arrays.
"""
from __future__ import annotations

import numpy as np

from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.pathing import GridPaths, _NEIGH


def lookahead(grid: OccupancyGrid, field: np.ndarray, x: float, y: float, units: float, target) -> float:
    """World direction from (x, y) to the point `units` ahead along the descent path of a distance field."""
    cy, cx = grid._cell(x, y)
    steps = max(1, int(round(units / grid.res)))
    for _ in range(steps):
        best, nxt = field[cy, cx], None
        for dy, dx in _NEIGH:
            r, c = cy + dy, cx + dx
            if 0 <= r < field.shape[0] and 0 <= c < field.shape[1] and field[r, c] < best:
                best, nxt = field[r, c], (r, c)
        if nxt is None:
            break
        cy, cx = nxt
    px, py = grid.x0 + cx * grid.res, grid.y0 + cy * grid.res
    if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
        px, py = target
    return float(np.arctan2(py - y, px - x))


# A fly that has been scraping a wall for this long re-plans: its current route is not
# working (a corner, a table edge, or a goal that points through a wall).
REPLAN_AFTER_BLOCKED_S = 0.6
PROGRESS_UNITS = 0.5  # getting this much closer (walkable distance) counts as progress

AVOID_FREE_UNITS = 0.0  # probe-fan obstacle avoidance; off by default (centring is used instead)


def free_direction(grid: OccupancyGrid, x, y, goal, min_free: float = 1.5, max_turn_deg: float = 60.0,
                   n_probe: int = 9) -> np.ndarray:
    """
    Wall-aware goal direction: probe the fly's line of sight in a fan around the planned goal
    direction and take the direction closest to the goal with at least `min_free` of open
    distance (or the freest probe if none has). Only the goal direction changes; steering still
    comes from the compass circuit.
    """
    x, y, goal = np.asarray(x, float), np.asarray(y, float), np.asarray(goal, float)
    ok = np.isfinite(goal)
    if not ok.any():
        return goal
    offs = np.deg2rad(np.linspace(-max_turn_deg, max_turn_deg, n_probe))
    order = np.argsort(np.abs(offs))  # probe the goal direction first, then outwards
    ang = np.nan_to_num(goal)[:, None] + offs[None, :]
    free = grid.raycast(x, y, ang, float(np.max(min_free) * 2 + 1.0))
    need = np.reshape(min_free, (-1, 1)) if np.ndim(min_free) else min_free
    good = free >= need
    pick = np.full(len(goal), -1)
    for j in order:  # first (most goal-aligned) probe that is clear enough
        pick = np.where((pick < 0) & good[:, j], j, pick)
    pick = np.where(pick < 0, np.argmax(free, axis=1), pick)
    return np.where(ok, goal + offs[pick], goal)


# Corridor centring: near a wall the goal direction is bent away from it, along the distance
# field's gradient, so the compass circuit steers the fly back towards the middle.
WALL_REPEL_GAIN = 1.5  # how hard the goal is bent at the wall itself
WALL_REPEL_RANGE = 0.6  # clearance at which the bend fades to zero (about a corridor half-width)


def wall_normal(grid: OccupancyGrid, x, y):
    """Unit vector pointing away from the nearest wall (gradient of the clearance field)."""
    h = grid.res
    gx = (grid.dist_at(x + h, y) - grid.dist_at(x - h, y)) / (2 * h)
    gy = (grid.dist_at(x, y + h) - grid.dist_at(x, y - h)) / (2 * h)
    n = np.hypot(gx, gy)
    return np.where(n > 0, gx / np.maximum(n, 1e-9), 0.0), np.where(n > 0, gy / np.maximum(n, 1e-9), 0.0)


def centre_goal(grid: OccupancyGrid, x, y, goal, gain=WALL_REPEL_GAIN, body_radius: float = 0.25):
    """Bend each fly's goal direction away from a wall it is close to."""
    x, y, goal = np.asarray(x, float), np.asarray(y, float), np.asarray(goal, float)
    ok = np.isfinite(goal)
    if not ok.any() or not np.any(np.asarray(gain) > 0):
        return goal
    near = np.clip((WALL_REPEL_RANGE - grid.dist_at(x, y)) / (WALL_REPEL_RANGE - body_radius), 0.0, 1.0)
    nx, ny = wall_normal(grid, x, y)
    w = np.asarray(gain, float) * near
    g0 = np.nan_to_num(goal)
    return np.where(ok, np.arctan2(np.sin(g0) + w * ny, np.cos(g0) + w * nx), goal)


# Goal persistence: the goal vector is low-pass filtered with this time constant, so grid
# quantisation in the path look-ahead does not make the goal direction jump.
GOAL_SMOOTH_S = 0.3


class GoalSmoother:
    """Per-fly exponential smoothing of a world goal direction (as a unit vector)."""

    def __init__(self, n: int):
        self.vx, self.vy = np.zeros(n), np.zeros(n)
        self.has = np.zeros(n, bool)

    def __call__(self, goal: np.ndarray, dt: float, tau) -> np.ndarray:
        goal = np.asarray(goal, float)
        ok = np.isfinite(goal)
        tau = np.asarray(tau, float)
        a = np.where(tau > 0, 1.0 - np.exp(-dt / np.maximum(tau, 1e-6)), 1.0)
        gx, gy = np.cos(np.nan_to_num(goal)), np.sin(np.nan_to_num(goal))
        fresh = ok & ~self.has  # no history yet: take the goal as is
        self.vx = np.where(fresh, gx, np.where(ok, self.vx + a * (gx - self.vx), self.vx))
        self.vy = np.where(fresh, gy, np.where(ok, self.vy + a * (gy - self.vy), self.vy))
        self.has = ok
        return np.where(ok, np.arctan2(self.vy, self.vx), np.nan)


class RouteGoalPolicy:
    def __init__(self, n: int, grid: OccupancyGrid, paths: GridPaths, rooms: np.ndarray, params: dict,
                 n_candidates: int = 24, bin_units: float = 1.0, seed: int = 0, centre: bool = True):
        self.n, self.grid, self.paths, self.rooms, self.p = n, grid, paths, rooms, params
        self.centre = centre
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
        self.blocked_s = np.zeros(n)  # how long this fly has been in wall contact
        self.smooth = GoalSmoother(n)
        # Commitment: a fly keeps its destination while it is getting closer to it; `replan_s` is
        # how long it tolerates no progress before choosing a new one.
        self.best_dist = np.full(n, np.inf)
        self.last_progress = np.zeros(n)
        self.t = 0.0

    def _param(self, key, i, default=None):
        if key not in self.p:  # adapters trained before this parameter existed
            return default
        v = self.p[key]
        return float(v[i]) if np.ndim(v) else float(v)

    def _vec(self, key, default):
        v = self.p.get(key, default)
        return np.full(self.n, float(v)) if not np.ndim(v) else np.asarray(v, float)

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
        self.best_dist[i] = np.inf
        self.last_progress[i] = self.t

    def _lookahead(self, i, x, y) -> float:
        return lookahead(self.grid, self.field[i], x, y, self._param("lookahead_units", i), self.target[i])

    def step(self, x, y, heading, dt: float, active: np.ndarray | None = None,
             blocked: np.ndarray | None = None) -> np.ndarray:
        """blocked: per-fly wall contact from the last tick; sustained contact forces a re-plan."""
        active = np.ones(self.n, bool) if active is None else active
        if blocked is None:
            self.blocked_s[:] = 0.0
        else:
            self.blocked_s = np.where(np.asarray(blocked, bool), self.blocked_s + dt, 0.0)
        give_up = self.blocked_s >= REPLAN_AFTER_BLOCKED_S
        self.blocked_s = np.where(give_up, 0.0, self.blocked_s)
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
            if self.field[i] is not None:  # walkable distance still to go
                d = float(self.field[i][cy[i], cx[i]])
                if d < self.best_dist[i] - PROGRESS_UNITS:
                    self.best_dist[i], self.last_progress[i] = d, self.t
            stalled = self.t - self.last_progress[i] >= self._param("replan_s", i)
            forced = give_up[i]  # scraping a wall
            if self.target[i] is None or reached or stalled or forced:
                self._plan(i, x[i], y[i])
            if self.field[i] is not None:
                goal[i] = self._lookahead(i, x[i], y[i])
        goal = free_direction(self.grid, x, y, goal, np.full(self.n, AVOID_FREE_UNITS))
        if self.centre:  # role policies apply centring + smoothing themselves, after their own overrides
            goal = centre_goal(self.grid, x, y, goal, self._vec("wall_repel", WALL_REPEL_GAIN))
            goal = self.smooth(goal, dt, self._vec("goal_smooth_s", GOAL_SMOOTH_S))
        self.t += dt
        return goal
