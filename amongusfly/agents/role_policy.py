"""
Seeker and hider policies: where each fly goes, using only what a player of that role knows.
The brain steers there through the goal signal.

Seeker: chase a hider in sight; otherwise go to where one was last seen (for `memory_s`);
otherwise, in Final Hide, head for the nearest ping; otherwise explore with the route planner.

Hider: pick a hiding spot scored by distance from the last known seeker position, concealment,
a nearby vent and travel cost; stand still there for `camp_s`; re-plan (at most once a second)
when the seeker is in sight or the danger meter passes `flee_danger`.

RoleController runs both roles for a mixed batch. All parameters may be per-fly arrays.
"""
from __future__ import annotations

import numpy as np

from amongusfly.agents.route_policy import (AVOID_FREE_UNITS, GOAL_SMOOTH_S, REPLAN_AFTER_BLOCKED_S, WALL_REPEL_GAIN,
                                         GoalSmoother, RouteGoalPolicy, centre_goal, free_direction, lookahead)
from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.pathing import GridPaths

_CONCEAL_CACHE: dict = {}


def _vec(params: dict, key: str, n: int, default: float = AVOID_FREE_UNITS) -> np.ndarray:
    v = params.get(key, default)
    return np.full(n, float(v)) if not np.ndim(v) else np.asarray(v, float)


def _p(params: dict, key: str, i: int) -> float:
    v = params[key]
    return float(v[i]) if np.ndim(v) else float(v)


class SeekerRolePolicy:
    def __init__(self, n: int, grid: OccupancyGrid, paths: GridPaths, rooms: np.ndarray, params: dict, seed: int = 0):
        self.n, self.grid, self.paths, self.p = n, grid, paths, params
        self.route = RouteGoalPolicy(n, grid, paths, rooms, params, seed=seed, centre=False)
        self.last_seen: list = [None] * n
        self.last_seen_t = np.full(n, -np.inf)
        self.chase_target: list = [None] * n
        self.chase_field: list = [None] * n
        self.pings: list = [None] * n
        self.smooth = GoalSmoother(n)
        self.t = 0.0

    def _field_to(self, i, target):
        old = self.chase_target[i]
        if old is None or np.hypot(old[0] - target[0], old[1] - target[1]) > 0.5:
            self.chase_target[i] = target
            self.chase_field[i] = self.paths.distance_field([target])
        return self.chase_field[i]

    def step(self, x, y, heading, dt, active, seen_xy: list, pings: list | None = None,
             blocked: np.ndarray | None = None):
        """seen_xy[i]: (x, y) of the nearest hider in sight or None. pings[i]: list of ping positions or None."""
        was_blocked = self.route.blocked_s.copy()
        goal = self.route.step(x, y, heading, dt, active, blocked)
        for i in np.flatnonzero((was_blocked > 0) & (self.route.blocked_s == 0.0)):
            self.chase_target[i] = None  # a chase route that was scraping a wall gets re-pathed too
        for i in range(self.n):
            if not active[i]:
                continue
            if pings is not None and pings[i]:
                self.pings[i] = list(pings[i])
            if seen_xy[i] is not None:
                self.last_seen[i], self.last_seen_t[i] = seen_xy[i], self.t
            target = None
            if self.last_seen[i] is not None and self.t - self.last_seen_t[i] <= _p(self.p, "memory_s", i):
                target = self.last_seen[i]
                if seen_xy[i] is None and np.hypot(target[0] - x[i], target[1] - y[i]) < 0.8:
                    self.last_seen[i] = None  # reached the last-seen spot, nobody there
                    target = None
            if target is None and self.pings[i] and _p(self.p, "ping_follow", i) > 0.5:
                target = min(self.pings[i], key=lambda q: np.hypot(q[0] - x[i], q[1] - y[i]))
                if np.hypot(target[0] - x[i], target[1] - y[i]) < 0.8:
                    self.pings[i] = [q for q in self.pings[i] if q != target]
                    target = None
            if target is None:
                continue
            if np.hypot(target[0] - x[i], target[1] - y[i]) < 1.0:
                goal[i] = float(np.arctan2(target[1] - y[i], target[0] - x[i]))
            else:
                goal[i] = lookahead(self.grid, self._field_to(i, target), x[i], y[i], _p(self.p, "chase_lookahead", i),
                                    target)
        goal = free_direction(self.grid, x, y, goal, np.full(self.n, AVOID_FREE_UNITS))
        goal = centre_goal(self.grid, x, y, goal, _vec(self.p, "wall_repel", self.n, WALL_REPEL_GAIN))
        goal = self.smooth(goal, dt, _vec(self.p, "goal_smooth_s", self.n, GOAL_SMOOTH_S))
        self.t += dt
        return goal, np.ones(self.n)


def concealment(grid: OccupancyGrid, xy: np.ndarray, view: float = 6.0, n_rays: int = 16) -> np.ndarray:
    """1 - mean free sight distance (capped at `view`) / view, per point."""
    ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    d = grid.raycast(xy[:, 0], xy[:, 1], np.broadcast_to(ang, (len(xy), n_rays)).copy(), view)
    return 1.0 - d.mean(axis=1) / view


class HiderRolePolicy:
    def __init__(self, n: int, grid: OccupancyGrid, paths: GridPaths, params: dict, vents: list, seeker_start,
                 n_candidates: int = 32, seed: int = 0):
        self.n, self.grid, self.paths, self.p = n, grid, paths, params
        self.nc = n_candidates
        self.rng = np.random.default_rng(seed)
        cand = np.argwhere(paths.free & (grid.dist >= 0.4))
        self.cand_xy = np.stack([grid.x0 + cand[:, 1] * grid.res, grid.y0 + cand[:, 0] * grid.res], axis=1)
        key = id(grid)
        if key not in _CONCEAL_CACHE:
            _CONCEAL_CACHE[key] = concealment(grid, self.cand_xy)
        self.conceal = _CONCEAL_CACHE[key]
        vxy = np.array([[v["x"], v["y"]] for v in vents])
        self.near_vent = (np.hypot(self.cand_xy[:, None, 0] - vxy[None, :, 0],
                                   self.cand_xy[:, None, 1] - vxy[None, :, 1]).min(axis=1) <= 2.0).astype(float)
        self.known = [tuple(map(float, seeker_start))] * n
        self.spot: list = [None] * n
        self.field: list = [None] * n
        self.camp_until = np.full(n, -np.inf)
        self.camping = np.zeros(n, bool)
        self.next_threat_plan = np.zeros(n)
        self.blocked_s = np.zeros(n)
        self.smooth = GoalSmoother(n)
        self.t = 0.0

    def _plan(self, i, x, y):
        idx = self.rng.choice(len(self.cand_xy), self.nc, replace=False)
        cxy = self.cand_xy[idx]
        from_me = self.paths.field_at(self.paths.distance_field([(float(x), float(y))]), cxy[:, 0], cxy[:, 1])
        from_seeker = self.paths.field_at(self.paths.distance_field([self.known[i]]), cxy[:, 0], cxy[:, 1])
        fin = np.isfinite(from_me) & np.isfinite(from_seeker)
        score = (_p(self.p, "w_far", i) * np.where(fin, from_seeker, 0) / 20.0
                 + _p(self.p, "w_conceal", i) * self.conceal[idx]
                 + _p(self.p, "w_vent", i) * self.near_vent[idx]
                 - _p(self.p, "w_dist", i) * np.where(fin, from_me, 1e3) / 20.0)
        score = np.where(fin & (from_me > 1.0), score, -np.inf)
        if not np.isfinite(score).any():
            return
        b = int(np.argmax(score))
        self.spot[i] = (float(cxy[b, 0]), float(cxy[b, 1]))
        self.field[i] = self.paths.distance_field([self.spot[i]])
        self.camping[i] = False

    def step(self, x, y, heading, dt, active, seeker_seen_xy: list, danger: np.ndarray,
             blocked: np.ndarray | None = None):
        goal = np.full(self.n, np.nan)
        speed = np.ones(self.n)
        if blocked is None:
            self.blocked_s[:] = 0.0
        else:
            self.blocked_s = np.where(np.asarray(blocked, bool), self.blocked_s + dt, 0.0)
        for i in range(self.n):
            if not active[i]:
                continue
            seen = seeker_seen_xy[i] is not None
            if seen:
                self.known[i] = seeker_seen_xy[i]
            threat = seen or danger[i] >= _p(self.p, "flee_danger", i)
            if threat and self.t >= self.next_threat_plan[i]:
                self._plan(i, x[i], y[i])
                self.next_threat_plan[i] = self.t + 1.0
            elif self.spot[i] is None:
                self._plan(i, x[i], y[i])
            elif self.camping[i] and self.t >= self.camp_until[i]:
                self._plan(i, x[i], y[i])
            elif self.blocked_s[i] >= REPLAN_AFTER_BLOCKED_S:  # scraping a wall: re-pick a spot
                self._plan(i, x[i], y[i])
                self.blocked_s[i] = 0.0
            if self.spot[i] is None:
                continue
            if not self.camping[i] and np.hypot(self.spot[i][0] - x[i], self.spot[i][1] - y[i]) < 0.6:
                self.camping[i] = True
                self.camp_until[i] = self.t + _p(self.p, "camp_s", i)
            if self.camping[i] and not threat:
                speed[i] = 0.0  # camping means standing still, not creeping into the wall
            else:
                goal[i] = lookahead(self.grid, self.field[i], x[i], y[i], _p(self.p, "lookahead_units", i), self.spot[i])
        goal = free_direction(self.grid, x, y, goal, np.full(self.n, AVOID_FREE_UNITS))
        goal = centre_goal(self.grid, x, y, goal, _vec(self.p, "wall_repel", self.n, WALL_REPEL_GAIN))
        goal = self.smooth(goal, dt, _vec(self.p, "goal_smooth_s", self.n, GOAL_SMOOTH_S))
        self.t += dt
        return goal, speed


class RoleController:
    """
    Drives any mix of brain seekers and brain hiders that share one FlyPopulation. Used by both
    the batched role environment and recorded matches, so the two behave identically.

    roles:  per batch column, "seeker" or "hider"
    values: {"seeker": decoded adapter values, "hider": decoded adapter values}; each value
            is a scalar or an array over that role's columns (in column order)
    """

    GAIN_DEFAULTS = {  # channels a role does not use are off, so they cannot leak in
        "seeker": {"target": 1.0, "loom": 0.0, "danger": 0.0, "ping": 1.0},
        "hider": {"target": 0.0, "loom": 1.0, "danger": 1.0, "ping": 0.0},
    }

    def __init__(self, roles: list[str], values: dict, grid: OccupancyGrid, rooms: np.ndarray, paths: GridPaths,
                 vents: list, seeker_start, seed: int = 0):
        self.roles = np.asarray(roles)
        self.n = len(roles)
        self.cols = {r: np.flatnonzero(self.roles == r) for r in ("seeker", "hider")}
        self.values = values
        self.seeker = self.hider = None
        from amongusfly.train.adapter import policy_values
        if len(self.cols["seeker"]):
            self.seeker = SeekerRolePolicy(len(self.cols["seeker"]), grid, paths, rooms,
                                           policy_values(values["seeker"]), seed=seed)
        if len(self.cols["hider"]):
            self.hider = HiderRolePolicy(len(self.cols["hider"]), grid, paths, policy_values(values["hider"]), vents,
                                         seeker_start, seed=seed)

    def _merge(self, prefix: str) -> dict:
        """Per-column arrays for every `prefix:` key used by any role present."""
        keys = sorted({k for r, c in self.cols.items() if len(c) for k in self.values[r] if k.startswith(prefix)})
        out = {}
        for k in keys:
            arr = np.full(self.n, np.nan)
            for r, c in self.cols.items():
                if len(c) and k in self.values[r]:
                    arr[c] = np.broadcast_to(np.asarray(self.values[r][k], float), (len(c),))
            out[k] = arr
        return out

    def decoder_values(self) -> dict:
        merged = self._merge("decoder:")
        return {k: np.where(np.isnan(v), np.nanmean(v), v) for k, v in merged.items()}

    def channel_gains(self, odor_ok: bool = True) -> dict:
        gains = {"photo": 1.0}
        for ch in ("compass", "goal"):
            arr = np.ones(self.n)
            for r, c in self.cols.items():
                if len(c):
                    arr[c] = float(self.values[r].get(f"gain:{ch}", 1.0))
            gains[ch] = arr
        for ch in ("target", "loom", "danger", "ping"):
            arr = np.zeros(self.n)
            for r, c in self.cols.items():
                if len(c):
                    v = self.values[r].get(f"gain:{ch}", self.GAIN_DEFAULTS[r][ch])
                    arr[c] = np.broadcast_to(np.asarray(v, float), (len(c),)) if self.GAIN_DEFAULTS[r][ch] else 0.0
            gains[ch] = arr
        obstacle = np.zeros(self.n)
        c = self.cols.get("seeker", [])
        if len(c):
            obstacle[c] = float(self.values["seeker"].get("gain:obstacle", 0.0))
        gains["obstacle"] = obstacle
        if not odor_ok:  # odour input ignites the mushroom body on full/pruned5
            gains["danger"], gains["ping"] = 0.0, 0.0
        return gains

    def step(self, x, y, heading, dt, active, seen_hider_xy: list, pings: list, seen_seeker_xy: list,
             danger: np.ndarray, blocked: np.ndarray | None = None):
        """
        All per-column inputs are indexed by batch column; seen_hider_xy / pings are only read
        for seeker columns, seen_seeker_xy / danger only for hider columns.
        Returns (goal [n], speed factor [n]).
        """
        goal, speed = np.full(self.n, np.nan), np.ones(self.n)
        x, y, heading, active = map(np.asarray, (x, y, heading, active))
        if self.seeker is not None:
            c = self.cols["seeker"]
            g, s = self.seeker.step(x[c], y[c], heading[c], dt, active[c], [seen_hider_xy[i] for i in c],
                                    [pings[i] for i in c], None if blocked is None else np.asarray(blocked)[c])
            goal[c], speed[c] = g, s
        if self.hider is not None:
            c = self.cols["hider"]
            g, s = self.hider.step(x[c], y[c], heading[c], dt, active[c], [seen_seeker_xy[i] for i in c],
                                   np.asarray(danger)[c], None if blocked is None else np.asarray(blocked)[c])
            goal[c], speed[c] = g, s
        return goal, speed
