"""
Role goal policies for Phase 5 (engineered, disclosed; weights learned by CMA-ES).

Like the Phase 4 route policy, these decide WHERE a fly goes, using only information a
player of that role has in Among Us; the connectome decides how to steer there (world
goal direction -> FC2 goal bump, EPG compass bump -> PFL3 -> DNs). Sensory pathways
(LC10a target, LC4/LPLC2 loom, aversive odor for the danger meter) drive the brain in
parallel; their gains are trainable too (see flyseek/train/adapter.py).

SeekerRolePolicy
  - a hider in sight (<= vision range, line of sight): chase it along the walkable path
  - otherwise go to where a hider was last seen, for up to `memory_s`
  - otherwise, during Final Hide, go to the nearest ping if `ping_follow` > 0.5
  - otherwise explore with the Phase 4 route planner

HiderRolePolicy
  - picks hiding spots from random candidates, scoring
        w_far     * walkable distance from the last known seeker position / 20
      + w_conceal * concealment (how little open space is in view from the spot)
      + w_vent    * a vent within 2 units
      - w_dist    * walkable distance from the hider / 20
    (before the seeker is ever seen, "last known" = the Cafeteria spawn, where it starts)
  - on arrival, camps for `camp_s` seconds at `camp_speed` x the forward speed, no goal
  - under threat (seeker in sight, or danger meter >= `flee_danger`): re-plans at most
    once per second, which, with w_far, moves it away from the seeker
All parameters may be per-fly arrays [A].
"""
from __future__ import annotations

import numpy as np

from flyseek.agents.route_policy import AVOID_FREE_UNITS, RouteGoalPolicy, free_direction, lookahead
from flyseek.world.grid import OccupancyGrid
from flyseek.world.pathing import GridPaths

_CONCEAL_CACHE: dict = {}


def _vec(params: dict, key: str, n: int) -> np.ndarray:
    v = params.get(key, AVOID_FREE_UNITS)
    return np.full(n, float(v)) if not np.ndim(v) else np.asarray(v, float)


def _p(params: dict, key: str, i: int) -> float:
    v = params[key]
    return float(v[i]) if np.ndim(v) else float(v)


class SeekerRolePolicy:
    def __init__(self, n: int, grid: OccupancyGrid, paths: GridPaths, rooms: np.ndarray, params: dict, seed: int = 0):
        self.n, self.grid, self.paths, self.p = n, grid, paths, params
        self.route = RouteGoalPolicy(n, grid, paths, rooms, params, seed=seed)
        self.last_seen: list = [None] * n
        self.last_seen_t = np.full(n, -np.inf)
        self.chase_target: list = [None] * n
        self.chase_field: list = [None] * n
        self.pings: list = [None] * n
        self.t = 0.0

    def _field_to(self, i, target):
        old = self.chase_target[i]
        if old is None or np.hypot(old[0] - target[0], old[1] - target[1]) > 0.5:
            self.chase_target[i] = target
            self.chase_field[i] = self.paths.distance_field([target])
        return self.chase_field[i]

    def step(self, x, y, heading, dt, active, seen_xy: list, pings: list | None = None):
        """seen_xy[i]: (x, y) of the nearest hider in sight or None. pings[i]: list of ping positions or None."""
        goal = self.route.step(x, y, heading, dt, active)
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
        goal = free_direction(self.grid, x, y, goal, _vec(self.p, "avoid_free_units", self.n))
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

    def step(self, x, y, heading, dt, active, seeker_seen_xy: list, danger: np.ndarray):
        goal = np.full(self.n, np.nan)
        speed = np.ones(self.n)
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
            if self.spot[i] is None:
                continue
            if not self.camping[i] and np.hypot(self.spot[i][0] - x[i], self.spot[i][1] - y[i]) < 0.6:
                self.camping[i] = True
                self.camp_until[i] = self.t + _p(self.p, "camp_s", i)
            if self.camping[i] and not threat:
                speed[i] = _p(self.p, "camp_speed", i)
            else:
                goal[i] = lookahead(self.grid, self.field[i], x[i], y[i], _p(self.p, "lookahead_units", i), self.spot[i])
        goal = free_direction(self.grid, x, y, goal, _vec(self.p, "avoid_free_units", self.n))
        self.t += dt
        return goal, speed


class RoleController:
    """
    Drives any mix of brain seekers and brain hiders that share one FlyPopulation
    (Phase 6). Used by the batched role environment (training / evaluation) and by
    recorded matches, so showcase matches behave exactly like evaluated ones.

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
        from flyseek.train.adapter import policy_values
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
        gains = {"photo": 1.0, "compass": 1.0, "goal": 1.0}
        for ch in ("target", "loom", "danger", "ping"):
            arr = np.zeros(self.n)
            for r, c in self.cols.items():
                if len(c):
                    v = self.values[r].get(f"gain:{ch}", self.GAIN_DEFAULTS[r][ch])
                    arr[c] = np.broadcast_to(np.asarray(v, float), (len(c),)) if self.GAIN_DEFAULTS[r][ch] else 0.0
            gains[ch] = arr
        if not odor_ok:  # mushroom-body odor ignition on full/pruned5 (PHASE1_REPORT 6.3)
            gains["danger"], gains["ping"] = 0.0, 0.0
        return gains

    def step(self, x, y, heading, dt, active, seen_hider_xy: list, pings: list, seen_seeker_xy: list,
             danger: np.ndarray):
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
                                    [pings[i] for i in c])
            goal[c], speed[c] = g, s
        if self.hider is not None:
            c = self.cols["hider"]
            g, s = self.hider.step(x[c], y[c], heading[c], dt, active[c], [seen_seeker_xy[i] for i in c],
                                   np.asarray(danger)[c])
            goal[c], speed[c] = g, s
        return goal, speed
