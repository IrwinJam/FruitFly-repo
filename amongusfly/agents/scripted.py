"""
Scripted opponents and baselines (no brain):

ScriptedSeeker: chases the nearest hider it can see (or the latest ping during
Final Hide) along geodesic paths; otherwise patrols random waypoints.
ScriptedHider: flees along the geodesic "away from seeker" gradient when the seeker
is visible; otherwise wanders between random waypoints with pauses.
"""
from __future__ import annotations

import numpy as np

from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.pathing import GridPaths


def line_of_sight(grid: OccupancyGrid, x0, y0, x1, y1) -> bool:
    d = float(np.hypot(x1 - x0, y1 - y0))
    if d < 1e-6:
        return True
    wall = grid.raycast(np.array([x0]), np.array([y0]), np.array([[np.arctan2(y1 - y0, x1 - x0)]]), d)[0, 0]
    return wall >= d - grid.res


def steer(heading: float, desired: float | None, max_omega: float, dt: float) -> float:
    if desired is None:
        return 0.0
    diff = (desired - heading + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(diff / dt, -max_omega, max_omega))


class ScriptedSeeker:
    REPATH_TICKS = 5

    def __init__(self, paths: GridPaths, rng: np.random.Generator, vision_range: float):
        self.paths, self.rng, self.range = paths, rng, vision_range
        self.goal = None
        self.field = None
        self.tick = 0
        self.last_ping: list | None = None

    def act(self, me: int, x, y, heading, visible, hider_idx, dt: float, ping=None):
        g = self.paths.grid
        self.tick += 1
        if ping:
            self.last_ping = ping
        seen = [h for h in hider_idx if visible[h]
                and np.hypot(x[h] - x[me], y[h] - y[me]) <= self.range
                and line_of_sight(g, x[me], y[me], x[h], y[h])]
        if seen:
            tgt = min(seen, key=lambda h: np.hypot(x[h] - x[me], y[h] - y[me]))
            goal = (float(x[tgt]), float(y[tgt]))
            if np.hypot(goal[0] - x[me], goal[1] - y[me]) < 1.0:
                return 1.0, steer(heading, float(np.arctan2(goal[1] - y[me], goal[0] - x[me])), 6.0, dt)
            if self.field is None or self.tick % self.REPATH_TICKS == 0 or self.goal is None:
                self.goal, self.field = goal, self.paths.distance_field([goal])
        elif self.last_ping:
            goal = min(self.last_ping, key=lambda p: np.hypot(p[0] - x[me], p[1] - y[me]))
            if self.goal != goal:
                self.goal, self.field = goal, self.paths.distance_field([goal])
            if np.hypot(goal[0] - x[me], goal[1] - y[me]) < 1.0:
                self.last_ping = [p for p in self.last_ping if p != goal] or None
        else:
            if self.goal is None or np.hypot(self.goal[0] - x[me], self.goal[1] - y[me]) < 1.0:
                p = g.random_walkable_points(1, self.rng, min_clearance=0.4)[0]
                self.goal, self.field = (float(p[0]), float(p[1])), self.paths.distance_field([tuple(p)])
        desired = self.paths.descend(self.field, x[me], y[me])
        return 1.0, steer(heading, desired, 6.0, dt)


class ScriptedHider:
    REPATH_TICKS = 5

    def __init__(self, paths: GridPaths, rng: np.random.Generator, vision_range: float):
        self.paths, self.rng, self.range = paths, rng, vision_range
        self.goal = None
        self.field = None
        self.flee_field = None
        self.pause_until = 0.0
        self.t = 0.0
        self.tick = 0

    def act(self, me: int, x, y, heading, seeker: int, seeker_visible: bool, dt: float):
        g = self.paths.grid
        self.t += dt
        self.tick += 1
        d = np.hypot(x[seeker] - x[me], y[seeker] - y[me])
        if seeker_visible and d <= self.range and line_of_sight(g, x[me], y[me], x[seeker], y[seeker]):
            if self.flee_field is None or self.tick % self.REPATH_TICKS == 0:
                self.flee_field = self.paths.distance_field([(float(x[seeker]), float(y[seeker]))])
            self.goal = None
            return 1.0, steer(heading, self.paths.descend(self.flee_field, x[me], y[me], ascend=True), 6.0, dt)
        self.flee_field = None
        if self.t < self.pause_until:
            return 0.0, 0.0
        if self.goal is None or np.hypot(self.goal[0] - x[me], self.goal[1] - y[me]) < 0.8:
            if self.goal is not None:
                self.pause_until = self.t + float(self.rng.uniform(3, 10))
            # hiding spot: of a few random candidates, the one geodesically farthest from the seeker
            cand = g.random_walkable_points(8, self.rng, min_clearance=0.4)
            from_seeker = self.paths.distance_field([(float(x[seeker]), float(y[seeker]))])
            far = self.paths.field_at(from_seeker, cand[:, 0], cand[:, 1])
            far = np.where(np.isfinite(far), far, -1)
            p = cand[int(np.argmax(far))]
            self.goal, self.field = (float(p[0]), float(p[1])), self.paths.distance_field([tuple(p)])
        return 1.0, steer(heading, self.paths.descend(self.field, x[me], y[me]), 6.0, dt)
