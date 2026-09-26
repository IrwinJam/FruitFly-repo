"""
Kinematic "unicycle" body for a batch of flies: position, heading, and wall
collisions with sliding. No legs or physics; Among Us is top-down 2D.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amongusfly.world.grid import OccupancyGrid


# Forward speed is scaled down near geometry: full speed at CAREFUL_CLEARANCE or more of wall
# clearance, CAREFUL_MIN_FACTOR of it at the body radius, so a fly does not jam into corners.
CAREFUL_CLEARANCE = 0.55
CAREFUL_MIN_FACTOR = 0.45


@dataclass
class KinematicBody:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    radius: float = 0.25
    speed: np.ndarray = field(default=None)
    omega: np.ndarray = field(default=None)
    wall_contact: np.ndarray = field(default=None)  # [A] bool, set on the last step

    def __post_init__(self):
        n = len(self.x)
        self.speed = np.zeros(n) if self.speed is None else self.speed
        self.omega = np.zeros(n) if self.omega is None else self.omega
        self.wall_contact = np.zeros(n, dtype=bool)

    @property
    def n(self) -> int:
        return len(self.x)

    def step(self, speed: np.ndarray, omega: np.ndarray, dt_s: float, grid: OccupancyGrid):
        """Integrate heading, then move with collision + sliding along walls."""
        self.heading = (self.heading + omega * dt_s + np.pi) % (2 * np.pi) - np.pi
        room = np.clip((grid.dist_at(self.x, self.y) - self.radius) / (CAREFUL_CLEARANCE - self.radius), 0.0, 1.0)
        speed = speed * (CAREFUL_MIN_FACTOR + (1.0 - CAREFUL_MIN_FACTOR) * room)
        self.speed, self.omega = speed, omega
        dx = np.cos(self.heading) * speed * dt_s
        dy = np.sin(self.heading) * speed * dt_s

        def clear(px, py):
            return grid.dist_at(px, py) >= self.radius

        # A body closer to a wall than its own radius (e.g. after a vent exit) would have every
        # direction blocked; nudge it toward more open space instead.
        wedged = ~clear(self.x, self.y)
        if wedged.any():
            ang = np.arange(8) * (np.pi / 4)
            px = self.x[:, None] + np.cos(ang)[None, :] * grid.res * 2
            py = self.y[:, None] + np.sin(ang)[None, :] * grid.res * 2
            best = ang[np.argmax(grid.dist_at(px, py), axis=1)]
            self.x = np.where(wedged, self.x + np.cos(best) * grid.res, self.x)
            self.y = np.where(wedged, self.y + np.sin(best) * grid.res, self.y)

        full = clear(self.x + dx, self.y + dy)
        only_x = ~full & clear(self.x + dx, self.y)
        only_y = ~full & ~only_x & clear(self.x, self.y + dy)
        self.x = np.where(full | only_x, self.x + dx, self.x)
        self.y = np.where(full | only_y, self.y + dy, self.y)
        self.wall_contact = ~full & (np.abs(dx) + np.abs(dy) > 0)
