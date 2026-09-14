"""
Kinematic "unicycle" body for a batch of flies: position, heading, and wall
collisions with sliding. No legs or physics; Among Us is top-down 2D
(PROJECT_PLAN.md section 3.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyseek.world.grid import OccupancyGrid


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
        self.speed, self.omega = speed, omega
        self.heading = (self.heading + omega * dt_s + np.pi) % (2 * np.pi) - np.pi
        dx = np.cos(self.heading) * speed * dt_s
        dy = np.sin(self.heading) * speed * dt_s

        def clear(px, py):
            return grid.dist_at(px, py) >= self.radius

        full = clear(self.x + dx, self.y + dy)
        only_x = ~full & clear(self.x + dx, self.y)
        only_y = ~full & ~only_x & clear(self.x, self.y + dy)
        self.x = np.where(full | only_x, self.x + dx, self.x)
        self.y = np.where(full | only_y, self.y + dy, self.y)
        self.wall_contact = ~full & (np.abs(dx) + np.abs(dy) > 0)
