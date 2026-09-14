"""
Vision encoder: world -> Poisson rates for visual roles, per fly and per side.

Channels (rates in Hz; caps come from docs/PHASE1_REPORT.md):
  target  -> LC10a (target_motion_detector). Any visible "target" object; rate grows
             with angular size, capped at 25 Hz (pursuit is only wiring-specific
             vs shuffles up to 25 Hz).
  loom    -> LC4 + LPLC2. Rate grows with the angular EXPANSION speed of a visible
             "threat" object, capped at 100 Hz.
  photo   -> photoreceptors. Low-rate hemifield brightness from wall distance.

Side assignment: relative bearing > 0 (object to the fly's left) drives the fly's
left-side neurons; objects within +/- binocular_half_deg drive both sides.
This is side-level coding, not retinotopic: a documented simplification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyseek.world.grid import OccupancyGrid


@dataclass
class VisualObject:
    x: np.ndarray  # [A] position of this object as seen by each fly (per-fly objects allowed)
    y: np.ndarray
    radius: float
    kind: str  # "target" or "threat"
    visible_to: np.ndarray | None = None  # [A] bool mask, default all


@dataclass
class VisionConfig:
    fov_deg: float = 270.0
    n_rays: int = 64
    max_range: float = 12.0
    binocular_half_deg: float = 5.0
    target_max_hz: float = 25.0
    target_ref_deg: float = 20.0  # angular size giving max rate
    loom_max_hz: float = 100.0
    loom_ref_deg_per_s: float = 60.0  # expansion speed giving max rate
    photo_max_hz: float = 10.0


@dataclass
class VisionOutput:
    rates: dict  # rates[channel][side] -> [A] Hz
    ray_dist: np.ndarray  # [A, R] for logging/visualization
    target_bearing: np.ndarray  # [A] relative bearing (rad) of the strongest target, nan if none


class VisionEncoder:
    def __init__(self, n_flies: int, cfg: VisionConfig | None = None):
        self.cfg = cfg or VisionConfig()
        self.n = n_flies
        half = np.deg2rad(self.cfg.fov_deg / 2)
        self.ray_rel = np.linspace(-half, half, self.cfg.n_rays)
        self._prev_threat_size: dict[int, np.ndarray] = {}

    def reset(self):
        self._prev_threat_size = {}

    def encode(self, x, y, heading, objects: list[VisualObject], grid: OccupancyGrid, dt_s: float) -> VisionOutput:
        c, A = self.cfg, self.n
        rays_abs = heading[:, None] + self.ray_rel[None, :]
        ray_dist = grid.raycast(x, y, rays_abs, c.max_range)

        zeros = lambda: np.zeros(A)
        rates = {ch: {"L": zeros(), "R": zeros()} for ch in ("target", "loom", "photo")}
        best_target_strength = np.zeros(A)
        target_bearing = np.full(A, np.nan)

        half_fov = np.deg2rad(c.fov_deg / 2)
        bino = np.deg2rad(c.binocular_half_deg)
        for oi, obj in enumerate(objects):
            dx, dy = obj.x - x, obj.y - y
            dist = np.hypot(dx, dy)
            bearing = (np.arctan2(dy, dx) - heading + np.pi) % (2 * np.pi) - np.pi
            ang_size = 2 * np.arctan2(obj.radius, np.maximum(dist, 1e-6))
            # occlusion: wall distance along the object's bearing must exceed the object distance
            wall = grid.raycast(x, y, (heading + bearing)[:, None], c.max_range)[:, 0]
            visible = (np.abs(bearing) <= half_fov) & (dist <= c.max_range) & (wall + obj.radius >= dist)
            if obj.visible_to is not None:
                visible &= obj.visible_to
            left = visible & (bearing > -bino)
            right = visible & (bearing < bino)

            if obj.kind == "target":
                strength = np.clip(ang_size / np.deg2rad(c.target_ref_deg), 0, 1) * visible
                r = c.target_max_hz * strength
                rates["target"]["L"] = np.maximum(rates["target"]["L"], np.where(left, r, 0))
                rates["target"]["R"] = np.maximum(rates["target"]["R"], np.where(right, r, 0))
                better = strength > best_target_strength
                target_bearing = np.where(better, bearing, target_bearing)
                best_target_strength = np.maximum(best_target_strength, strength)
            elif obj.kind == "threat":
                prev = self._prev_threat_size.get(oi)
                # no expansion on the first frame an object becomes visible (prev == 0),
                # otherwise appearing from behind a wall reads as an explosive loom
                expansion = np.zeros(A) if prev is None else np.where(prev > 0, (ang_size - prev) / dt_s, 0.0)
                self._prev_threat_size[oi] = np.where(visible, ang_size, 0.0)
                r = c.loom_max_hz * np.clip(expansion / np.deg2rad(c.loom_ref_deg_per_s), 0, 1)
                rates["loom"]["L"] = np.maximum(rates["loom"]["L"], np.where(left, r, 0))
                rates["loom"]["R"] = np.maximum(rates["loom"]["R"], np.where(right, r, 0))

        bright = 1.0 - ray_dist / c.max_range
        rates["photo"]["L"] = c.photo_max_hz * bright[:, self.ray_rel > 0].mean(axis=1)
        rates["photo"]["R"] = c.photo_max_hz * bright[:, self.ray_rel < 0].mean(axis=1)
        return VisionOutput(rates, ray_dist, target_bearing)
