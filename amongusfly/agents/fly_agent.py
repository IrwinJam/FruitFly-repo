"""
A batch of connectome-driven flies sharing one GPU brain.

Each tick (20 ms of brain time): senses -> Poisson input on sensory neurons -> LIF brain ->
descending-neuron spike counts -> motor decoder -> body. Each batch column is one fly; flies
interact only through the world.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from amongusfly.brain.cx_test import cx_neurons
from amongusfly.brain.lif_torch import LIFBrain, load_config
from amongusfly.paths import CONFIG_DIR
from amongusfly.brain.roles import full_idx_of, role_idx, type_idx
from amongusfly.motors.body_kinematic import KinematicBody
from amongusfly.motors.decoders import READOUT_TYPES, MotorDecoder
from amongusfly.senses.vision import VisionEncoder, VisualObject
from amongusfly.world.grid import OccupancyGrid

# sensory channel -> brain roles it drives
CHANNEL_ROLES = {
    "target": ["target_motion_detector"],
    "loom": ["looming_expansion", "looming_size"],
    "photo": ["photoreceptor_achromatic", "photoreceptor_color"],
    "danger": ["aversive_odor"],  # hider danger meter (navcore only: odor ignites the MB on full/pruned5)
    "ping": ["attractive_odor"],  # seeker Final Hide pings, lateralized by bearing
}


@dataclass
class TickResult:
    vision_rates: dict
    dn_hz: dict
    command: object


class FlyPopulation:
    def __init__(self, tag: str, x, y, heading, grid: OccupancyGrid, seed: int = 0,
                 channel_gain: dict | None = None, brain: LIFBrain | None = None):
        self.tag = tag
        self.grid = grid
        self.n = len(x)
        self.brain = brain or LIFBrain(tag=tag)
        self.brain.reset(self.n, seed=seed)
        self.tick_ms = load_config()["sim"]["brain_ms_per_game_tick"]
        self.steps_per_tick = int(round(self.tick_ms / self.brain.dt_ms))

        self.body = KinematicBody(np.asarray(x, float), np.asarray(y, float), np.asarray(heading, float))
        self.vision = VisionEncoder(self.n)
        self.decoder = MotorDecoder(self.n)
        self.channel_gain = channel_gain or {ch: 1.0 for ch in CHANNEL_ROLES}

        # neuron indices per (channel, side), in this graph's index space
        self.chan_idx = {
            ch: {s: np.array(sorted({i for r in roles for i in role_idx(r, s, graph=tag)}), dtype=np.int64)
                 for s in "LR"}
            for ch, roles in CHANNEL_ROLES.items()
        }
        # DN readout indices
        self.readout = {t: {s: np.array(type_idx(t, s, graph=tag), dtype=np.int64) for s in "LR"} for t in READOUT_TYPES}
        self.full_idx = full_idx_of(tag)

        # central-complex compass (EPG heading bump) and goal (FC2 bump)
        with open(CONFIG_DIR / "cx.yaml") as f:
            self.cx_cfg = yaml.safe_load(f)
        cx = cx_neurons(tag)
        self.epg_idx = np.array([i for i, _ in cx["EPG"]], dtype=np.int64)
        self.epg_phase = np.array([p for _, p in cx["EPG"]])
        self.fc2_idx = np.array([i for i, _ in cx["FC2"]], dtype=np.int64)
        self.fc2_phase = np.array([p for _, p in cx["FC2"]])

        # obstacle sense (config/obstacle.yaml): off unless channel_gain["obstacle"] > 0
        with open(CONFIG_DIR / "obstacle.yaml") as f:
            self.obstacle_cfg = yaml.safe_load(f)
        t = self.obstacle_cfg["target_type"]
        self.chan_idx["obstacle"] = {s: np.array(type_idx(t, s, graph=tag), dtype=np.int64) for s in "LR"}

    def _obstacle_rates(self) -> dict:
        """Side-level wall proximity from a fan of rays, as Poisson rates for the obstacle channel.

        Proximity per side = max over that side's rays of clip((range - d) / (range - body radius), 0, 1).
        With laterality "contra" a wall on the left drives the right-side cells, so the fly turns away.
        """
        c, b = self.obstacle_cfg, self.body
        rays = np.deg2rad(np.asarray(c["ray_deg"], float))
        ang = np.concatenate([b.heading[:, None] + rays[None, :], b.heading[:, None] - rays[None, :]], axis=1)
        d = self.grid.raycast(b.x, b.y, ang, c["range_units"])
        prox = np.clip((c["range_units"] - d) / (c["range_units"] - b.radius), 0.0, 1.0)
        left, right = prox[:, :len(rays)].max(axis=1), prox[:, len(rays):].max(axis=1)
        if c["laterality"] == "contra":
            left, right = right, left
        return {"L": c["max_hz"] * left, "R": c["max_hz"] * right}

    def _cx_rates(self, heading: np.ndarray, goal: np.ndarray | None, sensing: np.ndarray | None):
        """Returns (neurons, cols, rates) for the EPG heading bump and FC2 goal bump."""
        c = self.cx_cfg
        # optional overrides for bump sharpness and peak rate (config/cx.yaml holds the defaults)
        if "AMONGUSFLY_CX_KAPPA" in os.environ or "AMONGUSFLY_CX_RMAX" in os.environ:
            c = {**c, "kappa": float(os.environ.get("AMONGUSFLY_CX_KAPPA", c["kappa"])),
                 "r_max_hz": float(os.environ.get("AMONGUSFLY_CX_RMAX", c["r_max_hz"]))}
        if goal is None:  # no goal system in use: no compass or goal input at all
            return [], [], []
        on = np.ones(self.n, bool) if sensing is None else np.asarray(sensing, bool)
        parts = []
        sign = c.get("heading_sign", 1)
        if np.any(np.asarray(self.channel_gain.get("compass", 1.0)) > 0) and len(self.epg_idx):
            h_nom = sign * heading
            r = c["r_max_hz"] * np.exp(c["kappa"] * (np.cos(self.epg_phase[None, :] - h_nom[:, None]) - 1))
            parts.append((self.epg_idx, r * on[:, None] * np.reshape(self.channel_gain.get("compass", 1.0), (-1, 1))))
        if goal is not None and np.any(np.asarray(self.channel_gain.get("goal", 1.0)) > 0) and len(self.fc2_idx):
            has = on & np.isfinite(goal)
            goal = np.where(has, goal, 0.0)
            # Optional goal-error gain: place the goal bump at an exaggerated heading error (k = 1 is off).
            k = float(os.environ.get("AMONGUSFLY_GOAL_GAIN", c.get("goal_error_gain", 1.0)))
            if k != 1.0:
                err = (goal - heading + np.pi) % (2 * np.pi) - np.pi
                # amplify only errors the pathway cannot see; leave larger ones alone, so a big
                # error is never pushed past the fly's side (which would reverse the turn)
                lim = np.deg2rad(float(os.environ.get("AMONGUSFLY_GOAL_GAIN_LIMIT",
                                                      c.get("goal_error_amplify_to_deg", 45.0))))
                amp = np.minimum(k * np.abs(err), np.maximum(np.abs(err), lim))
                goal = heading + np.sign(err) * amp
            g_nom = sign * goal + np.deg2rad(c["goal_offset_deg"])
            r = c["r_max_hz"] * np.exp(c["kappa"] * (np.cos(self.fc2_phase[None, :] - g_nom[:, None]) - 1))
            parts.append((self.fc2_idx, r * has[:, None] * np.reshape(self.channel_gain.get("goal", 1.0), (-1, 1))))
        neu, cols, rr = [], [], []
        for idx, rates in parts:
            keep = rates >= 1.0
            ff, nn = np.nonzero(keep)
            neu.append(idx[nn]); cols.append(ff); rr.append(rates[ff, nn])
        return neu, cols, rr

    def _set_stimulus(self, rates: dict, extra: tuple | None = None):
        neu, cols, rr = ([], [], []) if extra is None else (list(extra[0]), list(extra[1]), list(extra[2]))
        for ch, sides in rates.items():
            gain = self.channel_gain.get(ch, 1.0)  # scalar or per-fly [A]
            if not np.any(np.asarray(gain) != 0) or ch not in self.chan_idx:
                continue
            for s in "LR":
                idx = self.chan_idx[ch][s]
                if len(idx) == 0:
                    continue
                per_fly = sides[s] * gain
                active = np.flatnonzero(per_fly > 0)
                if len(active) == 0:
                    continue
                neu.append(np.tile(idx, len(active)))
                cols.append(np.repeat(active, len(idx)))
                rr.append(np.repeat(per_fly[active], len(idx)))
        if neu:
            self.brain.set_stimulus(np.concatenate(neu), np.concatenate(cols), np.concatenate(rr))
        else:
            self.brain.clear_stimulus()

    def tick(self, objects: list[VisualObject], record_all_spikes: bool = False, extra_rates: dict | None = None,
             base_speed: np.ndarray | None = None, movable: np.ndarray | None = None,
             sensing: np.ndarray | None = None, goal_angle: np.ndarray | None = None):
        """
        extra_rates: additional channels, e.g. {"danger": {"L": [A], "R": [A]}}.
        base_speed:  per-fly forward speed (overrides config/motors.yaml).
        movable:     per-fly bool; False = frozen (dead, in a vent, or seeker during hide phase).
        sensing:     per-fly bool; False = all sensory input off (e.g. a caught fly).
        goal_angle:  per-fly WORLD goal direction (rad) for the FC2 goal bump; nan = no goal.
                     The EPG compass bump always tracks the body heading (idealized compass).
        """
        dt_s = self.tick_ms / 1000.0
        b = self.body
        vis = self.vision.encode(b.x, b.y, b.heading, objects, self.grid, dt_s)
        rates = dict(vis.rates)
        if extra_rates:
            rates.update(extra_rates)
        if np.any(np.asarray(self.channel_gain.get("obstacle", 0.0)) > 0):
            rates["obstacle"] = self._obstacle_rates()
        if sensing is not None:
            on = np.asarray(sensing, dtype=float)
            rates = {ch: {s: np.asarray(v[s]) * on for s in "LR"} for ch, v in rates.items()}
        self._set_stimulus(rates, extra=self._cx_rates(b.heading, goal_angle, sensing))

        out = self.brain.run(self.steps_per_tick, count_neurons=None if record_all_spikes else self._readout_tensor())
        counts = out["counts"]
        if record_all_spikes:
            all_counts = counts.cpu().numpy()
            dn_counts = {t: {s: all_counts[self.readout[t][s]] for s in "LR"} for t in READOUT_TYPES}
        else:
            all_counts = None
            c = counts.cpu().numpy()
            dn_counts, k = {}, 0
            for t in READOUT_TYPES:
                dn_counts[t] = {}
                for s in "LR":
                    m = len(self.readout[t][s])
                    dn_counts[t][s] = c[k:k + m]
                    k += m

        dn_hz = {t: {s: (dn_counts[t][s].mean(axis=0) if len(dn_counts[t][s]) else np.zeros(self.n)) / dt_s
                     for s in "LR"} for t in READOUT_TYPES}
        self.decoder.update_rates(dn_hz, self.tick_ms)
        cmd = self.decoder.decode(base_speed=base_speed)
        if movable is not None:
            cmd.speed = np.where(movable, cmd.speed, 0.0)
            cmd.omega = np.where(movable, cmd.omega, 0.0)
        b.step(cmd.speed, cmd.omega, dt_s, self.grid)
        return TickResult(rates, dn_hz, cmd), all_counts

    def _readout_tensor(self) -> torch.Tensor:
        if not hasattr(self, "_ro_t"):
            flat = np.concatenate([self.readout[t][s] for t in READOUT_TYPES for s in "LR"])
            self._ro_t = torch.as_tensor(flat, dtype=torch.int64, device=self.brain.device)
        return self._ro_t

    def state(self) -> dict:
        b = self.body
        return {"x": b.x, "y": b.y, "heading": b.heading, "speed": b.speed, "omega": b.omega,
                "alive": np.ones(self.n)}
