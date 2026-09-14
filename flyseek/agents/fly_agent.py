"""
FlyPopulation: a batch of connectome-driven flies sharing one GPU brain.

One tick (default 20 ms of brain time):
    senses (vision, later odor/touch) -> Poisson stimulus on role neurons
    -> run the LIF brain for the tick -> DN spike counts -> motor decoder -> body

Each batch column is one fly. Flies interact only through the world (what they see).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flyseek.brain.lif_torch import LIFBrain, load_config
from flyseek.brain.roles import full_idx_of, role_idx, type_idx
from flyseek.motors.body_kinematic import KinematicBody
from flyseek.motors.decoders import READOUT_TYPES, MotorDecoder
from flyseek.senses.vision import VisionEncoder, VisualObject
from flyseek.world.grid import OccupancyGrid

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

    def _set_stimulus(self, rates: dict):
        neu, cols, rr = [], [], []
        for ch, sides in rates.items():
            gain = self.channel_gain.get(ch, 1.0)
            if gain == 0 or ch not in self.chan_idx:
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
             sensing: np.ndarray | None = None):
        """
        extra_rates: additional channels, e.g. {"danger": {"L": [A], "R": [A]}}.
        base_speed:  per-fly engineered forward speed (overrides config/motors.yaml).
        movable:     per-fly bool; False = frozen (dead, in a vent, or seeker during hide phase).
        sensing:     per-fly bool; False = all sensory input off (e.g. a caught fly).
        """
        dt_s = self.tick_ms / 1000.0
        b = self.body
        vis = self.vision.encode(b.x, b.y, b.heading, objects, self.grid, dt_s)
        rates = dict(vis.rates)
        if extra_rates:
            rates.update(extra_rates)
        if sensing is not None:
            on = np.asarray(sensing, dtype=float)
            rates = {ch: {s: np.asarray(v[s]) * on for s in "LR"} for ch, v in rates.items()}
        self._set_stimulus(rates)

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
