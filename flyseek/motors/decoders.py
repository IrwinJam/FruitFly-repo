"""
Motor decoder: EMA-smoothed descending-neuron rates -> (speed, omega) per fly.
Config and the reasoning behind each choice: config/motors.yaml.

Any numeric config value may be a scalar or a per-fly array of shape [A]; that is how
training evaluates a whole population of adapter settings in one GPU batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from flyseek.paths import CONFIG_DIR

READOUT_TYPES = ["DNa02", "DNa03", "DNg13", "DNa01", "DNp01", "MDN", "DNg100", "DNp09", "PFL2"]


def load_motor_config() -> dict:
    with open(CONFIG_DIR / "motors.yaml") as f:
        return yaml.safe_load(f)


@dataclass
class DecodedCommand:
    speed: np.ndarray
    omega: np.ndarray
    dash: np.ndarray
    backward: np.ndarray
    turn_drive: np.ndarray  # pre-tanh weighted L-R, for logging


class MotorDecoder:
    def __init__(self, n_flies: int, cfg: dict | None = None):
        self.cfg = cfg or load_motor_config()
        self.n = n_flies
        # rates[type][side] -> [A] smoothed Hz
        self.rates = {t: {"L": np.zeros(n_flies), "R": np.zeros(n_flies)} for t in READOUT_TYPES}

    def update_rates(self, instant_hz: dict, dt_ms: float):
        """instant_hz[type][side] -> [A] rate over the last tick (Hz)."""
        a = 1.0 - np.exp(-dt_ms / np.asarray(self.cfg["ema_tau_ms"], dtype=float))
        for t in READOUT_TYPES:
            for s in "LR":
                if t in instant_hz and s in instant_hz[t]:
                    self.rates[t][s] += a * (instant_hz[t][s] - self.rates[t][s])

    def decode(self, weights: dict | None = None, base_speed: np.ndarray | float | None = None) -> DecodedCommand:
        c = self.cfg
        w = weights or c["turn"]["weights"]
        drive = np.zeros(self.n)
        for t, wt in w.items():
            if np.any(np.asarray(wt) != 0):
                drive += wt * (self.rates[t]["L"] - self.rates[t]["R"])
        drive = np.where(np.abs(drive) < c["turn"]["deadband_hz"], 0.0, drive)
        omega = np.tanh(drive / c["turn"]["scale_hz"]) * c["turn"]["max_omega_rad_per_s"]

        base = c["forward"]["base_speed_units_per_s"] if base_speed is None else base_speed
        speed = np.full(self.n, base, dtype=float) if np.isscalar(base) else np.asarray(base, float).copy()
        ds = c["dash"]
        dash = (self.rates[ds["source"]]["L"] + self.rates[ds["source"]]["R"]) / 2 > ds["threshold_hz"]
        speed = np.where(dash, speed * ds["speed_multiplier"], speed)
        bk = c["backward"]
        back = (self.rates[bk["source"]]["L"] + self.rates[bk["source"]]["R"]) / 2 > bk["threshold_hz"]
        speed = np.where(back, bk["speed_units_per_s"], speed)

        # goal behind (Phase 4): PFL3 steering fades to zero when the goal is behind the
        # fly; PFL2 (bilateral) is high there. When PFL2 exceeds threshold, turn in the
        # direction of the residual steering drive (left if none) and slow down.
        gb = c.get("goal_behind")
        if gb and np.any(gb.get("enabled", False)):
            pfl2 = (self.rates[gb["source"]]["L"] + self.rates[gb["source"]]["R"]) / 2
            behind = (pfl2 > gb["threshold_hz"]) & np.asarray(gb.get("enabled", False), dtype=bool)
            direction = np.where(drive < 0, -1.0, 1.0)
            omega = np.where(behind, np.clip(omega + direction * gb["turn_rad_per_s"],
                                             -c["turn"]["max_omega_rad_per_s"], c["turn"]["max_omega_rad_per_s"]), omega)
            speed = np.where(behind, speed * gb["speed_factor"], speed)

        speed = np.clip(speed, bk["speed_units_per_s"], c["forward"]["max_speed_units_per_s"])
        return DecodedCommand(speed, omega, dash, back, drive)
