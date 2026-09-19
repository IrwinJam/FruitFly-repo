"""
Hide n Seek rules engine (config/game.yaml + config/skeld_extras.json).

Phases: hide (seeker frozen) -> seek -> final_hide (seeker speed boost, periodic
pings revealing living hiders) -> over. Seeker wins when every hider is dead;
hiders win if any survive the round.

Engineered rule (disclosed): brains can't press "vent", so a hider auto-vents when
it's on a vent, has uses left, and the seeker is within auto_vent_danger_range. It
pops out of a linked vent (the one farthest from the seeker) after max_time_inside_s,
and can't be killed or seen while inside.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import yaml

from flyseek.paths import CONFIG_DIR


def load_game_config(preset: str | None = None) -> dict:
    with open(CONFIG_DIR / "game.yaml") as f:
        cfg = yaml.safe_load(f)
    if preset and preset != "full":  # "full" = the full-length default timers
        cfg["timers"].update(cfg["presets"][preset])
    cfg["preset"] = preset or "full"
    return cfg


def load_extras() -> dict:
    return json.loads((CONFIG_DIR / "skeld_extras.json").read_text())


@dataclass
class RulesOutput:
    speed_mult: np.ndarray  # [A] multiply role speed (0 = can't move)
    visible: np.ndarray  # [A] bool: can be seen/targeted (alive and not in a vent)
    danger: np.ndarray  # [A] 0..1 for hiders, 0 for the seeker
    teleports: dict  # agent -> (x, y)
    ping_positions: list | None  # [(x, y), ...] of living hiders when a ping fires
    events: list = field(default_factory=list)


class HideNSeekRules:
    def __init__(self, roles: list[str], tick_s: float, cfg: dict | None = None, extras: dict | None = None,
                 seed: int = 0):
        self.cfg = cfg or load_game_config()
        self.extras = extras or load_extras()
        self.roles = np.array(roles)
        self.seeker = np.flatnonzero(self.roles == "seeker")
        self.hiders = np.flatnonzero(self.roles == "hider")
        self.tick_s = tick_s
        self.rng = np.random.default_rng(seed)
        n = len(roles)
        self.t = 0.0
        self.tick = 0
        self.phase = "hide"
        self.alive = np.ones(n, dtype=bool)
        self.vent_uses = np.zeros(n, dtype=int)
        self.in_vent_until = np.full(n, -1.0)
        self.pending_exit: dict[int, tuple] = {}
        self.next_ping_t = None
        self.winner = None
        self.vents = self.extras["vents"]

    # ---------------------------------------------------------------- helpers
    @property
    def over(self) -> bool:
        return self.phase == "over"

    def _phase_at(self, t: float) -> str:
        tm = self.cfg["timers"]
        if t < tm["hide_phase_s"]:
            return "hide"
        if t >= tm["round_length_s"]:
            return "over"
        if t >= tm["round_length_s"] - tm["final_hide_s"]:
            return "final_hide"
        return "seek"

    def _linked_exit(self, vent_i: int, seeker_xy) -> tuple:
        v = self.vents[vent_i]
        linked = [j for j, u in enumerate(self.vents) if u["group"] == v["group"] and j != vent_i]
        far = max(linked, key=lambda j: np.hypot(self.vents[j]["x"] - seeker_xy[0], self.vents[j]["y"] - seeker_xy[1]))
        return self.vents[far]["x"], self.vents[far]["y"], far

    # ------------------------------------------------------------------- step
    def step(self, x: np.ndarray, y: np.ndarray) -> RulesOutput:
        c, tm = self.cfg, self.cfg["timers"]
        events = []
        teleports = {}
        ping_positions = None

        new_phase = "over" if self.over else self._phase_at(self.t)
        if new_phase != self.phase:
            events.append({"kind": "phase", "phase": new_phase})
            self.phase = new_phase
            if new_phase == "final_hide":
                self.next_ping_t = self.t

        in_vent = self.in_vent_until > self.t
        # vent exits
        for a in list(self.pending_exit):
            if not in_vent[a]:
                ex, ey, vi = self.pending_exit.pop(a)
                teleports[a] = (ex, ey)
                events.append({"kind": "vent_exit", "agent": int(a), "vent": int(vi)})

        sx, sy = (x[self.seeker[0]], y[self.seeker[0]]) if len(self.seeker) else (np.inf, np.inf)
        seeker_d = np.hypot(x - sx, y - sy)
        visible = self.alive & ~in_vent

        if self.phase in ("seek", "final_hide") and len(self.seeker) and self.alive[self.seeker[0]]:
            # kills
            for h in self.hiders:
                if visible[h] and seeker_d[h] <= c["kill_radius_units"]:
                    self.alive[h] = False
                    visible[h] = False
                    events.append({"kind": "kill", "victim": int(h), "x": float(x[h]), "y": float(y[h])})
            # auto-vent
            vc = c["vents"]
            for h in self.hiders:
                if not visible[h] or self.vent_uses[h] >= vc["max_uses_per_hider"]:
                    continue
                if seeker_d[h] > vc["auto_vent_danger_range_units"]:
                    continue
                vd = [np.hypot(v["x"] - x[h], v["y"] - y[h]) for v in self.vents]
                vi = int(np.argmin(vd))
                if vd[vi] <= vc["use_radius_units"]:
                    self.vent_uses[h] += 1
                    self.in_vent_until[h] = self.t + vc["max_time_inside_s"]
                    self.pending_exit[h] = self._linked_exit(vi, (sx, sy))
                    visible[h] = False
                    events.append({"kind": "vent_enter", "agent": int(h), "vent": vi})

        if self.phase == "final_hide" and self.next_ping_t is not None and self.t >= self.next_ping_t:
            ping_positions = [(float(x[h]), float(y[h])) for h in self.hiders if self.alive[h]]
            events.append({"kind": "ping", "positions": ping_positions})
            self.next_ping_t = self.t + tm["ping_interval_s"]

        # speed multipliers
        mult = np.where(self.alive, 1.0, 0.0)
        mult = np.where(self.in_vent_until > self.t, 0.0, mult)
        if len(self.seeker):
            s = self.seeker[0]
            if self.phase == "hide":
                mult[s] = 0.0
            elif self.phase == "final_hide":
                mult[s] *= tm["final_hide_speed_multiplier"]

        danger = np.zeros(len(self.roles))
        rng_ = c["danger_meter"]["max_range_units"]
        danger[self.hiders] = np.clip(1.0 - seeker_d[self.hiders] / rng_, 0, 1) * self.alive[self.hiders]

        # win conditions
        if self.phase != "over":
            if len(self.hiders) and not self.alive[self.hiders].any():
                self.phase, self.winner = "over", "seeker"
                events.append({"kind": "over", "winner": "seeker"})
        elif self.winner is None:
            self.winner = "hiders"
            events.append({"kind": "over", "winner": "hiders",
                           "survivors": [int(h) for h in self.hiders if self.alive[h]]})

        self.t += self.tick_s
        self.tick += 1
        return RulesOutput(mult, visible, danger, teleports, ping_positions, events)
