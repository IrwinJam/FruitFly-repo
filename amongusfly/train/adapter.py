"""
Trainable adapter: the parameters around the frozen connectome (readout, planner, role
policies, sensory gains). CMA-ES searches a normalised [0, 1] vector; `decode` maps it to
bounded physical values (log scale where marked). No connectome weight is ever changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str  # "decoder:<dotted cfg path>" or "policy:<key>"
    lo: float
    hi: float
    init: float
    log: bool = False


EXPLORE_PARAMS = [
    # readout bounds keep training away from a frantic readout that spins in long matches
    Param("decoder:turn.scale_hz", 15.0, 120.0, 20.0, log=True),
    Param("decoder:ema_tau_ms", 80.0, 250.0, 100.0, log=True),
    Param("decoder:turn.weights.DNa02", 0.0, 2.0, 1.0),
    Param("decoder:turn.weights.DNa03", 0.0, 2.0, 0.5),
    Param("decoder:turn.weights.DNg13", -1.0, 1.0, 0.5),
    Param("decoder:goal_behind.threshold_hz", 15.0, 28.0, 20.0),
    Param("decoder:goal_behind.turn_rad_per_s", 0.0, 3.0, 2.0),
    Param("decoder:goal_behind.speed_factor", 0.1, 1.0, 0.4),
    Param("decoder:turn.max_omega_rad_per_s", 3.0, 7.0, 6.0),
    Param("policy:w_free", -2.0, 4.0, 1.0),
    Param("policy:w_novel", -2.0, 4.0, 1.5),
    Param("policy:w_align", -2.0, 4.0, 0.5),
    Param("policy:w_keep", -2.0, 4.0, 0.5),
    Param("policy:decide_s", 0.3, 5.0, 1.5, log=True),
]


_DECODER_PARAMS = [p for p in EXPLORE_PARAMS if p.name.startswith("decoder:")]

# Route planner (amongusfly/agents/route_policy.py)
ROUTE_PARAMS = _DECODER_PARAMS + [
    Param("policy:w_novel", 0.0, 4.0, 2.0),
    Param("policy:w_room", 0.0, 6.0, 3.0),
    Param("policy:w_dist", 0.0, 4.0, 1.0),
    Param("policy:replan_s", 0.5, 10.0, 3.0, log=True),
    Param("policy:lookahead_units", 0.5, 4.0, 1.5, log=True),
]

# Role adapters (amongusfly/agents/role_policy.py); decoder and route values usually start from a
# trained walker (es.py --init-from)
SEEKER_PARAMS = ROUTE_PARAMS + [
    Param("gain:target", 0.0, 2.0, 1.0),  # LC10a channel
    Param("policy:chase_lookahead", 0.3, 3.0, 1.0, log=True),
    Param("policy:memory_s", 0.0, 15.0, 4.0),
    Param("policy:ping_follow", 0.0, 1.0, 0.75),
]

HIDER_PARAMS = _DECODER_PARAMS + [
    Param("gain:loom", 0.0, 2.0, 1.0),  # LC4 / LPLC2
    Param("gain:danger", 0.0, 2.0, 1.0),  # aversive-odor ORNs (danger meter)
    Param("policy:w_far", 0.0, 4.0, 2.0),
    Param("policy:w_conceal", 0.0, 4.0, 1.0),
    Param("policy:w_vent", 0.0, 4.0, 0.5),
    Param("policy:w_dist", 0.0, 4.0, 1.0),
    Param("policy:camp_s", 1.0, 40.0, 10.0, log=True),
    Param("policy:flee_danger", 0.05, 1.0, 0.4),
    Param("policy:lookahead_units", 0.5, 4.0, 1.5, log=True),
]

# Obstacle-sense gain, only in the *_obs sets: an adapter without gain:obstacle has the sense off.
OBSTACLE_PARAM = Param("gain:obstacle", 0.0, 2.0, 1.0)
ROUTE_OBS_PARAMS = ROUTE_PARAMS + [OBSTACLE_PARAM]
SEEKER_OBS_PARAMS = SEEKER_PARAMS + [OBSTACLE_PARAM]

PARAM_SETS = {"ray": EXPLORE_PARAMS, "route": ROUTE_PARAMS, "seeker": SEEKER_PARAMS, "hider": HIDER_PARAMS, "route_obs": ROUTE_OBS_PARAMS, "seeker_obs": SEEKER_OBS_PARAMS}


def gain_values(values: dict) -> dict:
    return {name.split(":", 1)[1]: v for name, v in values.items() if name.startswith("gain:")}


def to_unit(params: list[Param], values: dict | None = None) -> np.ndarray:
    out = []
    for p in params:
        v = p.init if values is None else values.get(p.name, p.init)
        if p.log:
            out.append((np.log(v) - np.log(p.lo)) / (np.log(p.hi) - np.log(p.lo)))
        else:
            out.append((v - p.lo) / (p.hi - p.lo))
    return np.clip(np.array(out, dtype=float), 0.0, 1.0)


def decode(params: list[Param], unit: np.ndarray) -> dict:
    """unit: [D] or [A, D] in [0,1] -> {name: scalar or [A] array}."""
    u = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
    out = {}
    for j, p in enumerate(params):
        col = u[..., j]
        if p.log:
            out[p.name] = np.exp(np.log(p.lo) + col * (np.log(p.hi) - np.log(p.lo)))
        else:
            out[p.name] = p.lo + col * (p.hi - p.lo)
    return out


def apply_decoder(cfg: dict, values: dict) -> None:
    """Write decoder:* values (scalars or per-fly arrays) into a motor-decoder cfg dict in place."""
    for name, v in values.items():
        if not name.startswith("decoder:"):
            continue
        node = cfg
        *parents, leaf = name.split(":", 1)[1].split(".")
        for key in parents:
            node = node[key]
        node[leaf] = v
    if "goal_behind" in cfg:
        cfg["goal_behind"]["enabled"] = True


def policy_values(values: dict) -> dict:
    return {name.split(":", 1)[1]: v for name, v in values.items() if name.startswith("policy:")}
