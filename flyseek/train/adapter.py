"""
Trainable adapter: the small set of engineered parameters around the frozen connectome
(Phase 4 exploration). CMA-ES searches a normalized [0, 1] vector; `decode` maps it to
physical values with bounds (log scale where marked).

Nothing here changes connectome weights.
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
    # init = best of the Phase 4.3 hand sweep (docs/phase4_arena_goal_sweep.json)
    Param("decoder:turn.scale_hz", 10.0, 120.0, 20.0, log=True),
    Param("decoder:ema_tau_ms", 20.0, 200.0, 100.0, log=True),
    Param("decoder:turn.weights.DNa02", 0.0, 2.0, 1.0),
    Param("decoder:turn.weights.DNa03", 0.0, 2.0, 0.5),
    Param("decoder:turn.weights.DNg13", -1.0, 1.0, 0.5),
    Param("decoder:goal_behind.threshold_hz", 15.0, 28.0, 20.0),
    Param("decoder:goal_behind.turn_rad_per_s", 0.0, 3.0, 2.0),
    Param("decoder:goal_behind.speed_factor", 0.1, 1.0, 0.4),
    Param("decoder:turn.max_omega_rad_per_s", 2.0, 10.0, 6.0),  # Phase 5.5 sweep: 6.0 best
    Param("policy:w_free", -2.0, 4.0, 1.0),
    Param("policy:w_novel", -2.0, 4.0, 1.5),
    Param("policy:w_align", -2.0, 4.0, 0.5),
    Param("policy:w_keep", -2.0, 4.0, 0.5),
    Param("policy:decide_s", 0.3, 5.0, 1.5, log=True),
]


_DECODER_PARAMS = [p for p in EXPLORE_PARAMS if p.name.startswith("decoder:")]

# Route policy (map-based planner, flyseek/agents/route_policy.py). Replaced the ray
# policy after docs/phase4_diag_explore.png showed flies couldn't find narrow exits.
ROUTE_PARAMS = _DECODER_PARAMS + [
    Param("policy:avoid_free_units", 0.0, 4.0, 1.5),  # Phase 5 wall-aware goal (0 = off)
    Param("policy:w_novel", 0.0, 4.0, 2.0),
    Param("policy:w_room", 0.0, 6.0, 3.0),
    Param("policy:w_dist", 0.0, 4.0, 1.0),
    Param("policy:replan_s", 0.5, 10.0, 3.0, log=True),
    Param("policy:lookahead_units", 0.5, 4.0, 1.5, log=True),
]

# Phase 5 role adapters (flyseek/agents/role_policy.py). Decoder and route inits are
# overridden by the graph's own Phase 4 explorer (--init-from), see es.py.
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
    Param("policy:camp_speed", 0.0, 1.0, 0.2),
    Param("policy:flee_danger", 0.05, 1.0, 0.4),
    Param("policy:lookahead_units", 0.5, 4.0, 1.5, log=True),
    Param("policy:avoid_free_units", 0.0, 4.0, 1.5),  # Phase 5 wall-aware goal (0 = off)
]

PARAM_SETS = {"ray": EXPLORE_PARAMS, "route": ROUTE_PARAMS, "seeker": SEEKER_PARAMS, "hider": HIDER_PARAMS}


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
