"""
Phase 5 diagnostic: how well does the connectome steering loop hold a goal direction?

The Phase 4 route diagnostics (docs/phase4_diag_route_init_wallcost.json) showed the
goal direction is rarely blocked (15% near walls) but the fly's heading is 57-66 deg
off it on average, so wall contact comes from steering error, not bad planning.

Here flies sit in a large open arena (no walls in reach) with an idealised compass and
a fixed world goal direction; we sweep the injection and decoder settings and measure
the steady-state |heading - goal| error over the last third of the run, plus the time
to get within 30 deg. Nothing here touches connectome weights.

  python -m flyseek.experiments.steering_fidelity --graph navcore --n 32 --seconds 9
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch

from flyseek.agents.fly_agent import FlyPopulation
from flyseek.paths import DOCS_DIR
from flyseek.world.grid import OccupancyGrid

# label -> {cx: {...}, decoder: {dotted: value}, gain: {...}}
SWEEP = {
    "baseline": {},
    "kappa4": {"cx": {"kappa": 4.0}},
    "kappa8": {"cx": {"kappa": 8.0}},
    "rmax250": {"cx": {"r_max_hz": 250.0}},
    "rmax250_kappa4": {"cx": {"r_max_hz": 250.0, "kappa": 4.0}},
    "gain2": {"gain": {"compass": 2.0, "goal": 2.0}},
    "scale10": {"decoder": {"turn.scale_hz": 10.0}},
    "scale5": {"decoder": {"turn.scale_hz": 5.0}},
    "ema40": {"decoder": {"ema_tau_ms": 40.0}},
    "ema40_scale10": {"decoder": {"ema_tau_ms": 40.0, "turn.scale_hz": 10.0}},
    "omega6": {"decoder": {"turn.max_omega_rad_per_s": 6.0}},
    "best_guess": {"cx": {"kappa": 4.0, "r_max_hz": 250.0},
                   "decoder": {"ema_tau_ms": 40.0, "turn.scale_hz": 10.0, "turn.max_omega_rad_per_s": 4.5}},
}


def run(tag: str, n: int, seconds: float, seed: int, spec: dict, brain=None) -> dict:
    grid = OccupancyGrid.arena(400, 400)  # walls far away: pure steering, no collisions
    rng = np.random.default_rng(seed)
    heading0 = rng.uniform(-np.pi, np.pi, n)
    goal = rng.uniform(-np.pi, np.pi, n)
    gain = {"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0, "compass": 1.0, "goal": 1.0}
    gain.update(spec.get("gain", {}))
    pop = FlyPopulation(tag, np.full(n, 200.0), np.full(n, 200.0), heading0, grid, seed=seed, brain=brain,
                        channel_gain=gain)
    pop.cx_cfg = {**pop.cx_cfg, **spec.get("cx", {})}
    cfg = copy.deepcopy(pop.decoder.cfg)
    for path, v in spec.get("decoder", {}).items():
        node = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = v
    pop.decoder.cfg = cfg
    dt = pop.tick_ms / 1000
    err = []
    for _ in range(int(seconds / dt)):
        pop.tick([], goal_angle=goal)
        err.append(np.abs((goal - pop.body.heading + np.pi) % (2 * np.pi) - np.pi))
    err = np.rad2deg(np.stack(err))  # [T, n]
    tail = err[int(len(err) * 2 / 3):]
    within = err < 30.0
    first = np.where(within.any(axis=0), within.argmax(axis=0) * dt, np.nan)
    del pop
    torch.cuda.empty_cache()
    return {"steady_err_deg": float(tail.mean()), "steady_err_median": float(np.median(tail)),
            "frac_within_30deg": float(within[int(len(err) * 2 / 3):].mean()),
            "time_to_30deg_s": float(np.nanmedian(first)), "n_never_within_30": int(np.isnan(first).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="phase5_steering_fidelity")
    args = ap.parse_args()
    from flyseek.brain.lif_torch import LIFBrain
    brain = LIFBrain(tag=args.graph)
    out = {}
    for label, spec in SWEEP.items():
        if args.only and label not in args.only:
            continue
        t0 = time.perf_counter()
        r = run(args.graph, args.n, args.seconds, args.seed, spec, brain=brain)
        out[label] = {"spec": spec, **r}
        print(f"{label:16s} steady err {r['steady_err_deg']:5.1f} deg (median {r['steady_err_median']:5.1f}) | "
              f"within 30 deg {r['frac_within_30deg']:.2f} | t_30 {r['time_to_30deg_s']:.1f}s | "
              f"never {r['n_never_within_30']}/{args.n} | {time.perf_counter() - t0:.0f}s", flush=True)
    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
