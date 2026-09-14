"""
Phase 4.3 closed-loop test: can a fly reach a goal it CANNOT see, steering only with
the central-complex compass circuit?

Each fly starts at the origin with a random heading. An invisible goal point sits
4 units away in a random direction. Every tick the fly receives:
  - an EPG heading bump from its true heading (idealized compass)
  - an FC2 goal bump for the world direction from the fly to the goal
No visual target is shown. Turning comes only from the connectome's steering DNs
through the existing motor decoder; forward speed is the engineered constant.

Conditions:
  real       navcore, compass + goal
  no_goal    navcore, compass only (FC2 off)
  pfl3_off   navcore, compass + goal, PFL3 neurons silenced
  shuffled   navcore_shuf0..K-1, compass + goal
Success: within 0.75 units of the goal within the time limit.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy import stats

from flyseek.agents.fly_agent import FlyPopulation
from flyseek.brain.roles import type_idx
from flyseek.paths import DOCS_DIR
from flyseek.world.grid import OccupancyGrid

GOAL_DIST = 4.0
REACH = 0.75


def run_condition(tag: str, n: int, seconds: float, seed: int, goal_on=True, silence_pfl3=False,
                  decoder_overrides: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid.arena(14, 14)
    heading0 = rng.uniform(-np.pi, np.pi, n)
    gdir = rng.uniform(-np.pi, np.pi, n)
    gx, gy = GOAL_DIST * np.cos(gdir), GOAL_DIST * np.sin(gdir)
    gain = {"target": 1.0, "loom": 1.0, "photo": 1.0, "compass": 1.0, "goal": 1.0 if goal_on else 0.0}
    pop = FlyPopulation(tag, np.zeros(n), np.zeros(n), heading0, grid, seed=seed, channel_gain=gain)
    for path, value in (decoder_overrides or {}).items():  # e.g. {"turn.scale_hz": 40}
        node = pop.decoder.cfg
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    if silence_pfl3:
        pop.brain.silence(type_idx("PFL3", graph=tag))
    dt = pop.tick_ms / 1000
    ticks = int(seconds / dt)
    reached = np.full(n, -1)
    err0 = np.abs((np.arctan2(gy, gx) - heading0 + np.pi) % (2 * np.pi) - np.pi)
    err_1s = None
    t0 = time.perf_counter()
    for t in range(ticks):
        b = pop.body
        goal = np.arctan2(gy - b.y, gx - b.x)
        pop.tick([], goal_angle=goal)
        d = np.hypot(gx - pop.body.x, gy - pop.body.y)
        reached = np.where((reached < 0) & (d <= REACH), t, reached)
        if t + 1 == int(1.0 / dt):
            err_1s = np.abs((np.arctan2(gy - pop.body.y, gx - pop.body.x) - pop.body.heading + np.pi) % (2 * np.pi) - np.pi)
    wall = time.perf_counter() - t0
    del pop
    torch.cuda.empty_cache()
    ok = reached >= 0
    return {
        "tag": tag, "goal_on": goal_on, "silence_pfl3": silence_pfl3, "n": n, "wall_s": round(wall, 1),
        "success": float(ok.mean()),
        "time_to_goal_s_median": float(np.median(reached[ok]) * dt) if ok.any() else None,
        "heading_error_start_deg_mean": float(np.rad2deg(err0).mean()),
        "heading_error_1s_deg_mean": float(np.rad2deg(err_1s).mean()),
        "heading_error_1s_deg_per_fly": np.rad2deg(err_1s).round(1).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = []

    def show(label, r):
        print(f"{label:10s} success {r['success']:.2f} | median time {r['time_to_goal_s_median']} s | heading error "
              f"start {r['heading_error_start_deg_mean']:.0f} -> 1s {r['heading_error_1s_deg_mean']:.0f} deg | {r['wall_s']}s", flush=True)

    r = run_condition("navcore", args.n, args.seconds, args.seed); show("real", r); res.append(r)
    r = run_condition("navcore", args.n, args.seconds, args.seed, goal_on=False); show("no_goal", r); res.append(r)
    r = run_condition("navcore", args.n, args.seconds, args.seed, silence_pfl3=True); show("pfl3_off", r); res.append(r)
    for k in range(args.shuffles):
        r = run_condition(f"navcore_shuf{k}", args.n, args.seconds, args.seed); show(f"shuf{k}", r); res.append(r)

    real, nog = res[0], res[1]
    shuf = np.array([x["success"] for x in res[3:]])
    summary = {
        "real_success": real["success"], "no_goal_success": nog["success"], "pfl3_off_success": res[2]["success"],
        "shuffle_success": shuf.tolist(),
        "real_vs_shuffles_empirical_p": float((1 + (shuf >= real["success"]).sum()) / (1 + len(shuf))),
        "real_vs_no_goal_heading_error_1s_p": float(stats.ttest_ind(real["heading_error_1s_deg_per_fly"],
                                                                     nog["heading_error_1s_deg_per_fly"], equal_var=False).pvalue),
    }
    print(json.dumps(summary, indent=2))
    (DOCS_DIR / "phase4_arena_goal.json").write_text(json.dumps({"summary": summary, "conditions": res}, indent=2))


if __name__ == "__main__":
    main()
