"""
Phase 2 closed-loop test for Hiders: does an approaching threat trigger the giant-
fiber escape dash?

Setup: empty 10x10 arena, fly at the origin facing +x and walking at the engineered
base speed. A threat (the "Seeker", radius 0.4) starts 5 units away at +90 deg
(left) or -90 deg (right) and moves straight at the fly at 2.5 units/s. The vision
encoder turns its angular expansion into LC4/LPLC2 drive (<= 100 Hz). A dash happens
when smoothed DNp01 (giant fiber) exceeds the decoder threshold (config/motors.yaml).

Conditions: real navcore, blind (looming channel off), shuffled navcore_shuf0..K-1.
Metrics: fraction of flies that dash before the threat reaches them, dash onset time
(time-to-contact at onset), and peak DNp01 rate.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from flyseek.agents.fly_agent import FlyPopulation
from flyseek.paths import DOCS_DIR
from flyseek.senses.vision import VisualObject
from flyseek.world.grid import OccupancyGrid

START_DIST = 5.0
THREAT_SPEED = 2.5
THREAT_RADIUS = 0.4


def run_condition(tag: str, trials_per_side: int, seed: int, blind: bool = False) -> dict:
    grid = OccupancyGrid.arena(12, 12)
    n = 2 * trials_per_side
    side = np.array([1.0] * trials_per_side + [-1.0] * trials_per_side)
    gain = {"target": 1.0, "loom": 0.0 if blind else 1.0, "photo": 1.0}
    pop = FlyPopulation(tag, np.zeros(n), np.zeros(n), np.zeros(n), grid, seed=seed, channel_gain=gain)
    dt = pop.tick_ms / 1000.0

    # threat starts beside the fly's start position and homes in on the fly's current position
    tx = np.zeros(n)
    ty = START_DIST * side
    contact_tick = np.full(n, -1)
    dash_tick = np.full(n, -1)
    peak_gf = np.zeros(n)
    max_ticks = int((START_DIST / THREAT_SPEED + 0.5) / dt)

    t0 = time.perf_counter()
    for t in range(max_ticks):
        b = pop.body
        res, _ = pop.tick([VisualObject(tx.copy(), ty.copy(), THREAT_RADIUS, "threat")])
        gf = (pop.decoder.rates["DNp01"]["L"] + pop.decoder.rates["DNp01"]["R"]) / 2
        peak_gf = np.maximum(peak_gf, gf)
        newly = (dash_tick < 0) & res.command.dash & (contact_tick < 0)
        dash_tick[newly] = t
        # move threat toward the fly
        dx, dy = b.x - tx, b.y - ty
        d = np.hypot(dx, dy)
        step = np.minimum(THREAT_SPEED * dt, np.maximum(d - 1e-6, 0))
        tx += dx / np.maximum(d, 1e-6) * step
        ty += dy / np.maximum(d, 1e-6) * step
        touching = (np.hypot(b.x - tx, b.y - ty) <= THREAT_RADIUS + b.radius) & (contact_tick < 0)
        contact_tick[touching] = t
    wall = time.perf_counter() - t0

    dashed_before = (dash_tick >= 0) & ((contact_tick < 0) | (dash_tick < contact_tick))
    ttc = np.where(dash_tick >= 0, (START_DIST - THREAT_RADIUS) / THREAT_SPEED - dash_tick * dt, np.nan)
    out = {
        "tag": tag, "blind": blind, "n": n, "wall_s": round(wall, 1),
        "dash_before_contact": float(dashed_before.mean()),
        "dash_before_contact_left": float(dashed_before[side > 0].mean()),
        "dash_before_contact_right": float(dashed_before[side < 0].mean()),
        "approx_time_to_contact_at_dash_s_mean": float(np.nanmean(ttc)) if np.isfinite(ttc).any() else None,
        "peak_giant_fiber_hz_mean": float(peak_gf.mean()),
        "contact_fraction": float((contact_tick >= 0).mean()),
    }
    del pop
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = []

    def show(label, r):
        print(f"{label:8s} dash-before-contact {r['dash_before_contact']:.2f} (L {r['dash_before_contact_left']:.2f} "
              f"R {r['dash_before_contact_right']:.2f}) | TTC at dash {r['approx_time_to_contact_at_dash_s_mean']} s "
              f"| peak GF {r['peak_giant_fiber_hz_mean']:.1f} Hz | {r['wall_s']}s", flush=True)

    r = run_condition("navcore", args.trials, args.seed); show("real", r); results.append(r)
    r = run_condition("navcore", args.trials, args.seed, blind=True); show("blind", r); results.append(r)
    for k in range(args.shuffles):
        r = run_condition(f"navcore_shuf{k}", args.trials, args.seed); show(f"shuf{k}", r); results.append(r)

    shuf = np.array([x["dash_before_contact"] for x in results[2:]])
    summary = {
        "real_dash_before_contact": results[0]["dash_before_contact"],
        "blind_dash_before_contact": results[1]["dash_before_contact"],
        "shuffle_dash_before_contact": shuf.tolist(),
        "real_vs_shuffles_empirical_p": float((1 + (shuf >= results[0]["dash_before_contact"]).sum()) / (1 + len(shuf))),
    }
    print(json.dumps(summary, indent=2))
    (DOCS_DIR / "phase2_arena_escape.json").write_text(json.dumps({"summary": summary, "conditions": results}, indent=2))


if __name__ == "__main__":
    main()
