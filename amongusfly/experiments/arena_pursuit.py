"""
Closed-loop pursuit test: does a fly turn toward a target it sees?

Each fly starts at the origin facing +x with a static target 4 units away at +/-60 degrees and
walks at a fixed speed; all turning comes from the brain through the motor decoder. Conditions:
real graph, blind (target channel off) and shuffled graphs. Success criterion: the real graph
turns toward the target within 1 s in at least 70% of trials, clearly above blind and shuffled.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy import stats

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.senses.vision import VisualObject
from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.replay import ReplayRecorder

TARGET_DIST = 4.0
TARGET_BEARING_DEG = 60.0
TARGET_RADIUS = 0.3


def run_condition(tag: str, trials_per_side: int, seconds: float, seed: int, blind: bool = False,
                  record_path=None) -> dict:
    grid = OccupancyGrid.arena(10, 10)
    n = 2 * trials_per_side
    side = np.array([1.0] * trials_per_side + [-1.0] * trials_per_side)  # +1 left, -1 right
    b0 = np.deg2rad(TARGET_BEARING_DEG) * side
    tx, ty = TARGET_DIST * np.cos(b0), TARGET_DIST * np.sin(b0)
    target = VisualObject(tx, ty, TARGET_RADIUS, "target")

    gain = {"target": 0.0 if blind else 1.0, "loom": 1.0, "photo": 1.0}
    pop = FlyPopulation(tag, np.zeros(n), np.zeros(n), np.zeros(n), grid, seed=seed, channel_gain=gain)
    n_ticks = int(round(seconds * 1000 / pop.tick_ms))
    t1 = int(round(1000 / pop.tick_ms))

    rec = None
    if record_path is not None:
        rec = ReplayRecorder(n, {"experiment": "arena_pursuit", "graph": tag, "tick_ms": pop.tick_ms,
                                 "full_idx_available": pop.full_idx is not None,
                                 "targets": {"x": tx.tolist(), "y": ty.tolist(), "radius": TARGET_RADIUS},
                                 "disclaimer": "Connectome-constrained LIF model; forward speed is an engineered constant."})

    heading0 = pop.body.heading.copy()
    heading_1s = None
    min_dist = np.full(n, np.inf)
    turn_trace, target_hz_trace = [], []
    t0 = time.perf_counter()
    for t in range(n_ticks):
        res, all_counts = pop.tick([target], record_all_spikes=rec is not None)
        if rec is not None:
            rec.record(pop.state(), all_counts)
        b = pop.body
        min_dist = np.minimum(min_dist, np.hypot(tx - b.x, ty - b.y))
        if t + 1 == t1:
            heading_1s = b.heading.copy()
        turn_trace.append(res.command.omega.copy())
        target_hz_trace.append(res.vision_rates["target"]["L"] + res.vision_rates["target"]["R"])
    wall = time.perf_counter() - t0

    b = pop.body
    d_head = (heading_1s - heading0 + np.pi) % (2 * np.pi) - np.pi
    correct = (np.sign(d_head) == side) & (np.abs(d_head) > np.deg2rad(5))
    bearing_end = (np.arctan2(ty - b.y, tx - b.x) - b.heading + np.pi) % (2 * np.pi) - np.pi
    reduced = np.abs(bearing_end) < np.abs(b0)
    out = {
        "tag": tag, "blind": blind, "n": n, "seconds": seconds, "wall_s": round(wall, 1),
        "correct_turn_1s": float(correct.mean()),
        "correct_turn_1s_left": float(correct[side > 0].mean()),
        "correct_turn_1s_right": float(correct[side < 0].mean()),
        "bearing_reduced": float(reduced.mean()),
        "heading_change_1s_deg_toward_target_mean": float(np.rad2deg(d_head * side).mean()),
        "heading_change_toward_target_per_fly_deg": np.rad2deg(d_head * side).round(2).tolist(),
        "min_dist_mean": float(min_dist.mean()),
        "mean_target_input_hz": float(np.mean(target_hz_trace)),
        "mean_abs_omega": float(np.mean(np.abs(turn_trace))),
    }
    if rec is not None:
        out["replay"] = rec.save(record_path)
    del pop
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = []
    r = run_condition("navcore", args.trials, args.seconds, args.seed,
                      record_path=RESULTS_DIR / "replays" / "arena_pursuit_navcore")
    print(f"real     correct {r['correct_turn_1s']:.2f} (L {r['correct_turn_1s_left']:.2f} R {r['correct_turn_1s_right']:.2f}) "
          f"| toward {r['heading_change_1s_deg_toward_target_mean']:+.1f} deg | bearing reduced {r['bearing_reduced']:.2f} "
          f"| min dist {r['min_dist_mean']:.2f} | input {r['mean_target_input_hz']:.1f} Hz | {r['wall_s']}s | replay {r['replay']}", flush=True)
    results.append(r)
    r = run_condition("navcore", args.trials, args.seconds, args.seed, blind=True)
    print(f"blind    correct {r['correct_turn_1s']:.2f} | toward {r['heading_change_1s_deg_toward_target_mean']:+.1f} deg "
          f"| bearing reduced {r['bearing_reduced']:.2f} | min dist {r['min_dist_mean']:.2f}", flush=True)
    results.append(r)
    for k in range(args.shuffles):
        r = run_condition(f"navcore_shuf{k}", args.trials, args.seconds, args.seed)
        print(f"shuf{k:<3}  correct {r['correct_turn_1s']:.2f} | toward {r['heading_change_1s_deg_toward_target_mean']:+.1f} deg "
              f"| bearing reduced {r['bearing_reduced']:.2f} | min dist {r['min_dist_mean']:.2f}", flush=True)
        results.append(r)

    real = results[0]
    shuf = [x for x in results[2:]]
    real_toward = np.array(real["heading_change_toward_target_per_fly_deg"])
    blind_toward = np.array(results[1]["heading_change_toward_target_per_fly_deg"])
    shuf_means = np.array([x["heading_change_1s_deg_toward_target_mean"] for x in shuf])
    summary = {
        "real_correct_turn_1s": real["correct_turn_1s"],
        "blind_correct_turn_1s": results[1]["correct_turn_1s"],
        "shuffle_correct_turn_1s": [x["correct_turn_1s"] for x in shuf],
        "real_vs_blind_toward_deg_welch_p": float(stats.ttest_ind(real_toward, blind_toward, equal_var=False).pvalue),
        "real_toward_mean_deg": float(real_toward.mean()),
        "shuffle_toward_mean_deg": shuf_means.tolist(),
        "real_vs_shuffles_empirical_p": float((1 + (shuf_means >= real_toward.mean()).sum()) / (1 + len(shuf_means))),
        "success_criterion_met": bool(real["correct_turn_1s"] >= 0.7),
    }
    print(json.dumps(summary, indent=2))
    (DOCS_DIR / "results" / "pursuit_test.json").write_text(json.dumps({"summary": summary, "conditions": results}, indent=2))


if __name__ == "__main__":
    main()
