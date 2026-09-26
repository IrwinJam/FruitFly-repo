"""
Benchmark the LIF simulator on an active network (photoreceptors and ORNs at 20 Hz, LC10a-L
at 50 Hz): steps per second, real-time factor and spikes per step for each graph and batch size.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.brain.roles import role_idx
from amongusfly.paths import DOCS_DIR


def bench_one(tag: str, batch: int, propagation: str | None = None, seconds: float = 0.5, warmup_ms: float = 200) -> dict:
    brain = LIFBrain(tag=tag, propagation=propagation)
    brain.reset(batch, seed=0)
    broad = sorted(set(role_idx("photoreceptor_achromatic", graph=tag) + role_idx("photoreceptor_color", graph=tag)
                       + role_idx("aversive_odor", graph=tag) + role_idx("attractive_odor", graph=tag)))
    lc = role_idx("target_motion_detector", "L", graph=tag)
    neu, cols, rates = [], [], []
    for c in range(batch):
        neu += broad + lc
        cols += [c] * (len(broad) + len(lc))
        rates += [20.0] * len(broad) + [50.0] * len(lc)
    brain.set_stimulus(neu, cols, rates)

    brain.run(int(warmup_ms / brain.dt_ms))
    n_steps = int(seconds * 1000 / brain.dt_ms)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(n_steps):
        spikes += int(brain.step().sum())
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    res = {
        "tag": tag, "batch": batch, "n_neurons": brain.n_neurons, "n_edges": brain.num_edges(),
        "propagation": brain.propagation, "dt_ms": brain.dt_ms,
        "ms_per_step": round(1000 * wall / n_steps, 3),
        "real_time_factor_per_fly_batch": round(seconds / wall, 4),
        "spikes_per_step": round(spikes / n_steps, 1),
    }
    del brain
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=[
        "navcore:1:event", "navcore:1:spmm", "navcore:16:event", "navcore:16:spmm", "navcore:64:event", "navcore:64:spmm",
        "navcore:256:spmm", "pruned5:1:auto", "pruned5:6:auto", "pruned5:6:spmm", "full:1:auto", "full:6:auto"])
    args = ap.parse_args()
    results = []
    for spec in args.configs:
        tag, b, mode = spec.split(":")
        r = bench_one(tag, int(b), mode)
        r["fly_seconds_per_wall_second"] = round(r["real_time_factor_per_fly_batch"] * r["batch"], 3)
        print(f"{tag:8s} B={r['batch']:<3} {mode:5s} {r['ms_per_step']:8.3f} ms/step | real-time {r['real_time_factor_per_fly_batch']:.3f}x "
              f"| throughput {r['fly_seconds_per_wall_second']:.2f} fly-s/s | {r['spikes_per_step']:.0f} spikes/step", flush=True)
        results.append(r)
    (DOCS_DIR / "results" / "benchmark.json").write_text(json.dumps(results, indent=2))
