"""
Benchmark the batched LIF simulator: steps/sec and real-time factor for
full/pruned5 graphs at B=1 and B=6 (Milestone M1 acceptance criterion).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from flyseek.brain.lif_torch import LIFBrain, load_config

RESULTS_PATH = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\docs\bench_results.json")


def bench_one(tag: str, batch_size: int, n_steps: int = 200, warmup: int = 20) -> dict:
    brain = LIFBrain(tag=tag)
    brain.reset(batch_size)

    n = brain.n_neurons
    dev = brain.device
    torch.manual_seed(0)

    for _ in range(warmup):
        ext = (torch.rand(n, batch_size, device=dev) < 0.001).float() * 5.0
        brain.step(ext)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_spikes = 0
    for _ in range(n_steps):
        ext = (torch.rand(n, batch_size, device=dev) < 0.001).float() * 5.0
        spikes = brain.step(ext)
        n_spikes += int(spikes.sum().item())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cfg = load_config()
    dt_ms = cfg["sim"]["dt_ms"]
    brain_ms_simulated = n_steps * dt_ms
    real_time_factor = (brain_ms_simulated / 1000.0) / elapsed

    return {
        "tag": tag,
        "batch_size": batch_size,
        "n_neurons": n,
        "n_edges": brain.num_edges(),
        "n_steps": n_steps,
        "wall_seconds": round(elapsed, 4),
        "steps_per_sec": round(n_steps / elapsed, 2),
        "ms_per_step": round(1000 * elapsed / n_steps, 4),
        "brain_ms_simulated": brain_ms_simulated,
        "real_time_factor": round(real_time_factor, 4),
        "mean_spikes_per_step": round(n_spikes / n_steps, 1),
        "device": str(dev),
    }


def main():
    results = []
    for tag in ["pruned5", "full"]:
        for batch_size in [1, 6]:
            print(f"Benchmarking tag={tag} batch_size={batch_size} ...")
            r = bench_one(tag, batch_size)
            print(f"  {r['steps_per_sec']} steps/sec, {r['ms_per_step']} ms/step, "
                  f"real-time factor {r['real_time_factor']}x, "
                  f"mean spikes/step {r['mean_spikes_per_step']}")
            results.append(r)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
