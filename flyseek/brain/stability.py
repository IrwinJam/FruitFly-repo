"""
Phase 1 stability test: under sustained, moderate sensory input, does network
activity settle, or does it run away (the recurrent-explosion risk the saturated
M3 looming result hinted at)?

Input: Poisson drive on all photoreceptors plus all olfactory/aversive/attractive
ORNs, at several rates, for several seconds. Each rate is one batch column.
Output: population rate over time, fraction of neurons ever active, and a
per-superclass rate breakdown for the last second.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from flyseek.brain.lif_torch import LIFBrain
from flyseek.brain.roles import full_idx_of, neurons, role_idx
from flyseek.paths import DOCS_DIR

SENSORY_ROLES = ["photoreceptor_achromatic", "photoreceptor_color", "aversive_odor", "attractive_odor"]


def run(tag: str, rates: list[float], seconds: float, seed: int = 0) -> dict:
    brain = LIFBrain(tag=tag)
    B = len(rates)
    brain.reset(B, seed=seed)

    sens = sorted({i for r in SENSORY_ROLES for i in role_idx(r, graph=tag)})
    neu, cols, rr = [], [], []
    for c, rate in enumerate(rates):
        neu += sens
        cols += [c] * len(sens)
        rr += [rate] * len(sens)
    brain.set_stimulus(neu, cols, rr)

    n_steps = int(seconds * 1000 / brain.dt_ms)
    bin_steps = int(100 / brain.dt_ms)  # 100 ms bins
    t0 = time.perf_counter()
    out = brain.run(n_steps, bin_steps=bin_steps)
    torch.cuda.synchronize() if brain.device.type == "cuda" else None
    wall = time.perf_counter() - t0

    counts = out["counts"].cpu().numpy()  # [N, B]
    pop = out["pop_rate_hz"].cpu().numpy()  # [bins, B]

    # per-superclass whole-run rates, excluding the directly stimulated neurons
    df = neurons().sort_values("idx").reset_index(drop=True)
    full = full_idx_of(tag)
    if full is not None:  # subgraph: rows in local index order
        df = df.iloc[full].reset_index(drop=True)
    non_stim = np.ones(brain.n_neurons, dtype=bool)
    non_stim[sens] = False

    results = []
    for c, rate in enumerate(rates):
        rate_per_neuron = counts[:, c] / seconds
        by_sc = (
            df.assign(rate=rate_per_neuron)[non_stim]
            .groupby("superclass")["rate"].mean()
            .sort_values(ascending=False)
        )
        early = float(pop[: max(1, len(pop) // 5), c].mean())
        late = float(pop[-max(1, len(pop) // 5):, c].mean())
        results.append({
            "input_rate_hz": rate,
            "measured_stim_rate_hz": float(counts[sens, c].mean() / seconds),
            "pop_rate_hz_first20pct": early,
            "pop_rate_hz_last20pct": late,
            "pop_rate_hz_max_bin": float(pop[:, c].max()),
            "late_over_early": late / early if early > 0 else None,
            "frac_nonstim_neurons_active": float((counts[non_stim, c] > 0).mean()),
            "mean_rate_nonstim_hz": float(rate_per_neuron[non_stim].mean()),
            "top_superclass_rates_hz": {k: round(float(v), 3) for k, v in by_sc.head(8).items()},
            "pop_rate_trace_hz": [round(float(x), 4) for x in pop[:, c]],
        })

    return {
        "tag": tag,
        "n_stimulated": len(sens),
        "seconds": seconds,
        "dt_ms": brain.dt_ms,
        "wall_seconds": round(wall, 1),
        "real_time_factor": round(seconds / wall * 1, 4),
        "batch": B,
        "results": results,
    }


def verdict(r: dict) -> str:
    ratio = r["late_over_early"]
    if r["pop_rate_hz_last20pct"] > 20:
        return "RUNAWAY (population >20 Hz)"
    if ratio is not None and ratio > 3:
        return "GROWING (late >3x early)"
    return "STABLE"


def plot(all_runs: list[dict], path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(all_runs), figsize=(5 * len(all_runs), 3.5), squeeze=False)
    for ax, run_ in zip(axes[0], all_runs):
        for r in run_["results"]:
            t = np.arange(len(r["pop_rate_trace_hz"])) * 0.1
            ax.plot(t, r["pop_rate_trace_hz"], label=f"{r['input_rate_hz']:g} Hz input")
        ax.set_title(run_["tag"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mean rate, all neurons (Hz)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pruned5", "pruned5_shuf0", "full"])
    ap.add_argument("--rates", type=float, nargs="+", default=[5, 20, 50])
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    all_runs = []
    for tag in args.tags:
        print(f"[{tag}] running {args.seconds}s at rates {args.rates} ...", flush=True)
        r = run(tag, args.rates, args.seconds)
        for x in r["results"]:
            print(f"  input {x['input_rate_hz']:>5g} Hz | stim measured {x['measured_stim_rate_hz']:6.1f} Hz | "
                  f"pop early {x['pop_rate_hz_first20pct']:.3f} late {x['pop_rate_hz_last20pct']:.3f} "
                  f"max {x['pop_rate_hz_max_bin']:.3f} Hz | active {100*x['frac_nonstim_neurons_active']:.1f}% "
                  f"| {verdict(x)}")
        print(f"  wall {r['wall_seconds']}s, real-time factor {r['real_time_factor']}x (batch {r['batch']})")
        all_runs.append(r)
        torch.cuda.empty_cache()

    for r in all_runs:
        for x in r["results"]:
            x["verdict"] = verdict(x)
    (DOCS_DIR / "phase1_stability.json").write_text(json.dumps(all_runs, indent=2))
    plot(all_runs, DOCS_DIR / "phase1_stability.png")
    print("saved docs/phase1_stability.json and .png")
