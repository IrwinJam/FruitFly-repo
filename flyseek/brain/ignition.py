"""
Phase 1 diagnostic: is the network self-sustaining after a brief input?

Batch columns (one run, same graph):
  0 none            -- no input at all
  1 photo_pulse     -- all photoreceptors, 20 Hz, 50 ms, then OFF
  2 lc10a_pulse     -- LC10a (left) only, 50 Hz, 50 ms, then OFF
  3 single_pulse    -- one DNa02 neuron, 100 Hz, 50 ms, then OFF
  4 photo_sustained -- all photoreceptors, 20 Hz, whole run

If columns 1-3 stay active long after the pulse ends, the model has a
self-sustained ("ignited") state, and descending-neuron readouts will stop
tracking the input.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flyseek.brain.lif_torch import LIFBrain, load_edges
from flyseek.brain.roles import role_idx
from flyseek.paths import DOCS_DIR

COLS = ["none", "photo_pulse", "lc10a_pulse", "single_pulse", "photo_sustained"]


def ei_stats(tag: str) -> dict:
    _, w, _ = load_edges(tag)
    pos, neg = w[w > 0], w[w < 0]
    return {
        "frac_edges_excitatory": float(len(pos) / len(w)),
        "total_exc_mv": float(pos.sum()),
        "total_inh_mv": float(-neg.sum()),
        "exc_over_inh_weight": float(pos.sum() / -neg.sum()),
    }


def stim_lists(brain, active_cols, seconds_pulse=True):
    photo = sorted(set(role_idx("photoreceptor_achromatic") + role_idx("photoreceptor_color")))
    lc10a_l = role_idx("target_motion_detector", "L")
    dna02 = role_idx("steering_high_gain", "L")[:1]
    spec = {1: (photo, 20.0), 2: (lc10a_l, 50.0), 3: (dna02, 100.0), 4: (photo, 20.0)}
    neu, cols, rates = [], [], []
    for c in active_cols:
        idx, r = spec[c]
        neu += idx
        cols += [c] * len(idx)
        rates += [r] * len(idx)
    return neu, cols, rates


def run(tag: str, seconds: float) -> dict:
    brain = LIFBrain(tag=tag)
    B = len(COLS)
    brain.reset(B, seed=0)
    bin_steps = int(50 / brain.dt_ms)

    # phase A: 50 ms pulse (cols 1-4 on)
    brain.set_stimulus(*stim_lists(brain, [1, 2, 3, 4]))
    a = brain.run(int(50 / brain.dt_ms), bin_steps=bin_steps)
    # phase B: pulses off, sustained stays on
    brain.set_stimulus(*stim_lists(brain, [4]))
    b = brain.run(int((seconds * 1000 - 50) / brain.dt_ms), bin_steps=bin_steps)

    trace = torch.cat([a["pop_rate_hz"], b["pop_rate_hz"]]).cpu().numpy()
    counts_after = b["counts"].cpu().numpy()
    last_n = max(1, int(0.5 * 1000 / 50))  # last 0.5 s of bins
    res = {}
    for c, name in enumerate(COLS):
        res[name] = {
            "pop_rate_last_0p5s_hz": float(trace[-last_n:, c].mean()),
            "pop_rate_peak_hz": float(trace[:, c].max()),
            "frac_active_after_pulse": float((counts_after[:, c] > 0).mean()),
            "trace_hz": [round(float(x), 4) for x in trace[:, c]],
        }
    return {"tag": tag, "seconds": seconds, "bin_ms": 50, "ei": ei_stats(tag), "columns": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pruned5", "pruned5_shuf0"])
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    out = []
    for tag in args.tags:
        r = run(tag, args.seconds)
        print(f"[{tag}] E/I: {r['ei']}")
        for name, x in r["columns"].items():
            print(f"  {name:16s} peak {x['pop_rate_peak_hz']:7.3f} Hz | last 0.5s {x['pop_rate_last_0p5s_hz']:7.3f} Hz "
                  f"| active after pulse {100*x['frac_active_after_pulse']:5.1f}%")
        out.append(r)
        torch.cuda.empty_cache()
    (DOCS_DIR / "phase1_ignition.json").write_text(json.dumps(out, indent=2))
