"""
Phase 1 follow-up: at the calibrated scale, does broad sensory input (photoreceptors
and/or ORNs) still leave a self-sustained subnetwork running after the input stops?

Columns: photo_pulse, orn_pulse, photo+orn_pulse (each 20 Hz for 500 ms, then off),
plus none. Measures activity 500-1500 ms after input offset, and the top cell types.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flyseek.brain.lif_torch import LIFBrain
from flyseek.brain.roles import full_idx_of, neurons, role_idx
from flyseek.paths import DOCS_DIR

COLS = ["none", "photo_pulse", "orn_pulse", "photo_orn_pulse"]


def run(tag: str) -> dict:
    b = LIFBrain(tag=tag)
    b.reset(len(COLS), seed=0)
    photo = sorted(set(role_idx("photoreceptor_achromatic", graph=tag) + role_idx("photoreceptor_color", graph=tag)))
    orn = sorted(set(role_idx("aversive_odor", graph=tag) + role_idx("attractive_odor", graph=tag)))
    groups = {1: photo, 2: orn, 3: photo + orn}
    neu, cols, rates = [], [], []
    for c, idx in groups.items():
        neu += idx; cols += [c] * len(idx); rates += [20.0] * len(idx)
    b.set_stimulus(neu, cols, rates)
    during = b.run(int(500 / b.dt_ms))
    b.clear_stimulus()
    b.run(int(500 / b.dt_ms))
    after = b.run(int(1000 / b.dt_ms))
    rate_during = during["counts"].cpu().numpy() / 0.5
    rate_after = after["counts"].cpu().numpy() / 1.0

    df = neurons().sort_values("idx").reset_index(drop=True)
    full = full_idx_of(tag)
    if full is not None:
        df = df.iloc[full].reset_index(drop=True)

    out = {"tag": tag, "columns": {}}
    for c, name in enumerate(COLS):
        act = rate_after[:, c] > 1
        top = (df[act].assign(rate=rate_after[act, c]).groupby("type")["rate"].agg(["count", "mean"])
               .sort_values("count", ascending=False).head(12).round(1).reset_index().to_dict(orient="records"))
        out["columns"][name] = {
            "pop_rate_during_hz": float(rate_during[:, c].mean()),
            "pop_rate_after_hz": float(rate_after[:, c].mean()),
            "n_active_after_gt1hz": int(act.sum()),
            "superclass_after": df[act]["superclass"].value_counts().head(8).to_dict(),
            "top_types_after": top,
        }
    del b
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pruned5", "full", "navcore"])
    args = ap.parse_args()
    res = []
    for t in args.tags:
        r = run(t)
        for name, x in r["columns"].items():
            print(f"[{t}] {name:16s} during {x['pop_rate_during_hz']:.3f} Hz | after-offset {x['pop_rate_after_hz']:.3f} Hz "
                  f"| active after {x['n_active_after_gt1hz']}")
            if x["n_active_after_gt1hz"]:
                print("     superclass:", x["superclass_after"])
                print("     types:", [(y["type"], y["count"], y["mean"]) for y in x["top_types_after"][:8]])
        res.append(r)
    (DOCS_DIR / "phase1_ignition_calibrated.json").write_text(json.dumps(res, indent=2))
