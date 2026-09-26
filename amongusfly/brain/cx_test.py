"""
Open-loop compass test: does the central-complex pathway turn a heading bump (EPG) and a
goal bump (FC2) into a steering signal that depends on goal minus heading?

Bumps are von Mises over each neuron's preferred angle:
    rate = r_max * exp(kappa * (cos(angle - centre) - 1))
Conditions cover 8 headings x 8 goal offsets plus EPG-only, FC2-only and no-input controls.
Readouts over 1 s: PFL3 grouped by steering side, DNa02 and DNa03 left and right. The real graph
is compared with shuffled graphs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.brain.roles import _to_graph, type_idx
from amongusfly.paths import CACHE_DIR, CONFIG_DIR, DOCS_DIR

ANGLES = np.deg2rad(np.arange(0, 360, 45))


def cx_neurons(graph: str) -> dict:
    cx = pd.DataFrame(json.loads((CACHE_DIR / "cx_map.json").read_text()))
    # Left/right bridge convention (config/cx.yaml pb_phase_convention): glomeruli are numbered
    # outward from the midline in both halves, so the preferred angle runs in opposite index
    # directions on the two sides ("mirror_R"). AMONGUSFLY_EPG_PHASE_MODE overrides the config.
    cfg = yaml.safe_load((CONFIG_DIR / "cx.yaml").read_text(encoding="utf-8"))
    mode = os.environ.get("AMONGUSFLY_EPG_PHASE_MODE", cfg.get("pb_phase_convention", "same"))
    if mode in ("mirror_L", "mirror_R"):
        side = mode[-1]
        m = (cx["type"] == "EPG") & (cx["pb_side"] == side)
        cx.loc[m, "pb_phase_deg"] = (-cx.loc[m, "pb_phase_deg"]) % 360.0

    def local(rows, col):
        out = []
        for _, r in rows.iterrows():
            li = _to_graph([int(r["idx"])], graph)
            if li:
                out.append((li[0], np.deg2rad(r[col]) if col else r["steer_side"]))
        return out

    return {
        "EPG": local(cx[cx["type"] == "EPG"], "pb_phase_deg"),
        "FC2": local(cx[cx["type"].isin(["FC2A", "FC2B", "FC2C"])], "fb_phase_deg"),
        "PFL3": local(cx[cx["type"] == "PFL3"], None),
    }


def bump(neurons, center, r_max, kappa):
    idx = np.array([i for i, _ in neurons])
    ph = np.array([p for _, p in neurons])
    rate = r_max * np.exp(kappa * (np.cos(ph - center) - 1))
    keep = rate >= 1.0
    return idx[keep], rate[keep]


def fit_cos(x, y):
    """Least squares y = c + a cos x + b sin x. Returns amplitude, phase (rad), R^2."""
    X = np.stack([np.ones_like(x), np.cos(x), np.sin(x)], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(np.hypot(coef[1], coef[2])), float(np.arctan2(coef[2], coef[1])), (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def run_graph(tag: str, replicates: int, r_max: float, kappa: float, seed: int = 0) -> dict:
    cx = cx_neurons(tag)
    conds = []
    for h in ANGLES:
        for d in ANGLES:
            conds.append(("both", h, (h + d) % (2 * np.pi), d))
    for h in ANGLES:
        conds.append(("epg_only", h, np.nan, np.nan))
    for g in ANGLES:
        conds.append(("fc2_only", np.nan, g, np.nan))
    conds.append(("none", np.nan, np.nan, np.nan))

    cols = [(ci, k) for ci in range(len(conds)) for k in range(replicates)]
    brain = LIFBrain(tag=tag)
    brain.reset(len(cols), seed=seed)
    neu, cc, rr = [], [], []
    for col, (ci, _) in enumerate(cols):
        kind, h, g, _ = conds[ci]
        if kind in ("both", "epg_only"):
            i, r = bump(cx["EPG"], h, r_max, kappa)
            neu.append(i); cc.append(np.full(len(i), col)); rr.append(r)
        if kind in ("both", "fc2_only"):
            i, r = bump(cx["FC2"], g, r_max, kappa)
            neu.append(i); cc.append(np.full(len(i), col)); rr.append(r)
    brain.set_stimulus(np.concatenate(neu), np.concatenate(cc), np.concatenate(rr))
    brain.run(int(200 / brain.dt_ms))
    t0 = time.perf_counter()
    out = brain.run(int(1000 / brain.dt_ms))
    wall = time.perf_counter() - t0
    rate = out["counts"].cpu().numpy().astype(float)  # Hz, [N, B]
    del brain
    torch.cuda.empty_cache()

    pfl3_L = [i for i, s in cx["PFL3"] if s == "L"]
    pfl3_R = [i for i, s in cx["PFL3"] if s == "R"]
    readouts = {
        "PFL3_steerL_minus_steerR": rate[pfl3_L].mean(0) - rate[pfl3_R].mean(0),
        "DNa02_L_minus_R": rate[type_idx("DNa02", "L", tag)].mean(0) - rate[type_idx("DNa02", "R", tag)].mean(0),
        "DNa03_L_minus_R": rate[type_idx("DNa03", "L", tag)].mean(0) - rate[type_idx("DNa03", "R", tag)].mean(0),
        "PFL3_mean": rate[pfl3_L + pfl3_R].mean(0),
        "PFL2_mean": rate[type_idx("PFL2", graph=tag)].mean(0) if type_idx("PFL2", graph=tag) else np.zeros(rate.shape[1]),
    }

    res = {"tag": tag, "replicates": replicates, "r_max_hz": r_max, "kappa": kappa, "measure_wall_s": round(wall, 1),
           "n_stim": {k: len(v) for k, v in cx.items()}, "fits": {}, "controls": {}}
    both = [ci for ci, c in enumerate(conds) if c[0] == "both"]
    col_of = {}
    for col, (ci, _) in enumerate(cols):
        col_of.setdefault(ci, []).append(col)
    for name, y in readouts.items():
        ym = np.array([y[col_of[ci]].mean() for ci in both])
        hs = np.array([conds[ci][1] for ci in both])
        gs = np.array([conds[ci][2] for ci in both])
        ds = np.array([conds[ci][3] for ci in both])
        A_d, ph_d, r2_d = fit_cos(ds, ym)
        A_g, _, r2_g = fit_cos(gs, ym)
        A_h, _, r2_h = fit_cos(hs, ym)
        per_d = {int(round(np.rad2deg(d))): round(float(ym[ds == d].mean()), 3) for d in ANGLES}
        res["fits"][name] = {"amp_rel": round(A_d, 3), "pref_offset_deg": round(float(np.rad2deg(ph_d)) % 360, 1),
                             "r2_rel": round(r2_d, 3), "r2_goal_only": round(r2_g, 3), "r2_heading_only": round(r2_h, 3),
                             "mean_by_offset": per_d}
        res["controls"][name] = {
            kind: round(float(np.mean([y[col_of[ci]].mean() for ci, c in enumerate(conds) if c[0] == kind])), 3)
            for kind in ("epg_only", "fc2_only", "none")
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--r-max", type=float, default=40.0)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--out", default="results/compass_open_loop")
    args = ap.parse_args()

    results = []
    tags = [args.graph] + [f"{args.graph}_shuf{k}" for k in range(args.shuffles)]
    for tag in tags:
        t0 = time.perf_counter()
        r = run_graph(tag, args.replicates, args.r_max, args.kappa)
        f = r["fits"]
        print(f"[{tag}] {time.perf_counter()-t0:.0f}s | "
              + " | ".join(f"{k}: A={v['amp_rel']:.2f} R2rel={v['r2_rel']:.2f} (goal {v['r2_goal_only']:.2f}, head {v['r2_heading_only']:.2f}) "
                           f"pref={v['pref_offset_deg']:.0f}" for k, v in f.items() if k != "PFL3_mean"), flush=True)
        results.append(r)

    real, shuf = results[0], results[1:]
    summary = {}
    for k in real["fits"]:
        ra, rr2 = real["fits"][k]["amp_rel"], real["fits"][k]["r2_rel"]
        sa = np.array([s["fits"][k]["amp_rel"] for s in shuf])
        s2 = np.array([s["fits"][k]["r2_rel"] for s in shuf])
        summary[k] = {"real_amp": ra, "shuffle_amp_max": float(sa.max()) if len(sa) else None,
                      "p_amp": float((1 + (sa >= ra).sum()) / (1 + len(sa))),
                      "real_r2_rel": rr2, "shuffle_r2_rel_max": float(s2.max()) if len(s2) else None,
                      "p_r2": float((1 + (s2 >= rr2).sum()) / (1 + len(s2)))}
    print(json.dumps(summary, indent=2))
    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps({"summary": summary, "graphs": results}, indent=2))


if __name__ == "__main__":
    main()
