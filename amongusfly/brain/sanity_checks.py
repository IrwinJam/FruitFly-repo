"""
Sensory-to-descending-neuron checks with controls.

For the real graph and shuffled graphs, each condition runs as independent replicate batch
columns (200 ms warm-up, 1 s measurement): pursuit (LC10a left or right at several rates ->
DNa02/DNa03), looming (LC4 + LPLC2 -> DNp01), ignition (a brief pulse, then silence) and no input.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy import stats

from amongusfly.brain.lif_torch import LIFBrain, load_config
from amongusfly.brain.roles import role_idx, type_idx
from amongusfly.paths import DOCS_DIR

READOUT_TYPES = ["DNa02", "DNa03", "DNa01", "DNg13", "DNp01", "DNg100", "DNp09", "MDN", "pIP1"]


def build_conditions(rates: list[float], replicates: int, graph: str):
    """Returns list of (name, idx, rate, pulse_only) and the (cond_name, replicate) column map."""
    lc10a = {s: role_idx("target_motion_detector", s, graph) for s in "LR"}
    loom = {s: role_idx("looming_expansion", s, graph) + role_idx("looming_size", s, graph) for s in "LR"}
    conds = [("none", None, None, False)]
    for r in rates:
        for s in "LR":
            conds.append((f"pursuit_{s}_{r:g}", lc10a[s], r, False))
            conds.append((f"loom_{s}_{r:g}", loom[s], r, False))
    conds.append(("pulse_dna02", role_idx("steering_high_gain", "L", graph)[:1], 100.0, True))
    conds.append(("pulse_lc10a", lc10a["L"], 50.0, True))

    columns = []  # (cond_name, replicate)
    for name, *_ in conds:
        for k in range(replicates):
            columns.append((name, k))
    return conds, columns


def stim_lists(conds, columns, which: str):
    """which: 'all' (sustained + pulses) or 'sustained'."""
    by_name = {c[0]: c for c in conds}
    neu, cols, rates = [], [], []
    for col, (name, _) in enumerate(columns):
        _, idx, rate, pulse_only = by_name[name]
        if idx is None or (which == "sustained" and pulse_only):
            continue
        neu += idx
        cols += [col] * len(idx)
        rates += [rate] * len(idx)
    return neu, cols, rates


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [float(x.mean()), float("nan"), float("nan")]
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return [float(m), float(m - h), float(m + h)]


def contrast(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sp if sp > 0 else float("inf") if a.mean() != b.mean() else 0.0
    return {"mean_a": ci95(a), "mean_b": ci95(b), "t": float(t), "p": float(p), "cohens_d": float(d)}


def run_graph(tag: str, rates: list[float], replicates: int, dt_ms: float | None, seed: int) -> dict:
    brain = LIFBrain(tag=tag, dt_ms=dt_ms)
    conds, columns = build_conditions(rates, replicates, tag)
    B = len(columns)
    brain.reset(B, seed=seed)

    # pulse phase (50 ms): pulses + sustained
    brain.set_stimulus(*stim_lists(conds, columns, "all"))
    brain.run(int(round(50 / brain.dt_ms)))
    # warmup to 250 ms total, sustained only
    brain.set_stimulus(*stim_lists(conds, columns, "sustained"))
    warm = brain.run(int(round(200 / brain.dt_ms)))
    # measurement: 1 s
    t0 = time.perf_counter()
    out = brain.run(int(round(1000 / brain.dt_ms)))
    wall = time.perf_counter() - t0
    rate = out["counts"].cpu().numpy().astype(np.float64)  # Hz over 1 s, [N, B]

    col_of = {}
    for col, (name, k) in enumerate(columns):
        col_of.setdefault(name, []).append(col)

    readout = {f"{t}_{s}": type_idx(t, s, tag) for t in READOUT_TYPES for s in "LR" if type_idx(t, s, tag)}

    def r_type(key, name):
        return rate[readout[key]][:, col_of[name]].mean(axis=0)  # per replicate

    result = {"tag": tag, "dt_ms": brain.dt_ms, "replicates": replicates, "batch": B,
              "measure_wall_s": round(wall, 1), "cfg_scale": load_config().get("connectome_weight_scale"),
              "conditions": {}, "tests": {}}

    for name in col_of:
        cond_entry = {
            "pop_rate_hz": ci95(rate[:, col_of[name]].mean(axis=0)),
            "frac_active_gt1hz": ci95((rate[:, col_of[name]] > 1).mean(axis=0)),
            "readout_hz": {k: ci95(r_type(k, name)) for k in readout},
        }
        cdef = next(c for c in conds if c[0] == name)
        if cdef[1] is not None and not cdef[3]:
            cond_entry["measured_stim_rate_hz"] = ci95(rate[cdef[1]][:, col_of[name]].mean(axis=0))
        result["conditions"][name] = cond_entry

    for r in rates:
        for dn in ("DNa02", "DNa03"):
            asymL = r_type(f"{dn}_L", f"pursuit_L_{r:g}") - r_type(f"{dn}_R", f"pursuit_L_{r:g}")
            asymR = r_type(f"{dn}_L", f"pursuit_R_{r:g}") - r_type(f"{dn}_R", f"pursuit_R_{r:g}")
            result["tests"][f"pursuit_{dn}_LminusR_Ldrive_vs_Rdrive_{r:g}"] = contrast(asymL, asymR)
        gfL = r_type("DNp01_L", f"loom_L_{r:g}") - r_type("DNp01_R", f"loom_L_{r:g}")
        gfR = r_type("DNp01_L", f"loom_R_{r:g}") - r_type("DNp01_R", f"loom_R_{r:g}")
        result["tests"][f"loom_DNp01_LminusR_Ldrive_vs_Rdrive_{r:g}"] = contrast(gfL, gfR)
        gf_any = (r_type("DNp01_L", f"loom_L_{r:g}") + r_type("DNp01_R", f"loom_L_{r:g}")) / 2
        gf_none = (r_type("DNp01_L", "none") + r_type("DNp01_R", "none")) / 2
        result["tests"][f"loom_DNp01_drive_vs_none_{r:g}"] = contrast(gf_any, gf_none)

    for p in ("pulse_dna02", "pulse_lc10a"):
        result["tests"][f"ignition_{p}_frac_active_after"] = ci95((rate[:, col_of[p]] > 1).mean(axis=0))

    del brain
    torch.cuda.empty_cache()
    return result


def write_markdown(all_results: list[dict], path):
    lines = ["# Sensory-pathway checks (calibrated model)", "",
             "Generated by `amongusfly/brain/sanity_checks.py`. Values: mean [95% CI] across replicates. "
             "p: Welch t-test. d: Cohen's d.", ""]
    for res in all_results:
        lines += [f"## `{res['tag']}` (dt={res['dt_ms']} ms, {res['replicates']} replicates, "
                  f"weight scale {res['cfg_scale']})", "",
                  "| test | left-drive | right-drive | p | d |", "|---|---|---|---|---|"]
        for k, v in res["tests"].items():
            if k.startswith("ignition"):
                continue
            fa = lambda c: f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"
            lines.append(f"| {k} | {fa(v['mean_a'])} | {fa(v['mean_b'])} | {v['p']:.2g} | {v['cohens_d']:+.2f} |")
        for p in ("pulse_dna02", "pulse_lc10a"):
            c = res["tests"][f"ignition_{p}_frac_active_after"]
            lines.append(f"| ignition after {p} (fraction of neurons >1 Hz) | {c[0]:.4f} | | | |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pruned5", "pruned5_shuf0", "pruned5_shuf1", "pruned5_shuf2"])
    ap.add_argument("--rates", type=float, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/brain_sensory_checks")
    args = ap.parse_args()

    all_results = []
    for tag in args.tags:
        t0 = time.perf_counter()
        res = run_graph(tag, args.rates, args.replicates, args.dt, args.seed)
        print(f"[{tag}] done in {time.perf_counter()-t0:.0f}s (batch {res['batch']})", flush=True)
        for k, v in res["tests"].items():
            if k.startswith("ignition"):
                print(f"   {k}: {v[0]:.4f}")
            else:
                print(f"   {k}: L {v['mean_a'][0]:+.1f}  R {v['mean_b'][0]:+.1f}  p={v['p']:.2g}  d={v['cohens_d']:+.2f}")
        all_results.append(res)

    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(all_results, indent=2))
    write_markdown(all_results, DOCS_DIR / f"{args.out}.md")
    print(f"saved docs/{args.out}.json and .md")
