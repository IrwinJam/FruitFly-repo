"""
Held-out evaluation of exploration adapters (episode seeds from 50000, never used in training).

Each condition is one adapter on one graph, optionally with silenced cell types, on the same
seeds. Reports rooms visited, coverage and wall-contact time with 95% CIs, and a paired test
against the first condition.

Condition spec: label=graph:adapter[:silence]
  adapter = init | best:<run> | mean:<run>; the silence token NOSENSE switches the obstacle sense off.
    python -m amongusfly.train.eval_explore untrained=navcore:init trained=navcore:best:explore_v5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS, decode, to_unit
from amongusfly.train.explore_env import run_episode


def adapter_values(spec: str, params) -> dict:
    if spec == "init":
        return decode(params, to_unit(params))
    kind, run = spec.split(":", 1)
    if kind == "best":
        return json.loads((RESULTS_DIR / "train" / run / "best.json").read_text())["values"]
    if kind == "mean":
        lines = (RESULTS_DIR / "train" / run / "log.jsonl").read_text().strip().splitlines()
        return json.loads(lines[-1])["cma_mean"]
    raise ValueError(spec)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [round(float(x.mean()), 3), round(float(x.mean() - h), 3), round(float(x.mean() + h), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--policy", default="route")
    ap.add_argument("--speed", type=float, default=2.0, help="forward speed; matches run at 2.0")
    ap.add_argument("--out", default="results/navigation")
    ap.add_argument("--chunk", type=int, default=None, help="flies per batch (the full network needs small batches)")
    ap.add_argument("--seed-base", type=int, default=50000)
    args = ap.parse_args()
    params = PARAM_SETS[args.policy]
    seeds = list(range(args.seed_base, args.seed_base + args.n))
    results = {}
    brains = {}
    for c in args.conditions:
        label, rest = c.split("=", 1)
        parts = rest.split(":")
        graph = parts[0]
        if parts[1] == "init":
            adapter, silence = "init", parts[2:]
        else:
            adapter, silence = ":".join(parts[1:3]), parts[3:]
        vals = adapter_values(adapter, params)
        if graph not in brains:
            brains = {graph: LIFBrain(tag=graph)}
        gain = [t for t in silence if t.startswith("input=")]
        if gain:  # compass and goal input strength, e.g. trained=navcore:best:explore_v5:input=0.75
            silence = [t for t in silence if not t.startswith("input=")]
            vals = {**vals, "gain:compass": float(gain[0].split("=", 1)[1]), "gain:goal": float(gain[0].split("=", 1)[1])}
        if "NOSENSE" in silence:  # ablation: the obstacle sense switched off, everything else unchanged
            silence = [t for t in silence if t != "NOSENSE"]
            vals = {**vals, "gain:obstacle": 0.0}
        step = args.chunk or len(seeds)
        parts_r = [run_episode(graph, vals, seeds[i:i + step], seconds=args.seconds, silence=silence or None,
                               brain=brains[graph], policy_kind=args.policy, speed=args.speed)
                   for i in range(0, len(seeds), step)]
        r = {k: np.concatenate([np.atleast_1d(x[k]) for x in parts_r]) for k in ("rooms", "coverage", "stuck_s")}
        r["wall_s"] = sum(x["wall_s"] for x in parts_r)
        results[label] = {"graph": graph, "adapter": adapter, "silence": silence,
                          "rooms": r["rooms"].tolist(), "coverage": r["coverage"].tolist(), "stuck_s": r["stuck_s"].tolist(),
                          "rooms_ci": ci(r["rooms"]), "coverage_ci": ci(r["coverage"]), "stuck_ci": ci(r["stuck_s"])}
        print(f"{label:12s} rooms {results[label]['rooms_ci']} | coverage {results[label]['coverage_ci']} "
              f"| wall s {results[label]['stuck_ci']} | {r['wall_s']:.0f}s", flush=True)

    ref = next(iter(results))
    for label, r in results.items():
        if label == ref:
            continue
        t = stats.ttest_rel(r["rooms"], results[ref]["rooms"])
        w = stats.wilcoxon(np.array(r["rooms"]) - np.array(results[ref]["rooms"])) if np.any(np.array(r["rooms"]) != np.array(results[ref]["rooms"])) else None
        r["vs_" + ref] = {"rooms_diff_mean": round(float(np.mean(np.array(r["rooms"]) - np.array(results[ref]["rooms"]))), 3),
                          "paired_t_p": float(t.pvalue), "wilcoxon_p": float(w.pvalue) if w else None}
        print(f"  {label} vs {ref}: rooms diff {r['vs_' + ref]['rooms_diff_mean']:+.2f}, paired t p={t.pvalue:.3g}"
              + (f", Wilcoxon p={w.pvalue:.3g}" if w else ""), flush=True)
    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
