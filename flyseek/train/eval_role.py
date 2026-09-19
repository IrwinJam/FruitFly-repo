"""
Held-out evaluation of Phase 5 role adapters. Match seeds start at 90000 (never used in
training). Every condition plays the same matches against the same scripted opponents.

Condition spec: label=controller[:graph:adapter[:silence,...]]
  controller = brain | scripted | random
  adapter    = init:<run> (the Phase 4 explorer's values, no role training) | best:<run> | mean:<run>
Examples:
  python -m flyseek.train.eval_role --role seeker trained=brain:navcore:best:seeker_navcore \
      explorer=brain:navcore:init:explore_navcore random=random scripted=scripted \
      lc10a_off=brain:navcore:best:seeker_navcore:LC10a
Reports mean +/- 95% CI of fitness, seeker win rate, time to first kill / hider survival,
and paired tests (per match) against the first condition.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from flyseek.brain.lif_torch import LIFBrain
from flyseek.paths import DOCS_DIR, RESULTS_DIR
from flyseek.train.adapter import PARAM_SETS, decode, to_unit
from flyseek.train.role_env import run_role_episode


def load_values(kind: str, run: str, params) -> dict:
    d = RESULTS_DIR / "train" / run
    if kind in ("init", "best"):  # init:<Phase 4 run> and best:<role run> both read best.json; missing role params use defaults
        return decode(params, to_unit(params, json.loads((d / "best.json").read_text())["values"]))
    if kind == "mean":
        return decode(params, to_unit(params, json.loads((d / "log.jsonl").read_text().strip().splitlines()[-1])["cma_mean"]))
    raise ValueError(kind)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return [round(float(x.mean()), 3), round(float(x.mean() - h), 3), round(float(x.mean() + h), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["seeker", "hider", "both"])
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=20, help="matches per batch")
    ap.add_argument("--preset", default="short")
    ap.add_argument("--spawn", default="default")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hiders", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=90000)
    args = ap.parse_args()
    params = PARAM_SETS.get(args.role)
    seeds = list(range(args.seed_base, args.seed_base + args.n))
    H = args.hiders
    results, brains = {}, {}
    for c in args.conditions:
        label, spec = c.split("=", 1)
        parts = spec.split(":")
        ctrl = parts[0]
        graph, values, silence = "navcore", None, None
        if ctrl == "brain":
            if args.role == "both":  # brain:graph:seekerKind:seekerRun:hiderKind:hiderRun[:silence]
                graph = parts[1]
                values = {"seeker": load_values(parts[2], parts[3], PARAM_SETS["seeker"]),
                          "hider": load_values(parts[4], parts[5], PARAM_SETS["hider"])}
                silence = parts[6].split(",") if len(parts) > 6 and parts[6] else None
            else:
                graph, kind, run = parts[1], parts[2], parts[3]
                silence = parts[4].split(",") if len(parts) > 4 and parts[4] else None
                values = load_values(kind, run, params)
            if graph not in brains:
                brains = {graph: LIFBrain(tag=graph)}
        per_match_fit, wins, first, surv, stuck = [], [], [], [], []
        per_fly = H if args.role == "hider" else 1
        for k in range(0, args.n, args.chunk):
            chunk = seeds[k:k + args.chunk]
            r = run_role_episode(graph, args.role, values, chunk, controller=ctrl, preset=args.preset,
                                 spawn=args.spawn, silence=silence, brain=brains.get(graph), n_hiders=H)
            fit = r["fitness_seeker"] if args.role == "both" else r["fitness"]  # "both": seeker fitness per match
            per_match_fit += list(np.asarray(fit).reshape(len(chunk), per_fly).mean(axis=1))
            wins += list(r["seeker_win"].astype(float))
            first += list(r["first_sighting_s"])
            if args.role in ("hider", "both"):
                surv += list(np.asarray(r["survival_s"]).reshape(len(chunk), H).mean(axis=1))
            if ctrl != "scripted":
                stuck += list(r["stuck_s"])
        res = {"controller": ctrl, "spec": spec, "fitness": per_match_fit, "fitness_ci": ci(per_match_fit),
               "seeker_win_rate": float(np.mean(wins)), "seeker_wins": int(np.sum(wins)),
               "first_sighting_s_median": float(np.nanmedian(first)) if np.isfinite(first).any() else None,
               "stuck_s_mean": float(np.mean(stuck)) if stuck else None}
        if surv:
            res["hider_survival_s"] = surv
            res["hider_survival_s_ci"] = ci(surv)
        results[label] = res
        print(f"{label:14s} fitness {res['fitness_ci']} | seeker wins {res['seeker_wins']}/{args.n}"
              + (f" | hider survival {res['hider_survival_s_ci']}" if surv else "")
              + (f" | stuck {res['stuck_s_mean']:.1f}s" if stuck else ""), flush=True)

    ref = next(iter(results))
    for label, r in results.items():
        if label == ref:
            continue
        a, b = np.array(r["fitness"]), np.array(results[ref]["fitness"])
        t = stats.ttest_rel(a, b)
        w = stats.wilcoxon(a - b) if np.any(a != b) else None
        r["vs_" + ref] = {"fitness_diff_mean": round(float((a - b).mean()), 3), "paired_t_p": float(t.pvalue),
                          "wilcoxon_p": float(w.pvalue) if w else None}
        print(f"  {label} vs {ref}: fitness diff {(a - b).mean():+.3f}, paired t p={t.pvalue:.3g}"
              + (f", Wilcoxon p={w.pvalue:.3g}" if w else ""), flush=True)
    out = args.out or f"phase5_eval_{args.role}"
    (DOCS_DIR / f"{out}.json").write_text(json.dumps({"role": args.role, "n": args.n, "preset": args.preset, "hiders": H,
                                                      "spawn": args.spawn, "results": results}, indent=2, default=float))


if __name__ == "__main__":
    main()
