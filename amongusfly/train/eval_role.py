"""
Held-out evaluation of role adapters (match seeds from 90000, never used in training).

Every condition plays the same matches against the same scripted opponents.
Condition spec: label=controller[:graph:adapter[:silence,...]]
  controller = brain | scripted | random;  adapter = init:<run> | best:<run> | mean:<run>
Reports fitness, seeker wins, first sighting, hider survival (95% CIs) and paired tests against
the first condition.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS, decode, to_unit
from amongusfly.train.role_env import run_role_episode


def load_values(kind: str, run: str, params) -> dict:
    d = RESULTS_DIR / "train" / run
    if kind in ("init", "best"):  # init: and best: both read best.json; missing role parameters use their defaults
        src = json.loads((d / "best.json").read_text())["values"]
    elif kind == "mean":
        src = json.loads((d / "log.jsonl").read_text().strip().splitlines()[-1])["cma_mean"]
    else:
        raise ValueError(kind)
    vals = decode(params, to_unit(params, src))
    # the obstacle-sense gain travels with the run that learned it; runs without it keep the sense off
    if "gain:obstacle" in src:
        vals["gain:obstacle"] = float(src["gain:obstacle"])
    return vals


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
    ap.add_argument("--save-paths", default=None, help="save every match's positions to <data>/results/paths/<name>/")
    ap.add_argument("--decoder-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="motor-config overrides for every brain condition, e.g. goal_behind.enabled=false")
    args = ap.parse_args()
    overrides = {}
    for kv in args.decoder_set:
        k, v = kv.split("=", 1)
        overrides[k] = {"true": True, "false": False}.get(v.lower(), None)
        if overrides[k] is None:
            overrides[k] = float(v)
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
            gain = [t for t in (silence or []) if t.startswith("input=")]
            if gain:  # e.g. brain:navcore:init:explore_v5:input=0.75
                silence = [t for t in silence if not t.startswith("input=")] or None
                g = float(gain[0].split("=", 1)[1])
                if args.role == "both":  # copies, so no other condition sees the change
                    values = {r: {**v, "gain:compass": g, "gain:goal": g} for r, v in values.items()}
                else:
                    values = {**values, "gain:compass": g, "gain:goal": g}
            if graph not in brains:
                brains = {graph: LIFBrain(tag=graph)}
        per_match_fit, wins, first, surv, stuck = [], [], [], [], []
        per_fly = H if args.role == "hider" else 1
        for k in range(0, args.n, args.chunk):
            chunk = seeds[k:k + args.chunk]
            r = run_role_episode(graph, args.role, values, chunk, controller=ctrl, preset=args.preset,
                                 spawn=args.spawn, silence=silence, brain=brains.get(graph), n_hiders=H,
                                 record_positions=bool(args.save_paths), decoder_overrides=overrides or None)
            if args.save_paths:  # positions per tick [T, match, agent] (float16), seeker wall contact, outcomes
                d = RESULTS_DIR / "paths" / args.save_paths
                d.mkdir(parents=True, exist_ok=True)
                tr = r.pop("traj")
                np.savez_compressed(d / f"{label}_{chunk[0]}.npz", x=tr[:, 0].astype(np.float16),
                                    y=tr[:, 1].astype(np.float16), seeker_contact=tr[:, 2, :, 0] > 0,
                                    seeds=np.array(chunk), end_s=r["match_end_s"], seeker_win=r["seeker_win"],
                                    survival_s=np.asarray(r["survival_s"]).reshape(len(chunk), H))
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
    out = args.out or f"results/{args.role}_eval"
    (DOCS_DIR / f"{out}.json").write_text(json.dumps({"role": args.role, "n": args.n, "preset": args.preset, "hiders": H,
                                                      "spawn": args.spawn, "decoder_set": overrides, "results": results},
                                                     indent=2, default=float))


if __name__ == "__main__":
    main()
