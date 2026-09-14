"""
CMA-ES training of the exploration adapter (Phase 4.4/4.5).

Each generation: CMA proposes P candidate adapters; each candidate is evaluated on E
episode seeds; all P*E flies run as ONE GPU batch on the connectome graph. Fitness is
the mean over seeds (rooms visited + 10 * map coverage), maximized.

Resumable: CMA state, the generation log and the best adapter are checkpointed every
generation under RESULTS_DIR/train/<run>/. Re-running the same command continues.

Usage:
  python -m flyseek.train.es --run explore_navcore --graph navcore --generations 60
  python -m flyseek.train.es --run explore_shuf0 --graph navcore_shuf0 --generations 60   # control
  python -m flyseek.train.es --run explore_pfl3off --graph navcore --silence PFL3          # control
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
from cmaes import CMA

from flyseek.brain.lif_torch import LIFBrain
from flyseek.paths import RESULTS_DIR
from flyseek.train.adapter import PARAM_SETS, decode, to_unit
from flyseek.train.explore_env import run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--silence", nargs="*", default=[])
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--seeds-per-candidate", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="route", choices=list(PARAM_SETS))
    args = ap.parse_args()

    out = RESULTS_DIR / "train" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt, log_path = out / "cma.pkl", out / "log.jsonl"
    params = PARAM_SETS[args.policy]
    if ckpt.exists():
        state = pickle.loads(ckpt.read_bytes())
        opt, gen0, best = state["opt"], state["generation"] + 1, state["best"]
        print(f"resuming {args.run} at generation {gen0}", flush=True)
    else:
        opt = CMA(mean=to_unit(params), sigma=args.sigma, bounds=np.array([[0.0, 1.0]] * len(params)),
                  population_size=args.pop, seed=args.seed)
        gen0, best = 0, {"fitness": -np.inf}
        (out / "config.json").write_text(json.dumps({**vars(args), "params": [p.__dict__ for p in params]}, indent=2))

    brain = LIFBrain(tag=args.graph)
    E = args.seeds_per_candidate
    for gen in range(gen0, args.generations):
        t0 = time.perf_counter()
        cands = [opt.ask() for _ in range(opt.population_size)]
        unit = np.repeat(np.stack(cands), E, axis=0)  # [P*E, D]
        # fresh episode seeds each generation (same seeds for every candidate in a generation)
        seeds = [args.seed * 100000 + gen * 1000 + e for _ in cands for e in range(E)]
        values = decode(params, unit)
        res = run_episode(args.graph, values, seeds, seconds=args.seconds, silence=args.silence or None, brain=brain,
                          policy_kind=args.policy)
        fit = res["fitness"].reshape(len(cands), E).mean(axis=1)
        opt.tell([(c, -float(f)) for c, f in zip(cands, fit)])

        i = int(np.argmax(fit))
        if fit[i] > best["fitness"]:
            best = {"fitness": float(fit[i]), "generation": gen,
                    "values": {k: float(v) for k, v in decode(params, cands[i]).items()}}
        mean_vals = {k: float(v) for k, v in decode(params, opt.mean).items()}
        entry = {"generation": gen, "fitness_mean": float(fit.mean()), "fitness_max": float(fit.max()),
                 "rooms_mean": float(res["rooms"].mean()), "coverage_mean": float(res["coverage"].mean()),
                 "stuck_s_mean": float(res["stuck_s"].mean()), "best_so_far": best["fitness"],
                 "cma_mean": mean_vals, "wall_s": round(time.perf_counter() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        ckpt.write_bytes(pickle.dumps({"opt": opt, "generation": gen, "best": best}))
        (out / "best.json").write_text(json.dumps(best, indent=2))
        print(f"[{args.run}] gen {gen:3d} | fitness mean {entry['fitness_mean']:.2f} max {entry['fitness_max']:.2f} "
              f"| rooms {entry['rooms_mean']:.2f} cov {entry['coverage_mean']:.3f} stuck {entry['stuck_s_mean']:.1f}s "
              f"| best {best['fitness']:.2f} | {entry['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
