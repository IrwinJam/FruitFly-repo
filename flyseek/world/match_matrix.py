"""
Phase 4.0 baseline: run a matrix of matches and save their summaries.

Default matrix: spawn {default, spread} x brains {all, seeker, hiders} x seeds {0,1,2},
short preset, navcore brains, no spike recording (positions/events only) for speed.
Results append to docs/<out>.json so an interrupted run can be resumed (--resume skips
matches already recorded). The same script re-runs the matrix after Phase 4 training.
"""
from __future__ import annotations

import argparse
import json
import time

from flyseek.paths import DOCS_DIR
from flyseek.world.match import parse_brains, run_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawns", nargs="+", default=["default", "spread"])
    ap.add_argument("--brains", nargs="+", default=["all", "seeker", "hiders"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--preset", default="short")
    ap.add_argument("--hiders", type=int, default=3)
    ap.add_argument("--out", default="phase4_baseline_matrix")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    path = DOCS_DIR / f"{args.out}.json"
    done = json.loads(path.read_text()) if (args.resume and path.exists()) else []
    have = {r["name"] for r in done}
    for spawn in args.spawns:
        for b in args.brains:
            for seed in args.seeds:
                name = f"{args.out}_{spawn}_{b}_s{seed}"
                if name in have:
                    continue
                t0 = time.perf_counter()
                s = run_match(args.hiders, parse_brains(b, 1 + args.hiders), args.graph, args.preset, seed, name,
                              record_spikes=False, spawn=spawn)
                s["brains_spec"] = b
                done.append(s)
                path.write_text(json.dumps(done, indent=2, default=str))
                print(f"{name}: winner={s['winner']} t={s['sim_seconds']}s kills={len(s['kills'])} "
                      f"first_sighting={s['first_sighting_s']} rooms={s['n_rooms_visited']} "
                      f"coverage={s['coverage_frac']} ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
