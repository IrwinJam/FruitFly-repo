"""
Diagnose exploration episodes: run flies with a trained (or initial) adapter, record
traces, and plot paths on The Skeld with wall-contact points marked. Also reports how
often, while stuck on a wall, the chosen goal points into the wall vs. away from it,
and the heading error to the goal.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flyseek.paths import DOCS_DIR, RESULTS_DIR
from flyseek.train.adapter import PARAM_SETS, decode, to_unit
from flyseek.train.explore_env import _world, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="explore_navcore")
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--init", action="store_true", help="use initial adapter instead of best.json")
    ap.add_argument("--out", default="phase4_diag_explore")
    ap.add_argument("--policy", default="route", choices=list(PARAM_SETS))
    ap.add_argument("--silence", nargs="*", default=[])
    args = ap.parse_args()

    params = PARAM_SETS[args.policy]
    if args.init:
        values = decode(params, to_unit(params))
    else:
        best = json.loads((RESULTS_DIR / "train" / args.run / "best.json").read_text())
        values = {k: v for k, v in best["values"].items()}
    seeds = list(range(900, 900 + args.n))
    res = run_episode(args.graph, values, seeds, seconds=args.seconds, return_traces=True, policy_kind=args.policy,
                      silence=args.silence or None)
    tr = res["trace"]  # [T, 4, A] x, y, heading, goal
    grid, _ = _world()

    x, y, hd, goal = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
    near_wall = grid.dist_at(x, y) < 0.35  # body radius 0.25 + one grid cell (dist values are quantized)
    err = np.abs((goal - hd + np.pi) % (2 * np.pi) - np.pi)
    free_goal = np.zeros_like(x)
    for t in range(0, len(x), 5):
        free_goal[t] = grid.raycast(x[t], y[t], goal[t][:, None], 3.0)[:, 0]
    sampled = np.zeros(len(x), bool)
    sampled[::5] = True
    stuck_s = sampled[:, None] & near_wall
    report = {
        "values": values,
        "rooms": res["rooms"].tolist(), "coverage": res["coverage"].tolist(), "stuck_s": res["stuck_s"].tolist(),
        "frac_time_near_wall": float(near_wall.mean()),
        "heading_error_deg_when_near_wall": float(np.rad2deg(err[near_wall]).mean()) if near_wall.any() else None,
        "heading_error_deg_when_free": float(np.rad2deg(err[~near_wall]).mean()),
        "goal_free_dist_when_near_wall_mean": float(free_goal[stuck_s].mean()) if stuck_s.any() else None,
        "frac_near_wall_goal_blocked_within_1u": float((free_goal[stuck_s] < 1.0).mean()) if stuck_s.any() else None,
        "mean_speed_units_per_s": float(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).mean() / 0.02),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "values"}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(grid.walkable, origin="lower", cmap="gray", alpha=0.35,
              extent=[grid.x0 - grid.res / 2, grid.x0 + grid.w * grid.res - grid.res / 2,
                      grid.y0 - grid.res / 2, grid.y0 + grid.h * grid.res - grid.res / 2])
    colors = plt.cm.tab10(np.arange(args.n))
    for a in range(args.n):
        ax.plot(x[:, a], y[:, a], color=colors[a], lw=1)
        ax.scatter(x[near_wall[:, a], a][::10], y[near_wall[:, a], a][::10], color=colors[a], s=6, marker="x")
        ax.scatter([x[0, a]], [y[0, a]], color="white", s=15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{'initial' if args.init else 'best'} adapter, {args.graph}, {args.seconds:.0f} s; x = wall contact",
                 color="white")
    fig.savefig(DOCS_DIR / f"{args.out}.png", dpi=110, bbox_inches="tight", facecolor="black")
    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
