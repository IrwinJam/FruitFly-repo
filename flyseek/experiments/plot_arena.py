"""Plot fly trajectories from an arena_pursuit replay (positions only)."""
from __future__ import annotations

import argparse
import json

import numpy as np

from flyseek.paths import DOCS_DIR, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(RESULTS_DIR / "replays" / "arena_pursuit_navcore"))
    ap.add_argument("--out", default=str(DOCS_DIR / "phase2_arena_trajectories.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.replay + ".npz")
    meta = json.loads(open(args.replay + ".json").read())
    states = data["states"]  # [T, A, F]
    tx, ty = np.array(meta["targets"]["x"]), np.array(meta["targets"]["y"])
    n = states.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    for f in range(n):
        left = ty[f] > 0
        col = "#f472b6" if left else "#60a5fa"
        ax.plot(states[:, f, 0], states[:, f, 1], color=col, alpha=0.6, lw=1)
    ax.scatter(tx[ty > 0][:1], ty[ty > 0][:1], s=200, c="#f472b6", marker="o", edgecolors="white", label="left target")
    ax.scatter(tx[ty < 0][:1], ty[ty < 0][:1], s=200, c="#60a5fa", marker="o", edgecolors="white", label="right target")
    ax.scatter([0], [0], c="white", marker="^", s=80, label="start (facing +x)")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.set_title(f"Closed-loop pursuit, {meta['graph']} ({n} flies, {meta['n_ticks'] * meta['tick_ms'] / 1000:.1f} s)\n"
                 "turning comes from the connectome; forward speed is an engineered constant",
                 color="white", fontsize=9)
    ax.legend(facecolor="black", labelcolor="white", fontsize=8, loc="lower left")
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="black")
    print("saved", args.out)


if __name__ == "__main__":
    main()
