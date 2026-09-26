"""
Draw the seeker's path over the map, so walking quality can be judged by eye.

Showcase mode: one panel per recorded match, with wall contact and catches marked.
Verification mode (--mass): every seeker path from the 300 verification games, one panel per hider
count, from positions saved by eval_role --save-paths. The GPU simulation is not bit-for-bit repeatable,
so a rerun of the same seeds gives different games; its win count is compared with the published run.

    python -m amongusfly.experiments.path_figure --pattern "showcase_v4_*" --out seeker_paths
    python -m amongusfly.experiments.path_figure --mass mass --out seeker_paths_300
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from amongusfly.experiments.clean_gate import BODY_RADIUS, exact_clearance  # noqa: E402
from amongusfly.paths import DOCS_DIR, RESULTS_DIR  # noqa: E402
from amongusfly.world.grid import OccupancyGrid  # noqa: E402
from amongusfly.world.rules import load_game_config  # noqa: E402

INK, BLUE, ORANGE = "#111827", "#1F3A5F", "#B8741A"
CONTACT_WORD = "amber"  # name of the wall-contact colour in the figure heading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="showcase_v4_*")
    ap.add_argument("--out", default="seeker_paths")
    ap.add_argument("--agent", type=int, default=0, help="0 = seeker")
    ap.add_argument("--mass", default=None, help="name of a saved-paths folder (eval_role --save-paths)")
    a = ap.parse_args()
    grid = OccupancyGrid.skeld()
    if a.mass:
        return mass(grid, a.mass, a.out)
    files = sorted(glob.glob(str(RESULTS_DIR / "replays" / f"{a.pattern}.npz")))
    if not files:
        raise SystemExit(f"no replays matching {a.pattern}")
    cols = 3
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
    extent = [grid.x0, grid.x0 + grid.w * grid.res, grid.y0, grid.y0 + grid.h * grid.res]
    for ax, f in zip(axes.flat, files):
        meta = json.loads(Path(f).with_suffix(".json").read_text(encoding="utf-8"))
        st = np.load(f)["states"]
        seek = next(e["tick"] for e in meta["events"] if e["kind"] == "phase" and e["phase"] == "seek")
        x, y = st[seek:, a.agent, 0], st[seek:, a.agent, 1]
        ax.imshow(grid.walkable, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=3.2, interpolation="nearest")
        ax.plot(x, y, color=BLUE, lw=0.6, alpha=0.8, solid_capstyle="round")
        touch = exact_clearance(grid, x, y) < BODY_RADIUS + 1e-6  # the body is in contact with a wall
        ax.scatter(x[touch], y[touch], s=2.2, color=ORANGE, alpha=0.9, linewidths=0, zorder=3)
        kills = [e for e in meta["events"] if e["kind"] == "kill"]
        ax.scatter([e["x"] for e in kills], [e["y"] for e in kills], s=22, facecolor="none", edgecolor=INK, lw=1.0)
        label = re.sub(r"^showcase_v\d+_", "", Path(f).stem)
        ax.set_title(f"{label} · touching a wall {touch.mean():.0%}",
                     loc="left", fontsize=8, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.flat[len(files):]:
        ax.axis("off")
    fig.suptitle(f"Seeker path per match ({CONTACT_WORD}: body touching a wall; circles: catches)",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    out = DOCS_DIR / f"{a.out}.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


def mass(grid: OccupancyGrid, name: str, out_name: str):
    files = sorted(glob.glob(str(RESULTS_DIR / "paths" / name / "*.npz")))
    if not files:
        raise SystemExit(f"no saved paths in {RESULTS_DIR / 'paths' / name}")
    cfg = load_game_config("full")
    T, dt = cfg["timers"]["round_length_s"], 0.02
    seek0 = int(cfg["timers"]["hide_phase_s"] / dt)
    groups: dict[int, list] = {}
    for f in files:
        d = np.load(f)
        groups.setdefault(d["survival_s"].shape[1], []).append(d)
    extent = [grid.x0, grid.x0 + grid.w * grid.res, grid.y0, grid.y0 + grid.h * grid.res]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.4 * len(groups), 4.0), squeeze=False)
    for ax, (H, ds) in zip(axes.flat, sorted(groups.items())):
        # check each game against the published per-game seeker fitness (same seeds, same batching)
        pub = {}
        for f in sorted((DOCS_DIR / "results" / "mass").glob(f"{H}hiders_chunk*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            base = 100000 if H == 3 else 200000
            c = int(re.search(r"chunk(\d+)", f.stem).group(1))
            for k, fit in enumerate(j["results"]["allbrain"]["fitness"]):
                pub[base + c * 50 + k] = fit
        ax.imshow(grid.walkable, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=3.2, interpolation="nearest")
        n = wins = checked = same = 0
        contact = []
        for d in ds:
            x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
            surv = d["survival_s"]
            caught = surv < T
            fit = (caught * (1.0 + (T - surv) / T)).sum(axis=1) / H
            for m, seed in enumerate(d["seeds"]):
                end = min(int(round(d["end_s"][m] / dt)), len(x) - 1)
                px, py = x[seek0:end, m, 0], y[seek0:end, m, 0]
                ax.plot(px, py, color=BLUE, lw=0.4, alpha=0.06)
                contact.append(exact_clearance(grid, px, py) < BODY_RADIUS + 1e-6)
                n += 1
                wins += bool(d["seeker_win"][m])
                if int(seed) in pub:
                    checked += 1
                    same += abs(pub[int(seed)] - fit[m]) < 1e-6
        frac = float(np.concatenate(contact).mean())
        if checked:  # compare win counts with the published run of the same seeds (Fisher exact test)
            pw = sum(json.loads(f.read_text(encoding="utf-8"))["results"]["allbrain"]["seeker_wins"]
                     for f in (DOCS_DIR / "results" / "mass").glob(f"{H}hiders_chunk*.json"))
            p = stats.fisher_exact([[wins, n - wins], [pw, len(pub) - pw]]).pvalue
            check = f"published run {pw}/{len(pub)}, Fisher p = {p:.2f}; {same} of {checked} games identical"
        else:
            check = "no published run to compare"
        print(f"{H} hiders: {n} games, seeker wins {wins}/{n}, seeker touching a wall {frac:.1%}; {check}")
        ax.set_title(f"{H} hiders · {n} games · seeker wins {wins}/{n} · touching a wall {frac:.0%}",
                     loc="left", fontsize=9, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Seeker paths in the 300 verification games (seek phase; darker = walked more often)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = DOCS_DIR / f"{out_name}.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
