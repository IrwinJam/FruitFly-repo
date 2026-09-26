"""
Walking quality of exploration flies from the Cafeteria spawn: rooms visited, exact wall
contact, long contacts, bounces per minute, time within one grid cell of a wall, ground speed as
a share of commanded, spinning and turn rate.

    python -m amongusfly.experiments.locomotion_quality --run explore_v5 --n 24 --seconds 120
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS, decode, to_unit
from amongusfly.train.explore_env import _world, run_episode

DT = 0.02
WINDOW = 100  # 2 s


def quality(trace: np.ndarray, speed: float) -> dict:
    grid, rooms = _world()
    x, y, hd = trace[:, 0], trace[:, 1], trace[:, 2]  # [T, A]
    clear = grid.dist_at(x, y)
    v = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) / DT
    dh = np.abs((np.diff(hd, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    spin = 0.0
    for a in range(0, len(v) - WINDOW, WINDOW):
        turn = dh[a:a + WINDOW].sum(axis=0)
        disp = np.hypot(x[a + WINDOW] - x[a], y[a + WINDOW] - y[a])
        spin += float(((turn > 2 * np.pi) & (disp < 1.0)).sum()) * WINDOW * DT
    cy, cx = grid._cell(x, y)
    # exact contact with wall geometry (grid.dist_at is quantised and hides contact; see clean_gate)
    from amongusfly.experiments.clean_gate import BODY_RADIUS, exact_clearance
    exact = exact_clearance(grid, x.ravel(), y.ravel()).reshape(x.shape)
    touch = exact < BODY_RADIUS + 1e-6
    bw = int(round(0.4 / DT))
    sharp = np.abs((hd[bw:] - hd[:-bw] + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(60.0)
    ev = sharp & touch[:-bw]
    bounces = float((ev[1:] & ~ev[:-1]).sum())
    run = np.zeros(x.shape[1])
    slide = 0.0
    for t in range(x.shape[0]):
        run = np.where(touch[t], run + 1, 0)
        slide += float((run * DT >= 0.5).sum())
    minutes = x.shape[0] * x.shape[1] * DT / 60
    return {"contact_frac": float(touch.mean()), "slide_frac": slide / touch.size,
            "bounces_per_min": bounces / minutes,
            "wall_frac": float((clear < 0.35).mean()), "speed_frac": float(v.mean() / speed),
            "spin_s_per_fly": spin / x.shape[1], "cafe_frac": float((rooms[cy, cx] == "Cafeteria").mean()),
            "turn_dps": float(np.degrees(dh.mean() / DT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="explore_v3")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--out", default="results/walking_quality")
    ap.add_argument("--conditions", nargs="+", default=[
        "trained_speed=1.5:repel=0", "match_speed=2.0:repel=0", "match_speed+centring=2.0:repel=1.5",
        "match_speed+strong_centring=2.0:repel=3.0"])
    args = ap.parse_args()
    params = PARAM_SETS["route"]
    src = json.loads((RESULTS_DIR / "train" / args.run / "best.json").read_text())["values"]
    base = decode(params, to_unit(params, src))
    if "gain:obstacle" in src:  # the obstacle sense travels with the run that learned it
        base["gain:obstacle"] = float(src["gain:obstacle"])
    brain = LIFBrain(tag="navcore")
    seeds = list(range(50000, 50000 + args.n))
    out = {}
    for c in args.conditions:
        label, spec = c.split("=", 1)
        spec, _, gain = spec.partition(":input=")  # optional compass and goal input strength, e.g. 0.75
        spec, _, obstacle = spec.partition(":obstacle=")  # optional override, e.g. 0 for the sense-off ablation
        speed, repel = spec.split(":repel=")
        vals = {**base, "policy:wall_repel": float(repel)}
        if obstacle:
            vals["gain:obstacle"] = float(obstacle)
        if gain:
            vals["gain:compass"] = vals["gain:goal"] = float(gain)
        r = run_episode("navcore", vals, seeds, seconds=args.seconds, brain=brain, return_traces=True, speed=float(speed))
        q = quality(r["trace"], float(speed))
        out[label] = {"speed": float(speed), "wall_repel": float(repel), "input_gain": float(gain or 1.0),
                      "rooms": float(r["rooms"].mean()),
                      "rooms_list": r["rooms"].tolist(), **q}
        print(f"{label:30s} rooms {out[label]['rooms']:4.2f} | contact {q['contact_frac']:5.1%} | "
              f"slides {q['slide_frac']:5.1%} | bounces {q['bounces_per_min']:4.1f}/min | "
              f"speed {q['speed_frac']:5.1%} "
              f"| spinning {q['spin_s_per_fly']:4.1f}s/fly | turn {q['turn_dps']:4.0f} deg/s",
              flush=True)
    (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
