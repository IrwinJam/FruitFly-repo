"""
Turn-response curve under a trained readout: turn rate as a function of heading error.

Flies in an open arena face a fixed goal direction from headings spread around the circle.
Binned by error, the curve shows bias (steady signed error), slope near zero, dead zones (errors
that produce almost no turn) and noise; it also records the raw steering-neuron drive and PFL2.

    python -m amongusfly.experiments.steer_curve --run explore_v5

Sensitivity: --weight-scale replaces the connection scale (config/brain.yaml) and --input-gain
multiplies the compass and goal input, to test whether the dead zone belongs to the wiring or to
these settings.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.brain.lif_torch import LIFBrain, load_edges
from amongusfly.motors.decoders import load_motor_config
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS, apply_decoder, decode, to_unit
from amongusfly.world.grid import OccupancyGrid

BINS = np.array([-180, -120, -90, -60, -40, -25, -15, -8, -3, 3, 8, 15, 25, 40, 60, 90, 120, 180])


def dead_zone(rows: list, key: str) -> list | None:
    """Heading errors around zero where the response stays below 10% of its peak: [low, high] in degrees."""
    peak = max(abs(r[key]) for r in rows)
    quiet = [abs(r[key]) < 0.1 * peak for r in rows]
    mid = next((i for i, r in enumerate(rows) if r["err_lo"] < 0 < r["err_hi"]), None)
    if mid is None or not quiet[mid]:
        return None
    lo = hi = mid
    while lo > 0 and quiet[lo - 1]:
        lo -= 1
    while hi < len(rows) - 1 and quiet[hi + 1]:
        hi += 1
    return [rows[lo]["err_lo"], rows[hi]["err_hi"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="explore_v5")
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override decoder values, e.g. decoder:turn.scale_hz=5")
    ap.add_argument("--weight-scale", type=float, default=None, help="connection scale (default: config/brain.yaml)")
    ap.add_argument("--input-gain", type=float, default=1.0, help="multiplies the compass and goal input rates")
    a = ap.parse_args()
    params = PARAM_SETS["route"]
    trained = None if a.run == "init" else json.loads((RESULTS_DIR / "train" / a.run / "best.json").read_text())["values"]
    values = decode(params, to_unit(params, trained))
    for kv in a.set:
        k, v = kv.split("=")
        values[k] = float(v)
    grid = OccupancyGrid.arena(60, 60)  # 8 s at 2 units/s stays far from the walls
    n = a.n
    h0 = np.linspace(-np.pi, np.pi, n, endpoint=False)  # goal = 0, so the starting error covers the circle
    pop = FlyPopulation(a.graph, np.zeros(n), np.zeros(n), h0.copy(), grid, seed=4242,  # arena is centred on 0
                        brain=LIFBrain(tag=a.graph, edges=load_edges(a.graph, weight_scale=a.weight_scale)),
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": a.input_gain, "goal": a.input_gain})
    pop.brain.keep_mask = None
    cfg = copy.deepcopy(load_motor_config())
    apply_decoder(cfg, values)
    pop.decoder.cfg = cfg
    dt = pop.tick_ms / 1000
    goal = np.zeros(n)
    errs, oms, drives, pfl2s, dna02, dna03 = [], [], [], [], [], []
    for _ in range(int(a.seconds / dt)):
        before = pop.body.heading.copy()
        err = (goal - before + np.pi) % (2 * np.pi) - np.pi  # + = goal is to the left
        res, _ = pop.tick([], goal_angle=goal, base_speed=np.full(n, a.speed))
        r = pop.decoder.rates
        drives.append(res.command.turn_drive.copy())
        pfl2s.append((r["PFL2"]["L"] + r["PFL2"]["R"]) / 2 if "PFL2" in r else np.zeros(n))
        dna02.append(r["DNa02"]["L"] - r["DNa02"]["R"])
        dna03.append(r["DNa03"]["L"] - r["DNa03"]["R"])
        om = ((pop.body.heading - before + np.pi) % (2 * np.pi) - np.pi) / dt
        errs.append(np.rad2deg(err))
        oms.append(om)
    E, O = np.stack(errs), np.stack(oms)  # [T, n]
    D, P2, A2, A3 = (np.stack(v).ravel() for v in (drives, pfl2s, dna02, dna03))
    tail = E[int(len(E) * 2 / 3):]
    rows = []
    idx = np.digitize(E.ravel(), BINS) - 1
    for b in range(len(BINS) - 1):
        m = idx == b
        if m.sum() < 20:
            continue
        o = O.ravel()[m]
        rows.append({"err_lo": int(BINS[b]), "err_hi": int(BINS[b + 1]), "n": int(m.sum()),
                     "omega_mean": float(o.mean()), "omega_sd": float(o.std()),
                     "drive_hz": float(D[m].mean()), "dna02_lr_hz": float(A2[m].mean()), "dna03_lr_hz": float(A3[m].mean()),
                     "pfl2_hz": float(P2[m].mean()),
                     "ideal": float(np.clip(10 * np.deg2rad((BINS[b] + BINS[b + 1]) / 2), -values["decoder:turn.max_omega_rad_per_s"],
                                            values["decoder:turn.max_omega_rad_per_s"]))})
    near = np.abs(E.ravel()) < 15
    slope = float(np.polyfit(np.deg2rad(E.ravel()[near]), O.ravel()[near], 1)[0]) if near.sum() > 50 else float("nan")
    out = {"run": a.run, "weight_scale": a.weight_scale, "input_gain": a.input_gain,
           "dead_zone_deg": {"dna02": dead_zone(rows, "dna02_lr_hz"), "turn": dead_zone(rows, "omega_mean")},
           "decoder": {k: v for k, v in values.items() if k.startswith("decoder:")},
           "steady_signed_err_deg": float(tail.mean()), "steady_abs_err_deg": float(np.abs(tail).mean()),
           "steady_err_sd_deg": float(tail.std()), "slope_near_zero_per_s": slope, "curve": rows}
    print(f"{a.run}: steady error mean {out['steady_signed_err_deg']:+.1f} deg (bias), |err| {out['steady_abs_err_deg']:.1f}, "
          f"sd {out['steady_err_sd_deg']:.1f}; slope near 0: {slope:.2f} (rad/s per rad; ideal 10)")
    print(" error bin       n   turn mean +- sd (rad/s)  ideal | drive Hz  DNa02 L-R  DNa03 L-R  PFL2 Hz")
    for r in rows:
        print(f" {r['err_lo']:5d}..{r['err_hi']:<5d} {r['n']:6d}   {r['omega_mean']:+6.2f} +- {r['omega_sd']:4.2f}      {r['ideal']:+5.2f} |"
              f" {r['drive_hz']:+7.2f}  {r['dna02_lr_hz']:+8.2f}  {r['dna03_lr_hz']:+8.2f}  {r['pfl2_hz']:6.1f}")
    print(f"dead zone (below 10% of peak): DNa02 {out['dead_zone_deg']['dna02']}, turn {out['dead_zone_deg']['turn']}")
    if a.out:
        path = DOCS_DIR / f"{a.out}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
