"""
Which visual cell types can steer the fly?

Drives each candidate visual cell type on one side, through the agent's own code path, with the
compass and goal bumps aligned (so baseline steering is zero) and bodies held still. Measures the
steering neurons' left-right difference (DNa02 + DNa03) and the escape neuron DNp01.

  steering index = (L-R | left drive) - (L-R | right drive)   [Hz]
  > 0: driving one side turns the fly toward it (feed walls contralaterally)
  < 0: driving one side turns it away (feed walls ipsilaterally)

The selection rule picks the largest |index| among types not already used as inputs, with
|index| >= 5 Hz and a DNp01 rise under 5 Hz.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.brain.roles import full_idx_of, neurons, type_idx
from amongusfly.paths import DOCS_DIR
from amongusfly.world.grid import OccupancyGrid

USED = {"LC10a", "LC4", "LPLC2"}
PATTERN = re.compile(r"^(LC\d+[a-z]?|LPLC\d|H[12]|DCH|VCH|LPC\d|LLPC\d|LPi\w*|Tm\d+\w*)$")
MIN_INDEX_HZ, MAX_DNP01_HZ = 5.0, 5.0


def candidates(graph: str) -> list[str]:
    df = neurons()
    full = set(int(i) for i in full_idx_of(graph))
    df = df[df["idx"].astype(int).isin(full) & df["type"].fillna("").str.match(PATTERN)]
    sides = df.groupby("type")["somaSide"].agg(lambda s: {"L", "R"} <= set(s))
    return sorted(t for t, ok in sides.items() if ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--rate", type=float, default=40.0)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    types = a.only or candidates(a.graph)
    conds = [(t, s) for t in types for s in "LR"] + [(None, None)]
    cols = [c for c in conds for _ in range(a.reps)]
    n = len(cols)
    heading = np.zeros(n)
    pop = FlyPopulation(a.graph, np.zeros(n), np.zeros(n), heading.copy(), OccupancyGrid.arena(20, 20),
                        seed=7, brain=LIFBrain(tag=a.graph),
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": 1.0, "goal": 1.0})
    pop.brain.keep_mask = None
    extra = {}
    for t in types:  # one probe channel per candidate type
        ch = f"probe_{t}"
        pop.chan_idx[ch] = {s: np.array(type_idx(t, s, graph=a.graph), dtype=np.int64) for s in "LR"}
        pop.channel_gain[ch] = 1.0
        extra[ch] = {s: np.array([a.rate if c == (t, s) else 0.0 for c in cols]) for s in "LR"}
    T = int(a.seconds / (pop.tick_ms / 1000))
    lr, esc, cnt = np.zeros(n), np.zeros(n), 0
    for k in range(T):
        res, _ = pop.tick([], extra_rates=extra, goal_angle=heading, movable=np.zeros(n, bool))
        if k >= T // 3:
            hz = res.dn_hz
            lr += (hz["DNa02"]["L"] - hz["DNa02"]["R"]) + (hz["DNa03"]["L"] - hz["DNa03"]["R"])
            esc += (hz["DNp01"]["L"] + hz["DNp01"]["R"]) / 2
            cnt += 1
    lr, esc = lr / cnt, esc / cnt
    base = np.array([c == (None, None) for c in cols])
    base_lr, base_esc = float(lr[base].mean()), float(esc[base].mean())
    rows = {}
    for t in types:
        mL = np.array([c == (t, "L") for c in cols])
        mR = np.array([c == (t, "R") for c in cols])
        idx = float(lr[mL].mean() - lr[mR].mean())
        rows[t] = {"index_hz": round(idx, 2), "lr_left_drive": round(float(lr[mL].mean()), 2),
                   "lr_right_drive": round(float(lr[mR].mean()), 2),
                   "dnp01_rise_hz": round(float(esc[mL | mR].mean()) - base_esc, 2),
                   "n_L": len(type_idx(t, "L", graph=a.graph)), "n_R": len(type_idx(t, "R", graph=a.graph))}
    ranked = sorted(rows.items(), key=lambda kv: -abs(kv[1]["index_hz"]))
    print(f"baseline L-R {base_lr:+.2f} Hz, DNp01 {base_esc:.1f} Hz | drive {a.rate:.0f} Hz, {a.reps} reps")
    print(f"{'type':8s} {'index':>7s} {'L drive':>8s} {'R drive':>8s} {'DNp01+':>7s}  n(L/R)")
    for t, r in ranked[:20]:
        print(f"{t:8s} {r['index_hz']:+7.2f} {r['lr_left_drive']:+8.2f} {r['lr_right_drive']:+8.2f} "
              f"{r['dnp01_rise_hz']:+7.2f}  {r['n_L']}/{r['n_R']}{'  (in use)' if t in USED else ''}")
    ok = [(t, r) for t, r in ranked if t not in USED and abs(r["index_hz"]) >= MIN_INDEX_HZ
          and r["dnp01_rise_hz"] < MAX_DNP01_HZ]
    choice = ok[0][0] if ok else "LC10a"
    lat = None
    if choice in rows:
        lat = "contra" if rows[choice]["index_hz"] > 0 else "ipsi"
    print(f"selected: {choice} (feed walls {lat}laterally)" if lat else f"selected: {choice} (fallback)")
    if a.out:
        (DOCS_DIR / f"{a.out}.json").write_text(json.dumps(
            {"graph": a.graph, "rate_hz": a.rate, "reps": a.reps, "baseline_lr_hz": base_lr,
             "rule": f"max |index| among types not in {sorted(USED)}, |index| >= {MIN_INDEX_HZ} Hz, "
                     f"DNp01 rise < {MAX_DNP01_HZ} Hz; else LC10a",
             "selected": choice, "laterality": lat, "types": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
