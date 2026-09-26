"""
Open-loop compass check through the agent's own code path (FlyPopulation + config/cx.yaml),
bodies held still: the steering neurons' left-right difference for 8 goal offsets at each of
several headings, fitted per heading to give a steering gain and phase error.

    python -m amongusfly.experiments.cx_agent_check --headings 16
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.brain.lif_torch import LIFBrain
from amongusfly.paths import DOCS_DIR
from amongusfly.world.grid import OccupancyGrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--headings", type=int, default=4)
    a = ap.parse_args()
    grid = OccupancyGrid.arena(20, 20)
    offs = np.arange(8) * 45.0  # world goal - heading, degrees (+ = goal to the left)
    heads = np.arange(a.headings) * (360.0 / a.headings)
    H, O = np.meshgrid(heads, offs, indexing="ij")
    H, O = H.ravel(), O.ravel()
    n = len(H)
    heading = np.deg2rad(H)
    goal = np.deg2rad(H + O)
    pop = FlyPopulation(a.graph, np.zeros(n), np.zeros(n), heading.copy(), grid, seed=1,  # arena is centred on 0
                        brain=LIFBrain(tag=a.graph),
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": 1.0, "goal": 1.0})
    pop.brain.keep_mask = None
    dt = pop.tick_ms / 1000
    T = int(a.seconds / dt)
    acc = {k: np.zeros(n) for k in ("DNa02", "DNa03", "PFL2", "EPGin")}
    cnt = 0
    for t in range(T):
        res, _ = pop.tick([], goal_angle=goal, movable=np.zeros(n, bool))
        if t >= T // 3:
            hz = res.dn_hz
            acc["DNa02"] += hz["DNa02"]["L"] - hz["DNa02"]["R"]
            acc["DNa03"] += hz["DNa03"]["L"] - hz["DNa03"]["R"]
            acc["PFL2"] += (hz["PFL2"]["L"] + hz["PFL2"]["R"]) / 2
            cnt += 1
    assert np.allclose(np.cos(pop.body.heading - heading), 1.0), "bodies should be held still"
    out = {"cx_cfg": pop.cx_cfg, "by_offset": {}}
    print("world offset  DNa02 L-R   DNa03 L-R   PFL2   (mean over 4 headings; + = goal to the left)")
    for o in offs:
        m = O == o
        row = {k: float(acc[k][m].mean() / cnt) for k in ("DNa02", "DNa03", "PFL2")}
        row["DNa02_by_heading"] = [round(float(v / cnt), 2) for v in acc["DNa02"][m]]
        out["by_offset"][int(o)] = row
        print(f"   {o:5.0f}      {row['DNa02']:+7.2f}    {row['DNa03']:+7.2f}   {row['PFL2']:5.1f}   by heading {row['DNa02_by_heading']}")
    # steering amplitude per heading: fit DNa02 L-R = A sin(offset) + B cos(offset) + C over the 8 offsets
    X = np.stack([np.sin(np.deg2rad(offs)), np.cos(np.deg2rad(offs)), np.ones(len(offs))], axis=1)
    out["by_heading"] = {}
    print("heading   steering gain A (Hz, + = correct)   phase error (deg)")
    for h in heads:
        y = acc["DNa02"][H == h] / cnt
        A, B, C = np.linalg.lstsq(X, y, rcond=None)[0]
        amp, ph = float(np.hypot(A, B)), float(np.rad2deg(np.arctan2(B, A)))
        out["by_heading"][int(h)] = {"sin_gain": float(A), "amp": amp, "phase_err_deg": ph, "offset_hz": float(C)}
        print(f"  {h:5.1f}      {A:+7.1f}   (amp {amp:5.1f})              {ph:+6.0f}")
    if a.out:
        (DOCS_DIR / f"{a.out}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
