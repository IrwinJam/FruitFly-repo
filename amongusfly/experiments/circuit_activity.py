"""
Circuit firing rates by role and game situation, from the spikes stored in recorded matches.

Situations per tick: seeker frozen / searching / chasing (a hider in view); hider safe / threatened
(seeker within danger range) / seen (in the seeker's line of sight). Circuits are marked "input"
when an encoder drives them directly and "computed" when the network produces the activity. Also
checks that the steering neurons' left-right difference matches the next turn (a pipeline
consistency check: the readout turns the fly by that difference).

    python -m amongusfly.experiments.circuit_activity --pattern "showcase_v4_*" --out circuit_activity
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from amongusfly.brain.roles import role_idx, type_idx
from amongusfly.experiments.clean_gate import visible
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.world.grid import OccupancyGrid

G = "navcore"
CIRCUITS = [  # (name, kind, cell types or role:<name>)
    ("Compass (EPG)", "input", ["EPG"]),
    ("Goal (FC2)", "input", ["FC2A", "FC2B", "FC2C"]),
    ("Visual pursuit (LC10a)", "input", ["LC10a"]),
    ("Looming (LC4, LPLC2)", "input", ["LC4", "LPLC2"]),
    # odour channels are omitted: their receptors have no somaSide, so the channel reaches no neurons
    ("Delta7 (compass ring)", "computed", ["Delta7"]),
    ("PFL3 (goal vs heading)", "computed", ["PFL3"]),
    ("PFL2 (goal behind)", "computed", ["PFL2"]),
    ("Steering (DNa02, DNa03)", "computed", ["DNa02", "DNa03"]),
    ("DNg13", "computed", ["DNg13"]),
    ("Escape (DNp01)", "computed", ["DNp01"]),
    ("Backward (MDN)", "computed", ["MDN"]),
]
SITUATIONS = {"seeker": ["frozen", "searching", "chasing"], "hider": ["safe", "threatened", "seen"]}


def members(spec: list[str], side: str | None = None) -> list[int]:
    out = []
    for s in spec:
        out += role_idx(s[5:], side, graph=G) if s.startswith("role:") else type_idx(s, side, graph=G)
    return sorted(set(out))


def situations(meta: dict, st: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    """[T, A] situation codes: seeker 0 frozen / 1 searching / 2 chasing; hider 0 safe / 1 threatened / 2 seen; -1 dead."""
    T, A = st.shape[:2]
    gc = meta["game_config"]
    rng, danger = gc["vision"]["seeker_range_units"], gc["danger_meter"]["max_range_units"]
    seek = next(e["tick"] for e in meta["events"] if e["kind"] == "phase" and e["phase"] == "seek")
    alive = st[:, :, 5] > 0.5
    code = np.full((T, A), -1, int)
    step = 5  # line of sight every 0.1 s, held in between
    for t0 in range(0, T, step):
        sl = slice(t0, min(T, t0 + step))
        s = st[t0, 0, :2]
        seen = [alive[t0, h] and visible(grid, s, st[t0, h, :2], rng) for h in range(A)]
        code[sl, 0] = 0 if t0 < seek else (2 if any(seen[1:]) else 1)
        for h in range(1, A):
            if not alive[t0, h]:
                continue
            d = float(np.hypot(*(st[t0, h, :2] - s)))
            code[sl, h] = 2 if (t0 >= seek and seen[h]) else (1 if (t0 >= seek and d <= danger) else 0)
    code[~alive] = -1
    return code


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="showcase_v[45]_*")
    ap.add_argument("--out", default="results/circuit_activity")
    a = ap.parse_args()
    grid = OccupancyGrid.skeld()
    C = len(CIRCUITS)
    sizes = np.array([len(members(spec)) for _, _, spec in CIRCUITS], float)
    lut = np.full(30000, C, int)  # C = "rest of the brain"
    for c, (_, _, spec) in enumerate(CIRCUITS):
        lut[members(spec)] = c
    steer_L, steer_R = members(["DNa02", "DNa03"], "L"), members(["DNa02", "DNa03"], "R")
    side = np.zeros(30000, int)
    side[steer_L], side[steer_R] = 1, -1

    sums = {r: np.zeros((3, C)) for r in SITUATIONS}
    ticks = {r: np.zeros(3) for r in SITUATIONS}
    lr_all, turn_all = {r: [] for r in SITUATIONS}, {r: [] for r in SITUATIONS}
    files = sorted(glob.glob(str(RESULTS_DIR / "replays" / f"{a.pattern}.npz")))
    for f in files:
        z = np.load(f)
        meta = json.loads(Path(f).with_suffix(".json").read_text(encoding="utf-8"))
        st, idx, cnt, off = z["states"], z["spike_idx"].astype(np.int64), z["spike_cnt"].astype(float), z["offsets"]
        T, A = st.shape[:2]
        block = np.repeat(np.arange(T * A), np.diff(off))
        per = np.bincount(block * (C + 1) + lut[idx], weights=cnt, minlength=T * A * (C + 1)).reshape(T, A, C + 1)[..., :C]
        lr = np.bincount(block, weights=cnt * side[idx], minlength=T * A).reshape(T, A)
        code = situations(meta, st, grid)
        dt = meta["tick_ms"] / 1000.0
        hd = st[:, :, 2]
        turn = ((np.roll(hd, -5, axis=0) - hd + np.pi) % (2 * np.pi) - np.pi)  # heading change over the next 0.1 s
        k = np.ones(5) / 5
        for ag in range(A):
            role = "seeker" if ag == 0 else "hider"
            for s in range(3):
                m = code[:, ag] == s
                sums[role][s] += per[m, ag].sum(axis=0)
                ticks[role][s] += m.sum()
            active = code[:-5, ag] >= (1 if role == "seeker" else 0)
            lr_s = np.convolve(lr[:, ag], k, mode="same")[:-5]  # 0.1 s of steering-neuron spikes
            lr_all[role].append(lr_s[active])
            turn_all[role].append(turn[:-5, ag][active])
        print(f"  {Path(f).stem}: {T} ticks", flush=True)
    dt = 0.02
    out = {"games": len(files), "pattern": a.pattern, "circuits": {}, "readout_consistency": {}}
    for c, (name, kind, spec) in enumerate(CIRCUITS):
        row = {"kind": kind, "neurons": int(sizes[c])}
        for role in SITUATIONS:
            for s, sit in enumerate(SITUATIONS[role]):
                n = ticks[role][s]
                row[f"{role}:{sit}"] = float(sums[role][s][c] / (sizes[c] * n * dt)) if n and sizes[c] else None
        out["circuits"][name] = row
    for role in SITUATIONS:
        x, y = np.concatenate(lr_all[role]), np.concatenate(turn_all[role])
        moving = np.abs(x) > 0  # ticks where the steering neurons fired asymmetrically
        agree = float(np.mean(np.sign(x[moving]) == np.sign(y[moving])))
        r = stats.spearmanr(x, y)
        out["readout_consistency"][role] = {"spearman_r": float(r.statistic), "p": float(r.pvalue),
                                               "sign_agreement": agree, "ticks": int(len(x))}
    for role in SITUATIONS:
        out["time_share"] = out.get("time_share", {})
        tot = ticks[role].sum()
        out["time_share"][role] = {sit: float(ticks[role][s] / tot) for s, sit in enumerate(SITUATIONS[role])}
    (DOCS_DIR / f"{a.out}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    cols = [f"{r}:{s}" for r in SITUATIONS for s in SITUATIONS[r]]
    print(f"\n{out['games']} games; mean firing rate per neuron (Hz)")
    print(f"{'circuit':28s} {'kind':8s} " + " ".join(f"{c.split(':')[1]:>10s}" for c in cols))
    print(f"{'':37s}" + "  ------ seeker ------   ------- hider -------")
    for name, row in out["circuits"].items():
        vals = " ".join(f"{row[c]:10.2f}" if row[c] is not None else f"{'-':>10s}" for c in cols)
        print(f"{name:28s} {row['kind']:8s} {vals}")
    for role, v in out["time_share"].items():
        print(f"time share {role}: " + ", ".join(f"{k} {p:.0%}" for k, p in v.items()))
    for role, v in out["readout_consistency"].items():
        print(f"consistency check ({role}): steering-neuron asymmetry vs next turn, Spearman r = {v['spearman_r']:.2f}, "
              f"same direction {v['sign_agreement']:.0%} of the time")


if __name__ == "__main__":
    main()
