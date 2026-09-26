"""
Sweep the connectome weight scale and neuromodulator sign for a regime that does not ignite
after a brief pulse but still carries sensory input to descending neurons. One seed per
configuration; the chosen one gets full statistics in sanity_checks.py.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from amongusfly.brain.lif_torch import LIFBrain, load_edges
from amongusfly.brain.roles import role_idx, type_idx
from amongusfly.paths import DOCS_DIR

COLS = ["none", "pulse_dna02", "pulse_lc10a", "lc10a_L", "lc10a_R", "loom_L", "broad"]
READOUT_TYPES = ["DNa02", "DNa03", "DNa01", "DNg13", "DNp01", "DNg100", "DNp09", "MDN", "pIP1"]


def stim_spec():
    photo_orn = sorted(set(
        role_idx("photoreceptor_achromatic") + role_idx("photoreceptor_color")
        + role_idx("aversive_odor") + role_idx("attractive_odor")))
    return {
        "pulse": {1: (role_idx("steering_high_gain", "L")[:1], 100.0), 2: (role_idx("target_motion_detector", "L"), 50.0)},
        "sustained": {
            3: (role_idx("target_motion_detector", "L"), 50.0),
            4: (role_idx("target_motion_detector", "R"), 50.0),
            5: (role_idx("looming_expansion", "L") + role_idx("looming_size", "L"), 50.0),
            6: (photo_orn, 20.0),
        },
    }


def to_lists(groups: dict):
    neu, cols, rates = [], [], []
    for c, (idx, r) in groups.items():
        neu += idx
        cols += [c] * len(idx)
        rates += [r] * len(idx)
    return neu, cols, rates


def run_config(tag: str, scale: float, mod_sign: float, seed: int = 0) -> dict:
    edges = load_edges(tag, weight_scale=scale, modulatory_sign=mod_sign)
    brain = LIFBrain(tag=tag, edges=edges)
    spec = stim_spec()
    brain.reset(len(COLS), seed=seed)

    brain.set_stimulus(*to_lists({**spec["pulse"], **spec["sustained"]}))
    brain.run(int(50 / brain.dt_ms))
    brain.set_stimulus(*to_lists(spec["sustained"]))
    brain.run(int(450 / brain.dt_ms))
    out = brain.run(int(1000 / brain.dt_ms))
    rate = out["counts"].cpu().numpy().astype(np.float64)  # spikes in 1 s = Hz, [N, B]

    readout = {}
    for t in READOUT_TYPES:
        for side in ("L", "R"):
            idx = type_idx(t, side)
            if idx:
                readout[f"{t}_{side}"] = [round(float(rate[idx, c].mean()), 2) for c in range(len(COLS))]

    col = {name: c for c, name in enumerate(COLS)}
    stim_rates = {name: float(rate[idx, col[name]].mean()) for name, (idx, _) in
                  {COLS[c]: v for c, v in spec["sustained"].items()}.items()}

    def asym(t, c):
        return readout.get(f"{t}_L", [0] * 7)[c] - readout.get(f"{t}_R", [0] * 7)[c]

    res = {
        "tag": tag, "weight_scale": scale, "modulatory_sign": mod_sign, "seed": seed,
        "pop_rate_hz": {name: round(float(rate[:, c].mean()), 4) for name, c in col.items()},
        "frac_active_gt1hz": {name: round(float((rate[:, c] > 1).mean()), 4) for name, c in col.items()},
        "measured_stim_rate_hz": {k: round(v, 1) for k, v in stim_rates.items()},
        "readout_hz": readout,
        "summary": {
            "ignites_after_dna02_pulse": bool((rate[:, col["pulse_dna02"]] > 1).mean() > 0.001),
            "ignites_after_lc10a_pulse": bool((rate[:, col["pulse_lc10a"]] > 1).mean() > 0.001),
            "pursuit_DNa02_asym_L_drive": asym("DNa02", col["lc10a_L"]),
            "pursuit_DNa02_asym_R_drive": asym("DNa02", col["lc10a_R"]),
            "pursuit_DNa03_asym_L_drive": asym("DNa03", col["lc10a_L"]),
            "pursuit_DNa03_asym_R_drive": asym("DNa03", col["lc10a_R"]),
            "loom_DNp01_L": readout.get("DNp01_L", [0] * 7)[col["loom_L"]],
            "loom_DNp01_R": readout.get("DNp01_R", [0] * 7)[col["loom_L"]],
            "broad_pop_rate_hz": round(float(rate[:, col["broad"]].mean()), 4),
        },
    }
    del brain
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pruned5")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75, 0.514, 0.4, 0.3])
    ap.add_argument("--mod-signs", type=float, nargs="+", default=[1, 0])
    ap.add_argument("--out", default="results/brain_calibration_sweep.json")
    args = ap.parse_args()

    results = []
    for mod in args.mod_signs:
        for s in args.scales:
            t0 = time.perf_counter()
            r = run_config(args.tag, s, mod)
            sm = r["summary"]
            print(f"scale {s:<5} mod {mod:<3} | ignite dna02 {str(sm['ignites_after_dna02_pulse']):5} lc10a {str(sm['ignites_after_lc10a_pulse']):5} "
                  f"| DNa02 L-R: Ldrive {sm['pursuit_DNa02_asym_L_drive']:+7.1f} Rdrive {sm['pursuit_DNa02_asym_R_drive']:+7.1f} "
                  f"| DNa03 L-R: {sm['pursuit_DNa03_asym_L_drive']:+7.1f} / {sm['pursuit_DNa03_asym_R_drive']:+7.1f} "
                  f"| DNp01 L/R {sm['loom_DNp01_L']:6.1f}/{sm['loom_DNp01_R']:6.1f} | broad pop {sm['broad_pop_rate_hz']:.3f} "
                  f"| {time.perf_counter()-t0:.0f}s", flush=True)
            results.append(r)
    (DOCS_DIR / args.out).write_text(json.dumps(results, indent=2))
    print(f"saved docs/{args.out}")
