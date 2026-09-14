"""
Phase 1 diagnostic: what sustains the ignited state, and is it an artifact of our
neurotransmitter sign rules?

1. Ignite pruned5 with a single-neuron pulse (DNa02-L, 100 Hz, 50 ms) and list
   the neurons still firing afterwards: types, superclasses, NT, NT confidence.
2. Re-run the same pulse with sign-rule variants, each changing only the
   presynaptic sign of one NT group, to see which (if any) removes ignition:
     baseline        -- current rules (ACh/DA/5-HT/OA = +, GABA/Glu/His = -, unclear = +)
     modulatory_zero -- dopamine/serotonin/octopamine synapses set to 0
     unclear_zero    -- 'unclear' NT synapses set to 0
     both_zero       -- both of the above
     no_vnc          -- all VNC-region neurons silenced
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from flyseek.brain.lif_torch import LIFBrain, load_edges
from flyseek.brain.roles import neurons, role_idx
from flyseek.paths import DOCS_DIR

MODULATORY = {"dopamine", "serotonin", "octopamine"}


def pulse_and_measure(brain: LIFBrain, pulse_idx: list[int], seconds: float = 1.5):
    brain.reset(1, seed=0)
    brain.set_stimulus(pulse_idx, [0] * len(pulse_idx), [100.0] * len(pulse_idx))
    brain.run(int(50 / brain.dt_ms))
    brain.clear_stimulus()
    brain.run(int(500 / brain.dt_ms))  # let it settle
    out = brain.run(int((seconds * 1000 - 550) / brain.dt_ms))
    window_s = (seconds * 1000 - 550) / 1000
    return out["counts"][:, 0].cpu().numpy() / window_s  # rate per neuron, Hz


def main():
    df = neurons().sort_values("idx").reset_index(drop=True)
    edge_index, w, n = load_edges("pruned5")
    pre_nt = df["nt"].fillna("unclear").to_numpy()[edge_index[0]]
    pulse = role_idx("steering_high_gain", "L")[:1]

    report = {}

    # --- 1. who is in the ignited core (baseline rules) ---
    brain = LIFBrain(tag="pruned5", edges=(edge_index, w, n))
    rate = pulse_and_measure(brain, pulse)
    active = rate > 1.0
    core = df[active].assign(rate=rate[active])
    report["baseline_core"] = {
        "n_active_gt1hz": int(active.sum()),
        "mean_rate_active_hz": float(rate[active].mean()) if active.any() else 0.0,
        "nt_counts_active": core["nt"].fillna("missing").value_counts().to_dict(),
        "nt_counts_all": df["nt"].fillna("missing").value_counts().to_dict(),
        "frac_nt_confident_active": float(core["nt_confident"].mean()) if len(core) else None,
        "frac_nt_confident_all": float(df["nt_confident"].mean()),
        "superclass_active": core["superclass"].value_counts().head(12).to_dict(),
        "top_types_by_rate": (
            core.groupby("type")["rate"].agg(["count", "mean"]).sort_values("mean", ascending=False)
            .head(25).round(1).reset_index().to_dict(orient="records")
        ),
    }
    del brain
    torch.cuda.empty_cache()

    # --- 2. sign-rule variants ---
    variants = {
        "baseline": np.ones_like(w, dtype=bool),
        "modulatory_zero": ~np.isin(pre_nt, list(MODULATORY)),
        "unclear_zero": pre_nt != "unclear",
    }
    variants["both_zero"] = variants["modulatory_zero"] & variants["unclear_zero"]

    results = {}
    for name, keep in variants.items():
        wv = np.where(keep, w, 0.0).astype(np.float32)
        b = LIFBrain(tag="pruned5", edges=(edge_index, wv, n))
        r = pulse_and_measure(b, pulse)
        results[name] = {"n_active_gt1hz": int((r > 1).sum()), "pop_rate_hz": float(r.mean())}
        print(f"{name:16s} active>1Hz {results[name]['n_active_gt1hz']:6d} | pop {results[name]['pop_rate_hz']:.3f} Hz", flush=True)
        del b
        torch.cuda.empty_cache()

    vnc = df["superclass"].fillna("").str.startswith("vnc_").to_numpy() | df["superclass"].isin(
        ["efferent_ascending", "efferent_descending"]).to_numpy()
    b = LIFBrain(tag="pruned5", edges=(edge_index, w, n))
    b.silence(np.where(vnc)[0].tolist())
    r = pulse_and_measure(b, pulse)
    results["no_vnc"] = {"n_active_gt1hz": int((r > 1).sum()), "pop_rate_hz": float(r.mean())}
    print(f"{'no_vnc':16s} active>1Hz {results['no_vnc']['n_active_gt1hz']:6d} | pop {results['no_vnc']['pop_rate_hz']:.3f} Hz")

    report["variants"] = results
    (DOCS_DIR / "phase1_ignition_core.json").write_text(json.dumps(report, indent=2, default=str))
    c = report["baseline_core"]
    print("\nbaseline core:", c["n_active_gt1hz"], "neurons, mean", round(c["mean_rate_active_hz"], 1), "Hz")
    print("NT (active):", c["nt_counts_active"])
    print("NT-confident fraction: active", c["frac_nt_confident_active"], "vs all", round(c["frac_nt_confident_all"], 3))
    print("superclass (active):", c["superclass_active"])
    print("top types:", [(t["type"], t["count"], t["mean"]) for t in c["top_types_by_rate"][:15]])


if __name__ == "__main__":
    main()
