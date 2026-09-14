"""
Phase 1 follow-up: does removing Kenyon-cell -> Kenyon-cell synapses stop the
odor-driven, self-sustained mushroom-body activity seen at the calibrated scale?
Stimulus: ORN roles at 20 Hz for 500 ms, then off; measure 500-1500 ms after offset.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from flyseek.brain.lif_torch import LIFBrain, load_edges
from flyseek.brain.roles import neurons, role_idx
from flyseek.paths import DOCS_DIR


def trial(tag, ei, w, n, orn, is_kc):
    b = LIFBrain(tag=tag, edges=(ei, w, n))
    b.reset(1, seed=0)
    b.set_stimulus(orn, [0] * len(orn), [20.0] * len(orn))
    d = b.run(int(500 / b.dt_ms))
    b.clear_stimulus()
    b.run(int(500 / b.dt_ms))
    a = b.run(int(1000 / b.dt_ms))
    rd = d["counts"][:, 0].cpu().numpy() / 0.5
    ra = a["counts"][:, 0].cpu().numpy() / 1.0
    del b
    torch.cuda.empty_cache()
    return {
        "pop_during_hz": float(rd.mean()), "pop_after_hz": float(ra.mean()),
        "active_after_gt1hz": int((ra > 1).sum()),
        "kc_active_during_gt1hz": int((rd[is_kc] > 1).sum()),
        "kc_mean_rate_during_hz": float(rd[is_kc].mean()),
    }


def main():
    df = neurons().sort_values("idx").reset_index(drop=True)
    is_kc = df["type"].fillna("").str.startswith("KC").to_numpy()
    orn = sorted(set(role_idx("aversive_odor") + role_idx("attractive_odor")))
    out = {"kc_neurons": int(is_kc.sum()), "kc_nt": df.loc[is_kc, "nt"].fillna("missing").value_counts().to_dict()}
    print("KC neurons:", out["kc_neurons"], "NT:", out["kc_nt"])
    for tag in ("pruned5", "full"):
        ei, w, n = load_edges(tag)
        kckc = is_kc[ei[0]] & is_kc[ei[1]]
        out[tag] = {"kc_kc_edges": int(kckc.sum()),
                    "kc_kc_synapse_weight_frac_of_kc_input": float(np.abs(w[kckc]).sum() / np.abs(w[is_kc[ei[1]]]).sum())}
        for name, wv in (("baseline", w), ("no_KC_KC", np.where(kckc, 0, w).astype(np.float32))):
            r = trial(tag, ei, wv, n, orn, is_kc)
            out[tag][name] = r
            print(f"[{tag}] {name:9s} KC->KC edges {int(kckc.sum())} | during {r['pop_during_hz']:.3f} Hz | "
                  f"after-offset {r['pop_after_hz']:.3f} Hz, active {r['active_after_gt1hz']} | "
                  f"KCs active during {r['kc_active_during_gt1hz']} (mean {r['kc_mean_rate_during_hz']:.1f} Hz)", flush=True)
        print(f"[{tag}] KC->KC share of KC input weight: {out[tag]['kc_kc_synapse_weight_frac_of_kc_input']:.3f}")
    (DOCS_DIR / "phase1_kc_test.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
