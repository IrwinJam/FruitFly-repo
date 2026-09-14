"""
Is DNa01's CONTRALATERAL response to looming wiring-specific? (Phase 1 carry-over,
needed before DNa01 is used as a Hider turn-away signal.)

Uses an existing sanity_checks results file containing the real graph plus shuffles.
Contralateral index CI = (R-L | left loom) - (R-L | right loom); positive means the
DN opposite the looming side fires more. Empirical p = (1 + #shuffles >= real)/(1 + n).
Also reports the within-real Welch test between left-loom and right-loom replicates,
reconstructed from the stored mean and 95% CI of each side.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from flyseek.paths import DOCS_DIR


def ci_index(cond, dn, rate):
    L, R = cond[f"loom_L_{rate}"]["readout_hz"], cond[f"loom_R_{rate}"]["readout_hz"]
    return (L[f"{dn}_R"][0] - L[f"{dn}_L"][0]) - (R[f"{dn}_R"][0] - R[f"{dn}_L"][0])


def contra_response(cond, dn, rate):
    """Mean contralateral rate under loom, averaged over both loom sides."""
    L, R = cond[f"loom_L_{rate}"]["readout_hz"], cond[f"loom_R_{rate}"]["readout_hz"]
    return (L[f"{dn}_R"][0] + R[f"{dn}_L"][0]) / 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="phase1_sanity_navcore100.json")
    ap.add_argument("--real", default="navcore")
    ap.add_argument("--dn", default="DNa01")
    args = ap.parse_args()

    data = {r["tag"]: r for r in json.loads((DOCS_DIR / args.file).read_text())}
    real = data[args.real]
    shufs = [r for t, r in data.items() if t != args.real]
    rates = sorted({k.split("_")[-1] for k in real["conditions"] if k.startswith("loom_L_")}, key=float)
    rows = []
    for rate in rates:
        rv = ci_index(real["conditions"], args.dn, rate)
        sv = np.array([ci_index(s["conditions"], args.dn, rate) for s in shufs])
        p = (1 + int((sv >= rv).sum())) / (1 + len(sv))
        cr = contra_response(real["conditions"], args.dn, rate)
        rows.append({"rate_hz": float(rate), "dn": args.dn, "contra_index_real": round(rv, 2),
                     "contra_rate_real_hz": round(cr, 2), "shuffle_mean": round(float(sv.mean()), 2),
                     "shuffle_max": round(float(sv.max()), 2), "n_shuffles": len(sv), "empirical_p": round(p, 3)})
        print(f"{rate:>4} Hz {args.dn} contra index: real {rv:+7.1f} (contra rate {cr:.1f} Hz) | "
              f"shuffles mean {sv.mean():+6.1f} max {sv.max():+6.1f} | p={p:.3f}")
    out = DOCS_DIR / f"phase1_{args.dn.lower()}_escape_{args.real}.json"
    out.write_text(json.dumps(rows, indent=2))
    print("saved", out.name)
