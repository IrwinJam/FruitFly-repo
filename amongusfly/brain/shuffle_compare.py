"""
Compare the real graph against a set of shuffled graphs using a lateralization
index (LI), computed from a sanity_checks results file:

    LI = (DN_L - DN_R | left drive) - (DN_L - DN_R | right drive)

Positive LI = the DN pair responds more on the same side as the input. The
empirical p-value is (1 + #shuffles with LI >= real LI) / (1 + #shuffles).
With 10 shuffles the smallest possible p is 1/11 ~ 0.09, so this reports
effect-vs-null separation, not a strong significance test on its own.
"""
from __future__ import annotations

import argparse
import json

from amongusfly.paths import DOCS_DIR


def li(cond: dict, dn: str, prefix: str, rate: str) -> float:
    L = cond[f"{prefix}_L_{rate}"]["readout_hz"]
    R = cond[f"{prefix}_R_{rate}"]["readout_hz"]
    return (L[f"{dn}_L"][0] - L[f"{dn}_R"][0]) - (R[f"{dn}_L"][0] - R[f"{dn}_R"][0])


def drive_response(cond: dict, dn: str, prefix: str, rate: str) -> float:
    """Mean of both DN sides under left+right drive, minus no-input baseline."""
    vals = [cond[f"{prefix}_{s}_{rate}"]["readout_hz"][f"{dn}_{side}"][0] for s in "LR" for side in "LR"]
    base = [cond["none"]["readout_hz"][f"{dn}_{side}"][0] for side in "LR"]
    return sum(vals) / 4 - sum(base) / 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/brain_sensory_checks.json")
    ap.add_argument("--real", default="navcore")
    args = ap.parse_args()

    data = {r["tag"]: r for r in json.loads((DOCS_DIR / args.file).read_text())}
    real = data[args.real]
    shufs = [r for t, r in data.items() if t != args.real and "_shuf" in t]
    rates = sorted({k.split("_")[-1] for k in real["conditions"] if k.startswith("pursuit_")}, key=float)

    rows = []
    for rate in rates:
        for metric, dn, prefix, fn in [
            ("LI", "DNa02", "pursuit", li), ("LI", "DNa03", "pursuit", li), ("LI", "DNp01", "loom", li),
            ("drive", "DNp01", "loom", drive_response),
        ]:
            rv = fn(real["conditions"], dn, prefix, rate)
            sv = [fn(s["conditions"], dn, prefix, rate) for s in shufs]
            p = (1 + sum(1 for x in sv if x >= rv)) / (1 + len(sv))
            rows.append({"rate_hz": float(rate), "metric": metric, "dn": dn, "input": prefix, "real": round(rv, 2),
                         "shuffle_mean": round(sum(sv) / len(sv), 2), "shuffle_max": round(max(sv), 2),
                         "n_shuffles": len(sv), "empirical_p": round(p, 3)})
            print(f"{rate:>4} Hz {prefix:7s} {metric:5s} {dn}: real {rv:+8.1f} | shuffles mean {sum(sv)/len(sv):+7.1f} "
                  f"max {max(sv):+7.1f} (n={len(sv)}) | p={p:.3f}")
    out = DOCS_DIR / args.file.replace(".json", "_vs_shuffles.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out.name}")
