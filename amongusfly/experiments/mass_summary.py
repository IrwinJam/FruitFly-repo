"""
Pool the verification chunks (docs/results/mass/*.json): seeker win rate with a Wilson
95% interval, hider survival with a t interval and seeker wall contact, per hider count; then
test whether the smaller samples (recorded games, 20-game evaluations) are consistent with the
pooled rate.

    python -m amongusfly.experiments.mass_summary
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
from scipy import stats

from amongusfly.paths import DOCS_DIR, RESULTS_DIR


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def showcase(h: int, prefix: str = "showcase_v5") -> tuple[int, int]:
    wins = n = 0
    for f in glob.glob(str(RESULTS_DIR / "replays" / f"{prefix}_{h}h_s*.json")):
        over = next(e for e in json.load(open(f, encoding="utf-8"))["events"] if e["kind"] == "over")
        wins += over.get("winner") == "seeker"
        n += 1
    return wins, n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for h in (3, 5):
        chunks = sorted(glob.glob(str(DOCS_DIR / "results" / "mass" / f"{h}hiders_chunk*.json")))
        if not chunks:
            continue
        wins = n = 0
        surv, stuck = [], []
        for f in chunks:
            d = json.load(open(f, encoding="utf-8"))
            r = d["results"]["allbrain"]
            wins += r["seeker_wins"]
            n += d["n"]
            surv += r["hider_survival_s"]
            stuck.append((r["stuck_s_mean"], d["n"]))
        lo, hi = wilson(wins, n)
        s = np.asarray(surv, float)
        t = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
        row = {"games": n, "seeker_wins": wins, "win_rate": wins / n, "win_rate_ci95": [lo, hi],
               "hider_survival_s": float(s.mean()), "hider_survival_ci95": [float(s.mean() - t), float(s.mean() + t)],
               "seeker_wall_contact_s": float(sum(m * k for m, k in stuck) / n), "checks": {}}
        print(f"{h} hiders: {n} games | seeker wins {wins}/{n} = {wins / n:.1%} (95% CI {lo:.1%}–{hi:.1%}) | "
              f"hiders survive {s.mean():.1f} s ({s.mean() - t:.1f}–{s.mean() + t:.1f}) | "
              f"seeker wall contact {row['seeker_wall_contact_s']:.1f} s per game")
        k, m = showcase(h)
        smaller = {f"showcase ({m} recorded games)": (k, m)}
        for name in ("allbrain", "replication"):
            f = DOCS_DIR / "results" / f"{name}_300s_{h}hiders.json"
            if f.exists():
                d = json.load(open(f, encoding="utf-8"))
                smaller[f"{name} (20 games)"] = (d["results"]["allbrain"]["seeker_wins"], d["n"])
        for label, (k, m) in smaller.items():
            if not m:
                continue
            p = stats.binomtest(k, m, wins / n).pvalue  # is the small sample consistent with the pooled rate?
            row["checks"][label] = {"wins": k, "games": m, "p_vs_pooled": p}
            print(f"    {label:28s} {k}/{m} = {k / m:.0%}  consistent with pooled rate: "
                  f"{'yes' if p >= 0.05 else 'NO'} (binomial p = {p:.2f})")
        out[f"{h}_hiders"] = row
    (DOCS_DIR / "results" / "mass_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
