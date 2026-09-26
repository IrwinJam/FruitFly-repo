"""
Print the results tables (markdown) straight from the result files, so numbers are pasted,
never retyped. Missing files are skipped.

    python -m amongusfly.experiments.results_tables
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from amongusfly.paths import DOCS_DIR

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pfmt(p: float) -> str:
    if p >= 0.001:
        return f"p = {p:.3f}".rstrip("0").rstrip(".") if p < 0.01 else f"p = {p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"p = {m}×10{str(int(e)).translate(SUP)}"


def load(name: str) -> dict | None:
    f = DOCS_DIR / "results" / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def explore():
    d = load("navigation")
    if not d:
        return
    print(f"### Navigation ({len(d['untrained']['rooms'])} held-out episodes)\n")
    print("| Condition | Rooms visited | Wall contact | vs untrained |\n|---|---|---|---|")
    labels = {"untrained": "Untrained", "trained": "**Trained**", "shuf_trained": "Shuffled wiring, same training budget",
              "pfl3_off": "PFL3 silenced"}
    for k, lab in labels.items():
        if k not in d:
            continue
        r = d[k]
        vs = r.get("vs_untrained")
        cmp_ = f"{vs['rooms_diff_mean']:+.2f}, {pfmt(vs['paired_t_p'])}" if vs else "-"
        print(f"| {lab} | {np.mean(r['rooms']):.2f} | {np.mean(r['stuck_s']):.1f} s | {cmp_} |")
    print()


def role(which: str):
    d = load("seeker_vs_scripted" if which == "seeker" else "hiders_vs_scripted")
    if not d:
        return
    r = d["results"]
    ref = "explorer"
    head = "Seeker (vs 3 scripted hiders)" if which == "seeker" else "Hiders (vs scripted seeker)"
    col = "Wins" if which == "seeker" else "Survival"
    print(f"### {head}: {d['n']} held-out {d['preset']} matches\n")
    print(f"| Condition | {col} | vs walker | vs the adapter it was derived from |\n|---|---|---|---|")
    for k, v in r.items():
        val = (f"{v['seeker_wins']}/{d['n']}" if which == "seeker"
               else f"{np.mean(v['hider_survival_s']):.1f} s")
        vs = v.get(f"vs_{ref}")
        cmp_ = f"{vs['fitness_diff_mean']:+.2f}, {pfmt(vs['paired_t_p'])}" if vs else "-"
        # an ablation is compared with its own unablated adapter: same seeds, so a paired test on per-match fitness
        spec = v.get("spec", "")
        parts = spec.split(":")
        base = next((bk for bk, bv in r.items()
                     if bk != k and len(parts) == 5 and bv.get("spec") == ":".join(parts[:4])), None)
        own = "-"
        if base and base != ref:
            a, b = np.asarray(v["fitness"]), np.asarray(r[base]["fitness"])
            own = f"{(a - b).mean():+.2f} vs {base}, {pfmt(stats.ttest_rel(a, b).pvalue)}"
        print(f"| {k} (`{spec or k}`) | {val} | {cmp_} | {own} |")
    print()


def allbrain():
    rows = []
    for name in ("allbrain_90s_3hiders", "allbrain_300s_3hiders", "allbrain_300s_5hiders"):
        d = load(name)
        if not d:
            continue
        ref = next(iter(d["results"]))
        for k, v in d["results"].items():
            vs = v.get(f"vs_{ref}")
            rows.append((f"{d['preset']}, {d['hiders']} hiders, {k}", f"{v['seeker_wins']}/{d['n']}",
                         f"{np.mean(v['hider_survival_s']):.1f} s" if "hider_survival_s" in v else "-",
                         pfmt(vs["paired_t_p"]) if vs else "-"))
    if not rows:
        return
    print("### Every fly a brain\n")
    print("| Setting | Seeker wins | Hider survival | vs trained hiders |\n|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    explore()
    role("seeker")
    role("hider")
    allbrain()
    loc = load("walking_quality")
    if loc:
        print("### Walking quality\n\n| Condition | Rooms | Wall contact | Bounces / min | Speed | Spinning |\n"
              "|---|---|---|---|---|---|")
        for k, v in loc.items():
            print(f"| {k} | {v['rooms']:.2f} | {v['contact_frac']:.0%} | {v['bounces_per_min']:.1f} | "
                  f"{v['speed_frac']:.0%} | {v['spin_s_per_fly']:.1f} s |")
        print()
    gate = load("quality_checks")
    if gate:
        print("### Clean-match gate\n\n| Check | Value | Needs | |\n|---|---|---|---|")
        for k, c in gate["checks"].items():
            print(f"| {k} | {c['value']:.3f} | {c['threshold']} | {'pass' if c['pass'] else '**FAIL**'} |")


if __name__ == "__main__":
    main()
