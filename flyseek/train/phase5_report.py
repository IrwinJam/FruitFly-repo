"""
Phase 5 status report, regenerated after every overnight pipeline stage.

  python -m flyseek.train.phase5_report            -> docs/PHASE5_OVERNIGHT.md + docs/phase5_training_curves.png
  python -m flyseek.train.phase5_report --decide seeker_navcore
        prints "ok" if that trained seeker beats the Phase 4 explorer on the held-out
        evaluation (fitness diff > 0 and paired t p < 0.05, best or mean adapter), else "retrain"
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import numpy as np

from flyseek.paths import DATA_DIR, DOCS_DIR, RESULTS_DIR

RUNS = ["seeker_navcore", "hider_navcore", "seeker_navcore_v2", "hider_shuf0", "seeker_shuf0", "seeker_shuf0_v2"]
EVALS = ["phase5_eval_seeker", "phase5_eval_hider", "phase5_eval_seeker_v2", "phase5_eval_seeker_trained_ablations",
         "phase5_eval_hider_shuffle", "phase5_eval_seeker_shuffle"]


def load_log(run):
    p = RESULTS_DIR / "train" / run / "log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def decide(eval_name: str, trained_prefix: str = "trained") -> str:
    p = DOCS_DIR / f"{eval_name}.json"
    if not p.exists():
        return "retrain"
    res = json.loads(p.read_text())["results"]
    for label, r in res.items():
        if label.startswith(trained_prefix):
            v = r.get("vs_explorer")
            if v and v["fitness_diff_mean"] > 0 and v["paired_t_p"] < 0.05:
                return "ok"
    return "retrain"


def curves_png(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [r for r in runs if load_log(r)]
    if not have:
        return None
    fig, axes = plt.subplots(len(have), 1, figsize=(8, 2.6 * len(have)), squeeze=False)
    for ax, run in zip(axes[:, 0], have):
        L = load_log(run)
        g = [e["generation"] for e in L]
        ax.plot(g, [e["fitness_mean"] for e in L], label="population mean")
        ax.plot(g, [e["fitness_max"] for e in L], alpha=0.5, label="population max")
        k = 5
        m = np.convolve([e["fitness_mean"] for e in L], np.ones(k) / k, mode="valid") if len(L) >= k else []
        if len(m):
            ax.plot(g[k - 1:], m, "k--", label=f"{k}-gen running mean")
        ax.set_title(run)
        ax.set_ylabel("fitness")
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("generation")
    fig.tight_layout()
    out = DOCS_DIR / "phase5_training_curves.png"
    fig.savefig(out, dpi=100)
    return out


def report():
    lines = [f"# Phase 5 overnight status", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}. "
             "Regenerated after every pipeline stage; see `C:\\flyseek-data\\phase5_overnight.log`._", ""]
    plog = DATA_DIR / "phase5_overnight.log"
    if plog.exists():
        stages = [l for l in plog.read_text(errors="ignore").splitlines() if l.startswith("[stage]")]
        lines += ["## Pipeline stages", "", "```", *stages[-30:], "```", ""]
        errs = [l for l in plog.read_text(errors="ignore").splitlines() if re.search(r"Traceback|Error|CUDA", l)]
        if errs:
            lines += ["**Errors seen in the pipeline log:**", "", "```", *errs[-10:], "```", ""]

    lines += ["## Training runs", "",
              "| Run | Generations | Fitness mean, first 5 gens | Fitness mean, last 5 gens | Best single | Seeker win rate, last 5 | Stuck s, last 5 |",
              "|---|---|---|---|---|---|---|"]
    for run in RUNS:
        L = load_log(run)
        if not L:
            continue
        f5 = np.mean([e["fitness_mean"] for e in L[:5]])
        l5 = np.mean([e["fitness_mean"] for e in L[-5:]])
        wr = np.mean([e.get("seeker_win_rate", np.nan) for e in L[-5:]])
        st = np.mean([e["stuck_s_mean"] for e in L[-5:]])
        lines.append(f"| {run} | {len(L)} | {f5:.2f} | {l5:.2f} | {max(e['fitness_max'] for e in L):.2f} | {wr:.2f} | {st:.1f} |")
    png = curves_png(RUNS)
    if png:
        lines += ["", f"![training curves]({png.name})", ""]
    lines += ["Round-to-round swings mostly reflect which matches were drawn that round (every candidate in a round "
              "plays the same seeds). Only the held-out evaluations below decide whether training helped.", ""]

    for ev in EVALS:
        p = DOCS_DIR / f"{ev}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        role = d["role"]
        lines += [f"## Held-out: `{ev}` ({role}, {d['n']} matches, preset {d['preset']})", "",
                  "| Condition | Fitness (95% CI) | Seeker wins | " + ("Hider survival s (95% CI) | " if role == "hider" else "")
                  + "Stuck s | vs first condition (diff, paired t p) |",
                  "|---|---|---|" + ("---|" if role == "hider" else "") + "---|---|"]
        for label, r in d["results"].items():
            vs = next((v for k, v in r.items() if k.startswith("vs_")), None)
            f = r["fitness_ci"]
            row = f"| {label} | {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}] | {r['seeker_wins']}/{d['n']} | "
            if role == "hider":
                s = r["hider_survival_s_ci"]
                row += f"{s[0]:.1f} [{s[1]:.1f}, {s[2]:.1f}] | "
            row += (f"{r['stuck_s_mean']:.1f}" if r.get("stuck_s_mean") is not None else "–") + " | "
            row += (f"{vs['fitness_diff_mean']:+.3f}, p={vs['paired_t_p']:.2g}" if vs else "reference") + " |"
            lines.append(row)
        lines.append("")
    (DOCS_DIR / "PHASE5_OVERNIGHT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", DOCS_DIR / "PHASE5_OVERNIGHT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", default=None, help="eval name to decide on, e.g. phase5_eval_seeker")
    args = ap.parse_args()
    if args.decide:
        print(decide(args.decide))
    else:
        report()
