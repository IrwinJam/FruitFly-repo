"""
Write viewer/public/replays/showcase_stats.json, the statistics shown on a showcase replay's
end card in the interactive viewer, from the result files.

    python -m amongusfly.experiments.showcase_stats
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from amongusfly.paths import DOCS_DIR, REPO_ROOT

OUT = REPO_ROOT / "viewer" / "public" / "replays" / "showcase_stats.json"
DISCLOSURE = ("Engineered: senses, route planning, forward speed, auto-vent, camping. "
              "Never trained: the connectome itself.")


def load(name: str) -> dict | None:
    f = DOCS_DIR / "results" / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="showcase_v4")
    a = ap.parse_args()
    full3, short3, full5 = (load(f"allbrain_{n}") for n in ("300s_3hiders", "90s_3hiders", "300s_5hiders"))
    if not full3 or not short3:
        raise SystemExit("need docs/results/allbrain_300s_3hiders.json and allbrain_90s_3hiders.json")
    surv = lambda d, k: np.mean(d["results"][k]["hider_survival_s"])
    wins = lambda d, k: f"{d['results'][k]['seeker_wins']}/{d['n']}"

    mass = load("mass_verification")
    circ = load("seeker_circuits_300s")
    three = []
    if mass and "3_hiders" in mass:
        m3 = mass["3_hiders"]
        three.append(f"Over {m3['games']} more games on fresh seeds the seeker wins {m3['win_rate']:.0%} "
                     f"(95% CI {m3['win_rate_ci95'][0]:.0%}–{m3['win_rate_ci95'][1]:.0%}).")
    else:
        three.append(f"Held-out, same setup ({full3['n']} matches, 300 s): seeker wins {wins(full3, 'allbrain')}, "
                     f"hiders survive {surv(full3, 'allbrain'):.0f} s on average.")
    if circ:
        r = circ["results"]
        three.append(f"Silence 24 PFL3 or 12 PFL2 steering neurons in the seeker: wins fall from "
                     f"{r['walker']['seeker_wins']}/{circ['n']} to {r['pfl3_off']['seeker_wins']}/{circ['n']} "
                     f"and {r['pfl2_off']['seeker_wins']}/{circ['n']}.")
    if not (mass and circ):  # fall back to the held-out 90 s statistics
        three.append(f"Shorter 90 s rounds ({short3['n']} matches): seeker wins {wins(short3, 'allbrain')} against "
                     f"trained hiders, {wins(short3, 'untrained_hiders')} against untrained ones.")
    if "shuffled" in short3["results"]:
        three.append(f"Same wiring shuffled, same training: the seeker wins {wins(short3, 'shuffled')} "
                     f"and the hiders survive {surv(short3, 'shuffled'):.0f} s of 90.")
    three.append(DISCLOSURE)

    stats = {f"{a.prefix}_3h": three}
    if full5:
        stats[f"{a.prefix}_5h"] = [
            "Five hiders reuse the three-hider adapter with no retraining.",
            (f"Over {mass['5_hiders']['games']} more games on fresh seeds the seeker wins "
             f"{mass['5_hiders']['win_rate']:.0%} (95% CI {mass['5_hiders']['win_rate_ci95'][0]:.0%}–"
             f"{mass['5_hiders']['win_rate_ci95'][1]:.0%})." if mass and "5_hiders" in mass else
             f"Held-out ({full5['n']} matches, 300 s): seeker wins {wins(full5, 'allbrain')}, "
             f"hiders survive {surv(full5, 'allbrain'):.0f} s on average."),
            *( [f"Same wiring shuffled, same training: the seeker wins {wins(short3, 'shuffled')}."]
               if "shuffled" in short3["results"] else [] ),
            DISCLOSURE,
        ]
    OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for k, lines in stats.items():
        print(k)
        for line in lines:
            print("   ", line)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
