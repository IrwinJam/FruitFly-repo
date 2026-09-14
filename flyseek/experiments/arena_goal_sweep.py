"""
Phase 4.3: small hand-tuning sweep of the goal-navigation decoder before training
(overshoot from latency + gain, and the goal-behind PFL2 rule). Every configuration
is reported; the best becomes the STARTING adapter for Phase 4.4 training, not a
final result. Real navcore only.
"""
from __future__ import annotations

import itertools
import json

from flyseek.experiments.arena_goal import run_condition
from flyseek.paths import DOCS_DIR


def main():
    rows = []
    grid = list(itertools.product([20, 40, 80], [50, 100], [True, False]))
    for scale, tau, behind in grid:
        ov = {"turn.scale_hz": scale, "ema_tau_ms": tau, "goal_behind.enabled": behind}
        r = run_condition("navcore", 40, 8.0, 0, decoder_overrides=ov)
        r["overrides"] = ov
        rows.append(r)
        print(f"scale {scale:>2} tau {tau:>3} goal_behind {str(behind):5} | success {r['success']:.2f} "
              f"| median t {r['time_to_goal_s_median']} | err start {r['heading_error_start_deg_mean']:.0f} "
              f"-> 1s {r['heading_error_1s_deg_mean']:.0f} | {r['wall_s']}s", flush=True)
    (DOCS_DIR / "phase4_arena_goal_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
