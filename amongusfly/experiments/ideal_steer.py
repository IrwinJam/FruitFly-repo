"""
Separate route geometry from steering: run the same planner, body, seeds and speed as
locomotion_quality with idealised controllers instead of the brain (CPU only).

  ideal      omega = clip(10 x heading error, +/- max turn)
  lagged     heading error low-passed with the readout's time constant, then tanh-saturated
  dead zone  the ideal controller ignoring errors below a given angle

    python -m amongusfly.experiments.ideal_steer --run explore_v5
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from amongusfly.agents.route_policy import RouteGoalPolicy
from amongusfly.experiments.locomotion_quality import quality
from amongusfly.motors.body_kinematic import KinematicBody
from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS, decode, policy_values, to_unit
from amongusfly.train.explore_env import _paths, _world
from amongusfly.world.match import spawn_positions
from amongusfly.world.rules import load_extras

DT = 0.02


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(values: dict, seeds: list[int], seconds: float, speed: float, mode: str, tau_ms: float, max_omega: float,
        dead_deg: float = 0.0):
    grid, rooms = _world()
    extras = load_extras()
    n = len(seeds)
    xs, ys, hs = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, s in enumerate(seeds):  # identical spawns to explore_env.run_episode
        rng = np.random.default_rng(s)
        p = spawn_positions(grid, 1, extras["spawn"]["xy"], rng, "default", rooms)[0]
        xs[i], ys[i], hs[i] = p[0], p[1], rng.uniform(-np.pi, np.pi)
    body = KinematicBody(xs, ys, hs)
    policy = RouteGoalPolicy(n, grid, _paths(), rooms, policy_values(values), seed=int(seeds[0]))
    err_f = np.zeros(n)
    a = DT / (tau_ms / 1000.0 + DT)
    trace = []
    for _ in range(int(seconds / DT)):
        goal = policy.step(body.x, body.y, body.heading, DT, blocked=body.wall_contact)
        err = wrap(goal - body.heading)
        err = np.where(np.abs(err) < np.deg2rad(dead_deg), 0.0, err)  # a steering dead zone like the brain's
        if mode == "ideal":
            om = np.clip(10.0 * err, -max_omega, max_omega)
        else:
            err_f = err_f + a * (err - err_f)
            om = np.tanh(err_f / 0.5) * max_omega
        body.step(np.full(n, speed), om, DT, grid)
        trace.append(np.stack([body.x, body.y, body.heading]))
    tr = np.asarray(trace)  # [T, 3, n]
    q = quality(tr, speed)
    visited = [len({str(r) for r in set(rooms[grid._cell(tr[:, 0, i], tr[:, 1, i])])} - {""}) for i in range(n)]
    q["rooms"] = float(np.mean(visited))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="explore_v5")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dead", type=float, nargs="*", default=[10, 20, 30, 40])
    args = ap.parse_args()
    params = PARAM_SETS["route"]
    values = decode(params, to_unit(params, json.loads((RESULTS_DIR / "train" / args.run / "best.json").read_text())["values"]))
    seeds = list(range(50000, 50000 + args.n))
    tau = values["decoder:ema_tau_ms"]
    mo = values["decoder:turn.max_omega_rad_per_s"]
    out = {}
    conds = [("ideal", "ideal", 0.0, 0.0), (f"lagged {tau:.0f} ms", "lagged", tau, 0.0), ("lagged 200 ms", "lagged", 200.0, 0.0)]
    conds += [(f"ideal, dead zone {d} deg", "ideal", 0.0, float(d)) for d in args.dead]
    for label, mode, t, dz in conds:
        out[label] = run(values, seeds, args.seconds, 2.0, mode, max(t, 1e-3), mo, dz)
        q = out[label]
        print(f"{label:16s} rooms {q['rooms']:5.2f}  wall {q['wall_frac']:4.0%}  speed {q['speed_frac']:4.0%}  "
              f"spin {q['spin_s_per_fly']:4.1f} s  turn {q['turn_dps']:4.0f} deg/s", flush=True)
    if args.out:
        (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
