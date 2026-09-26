"""
Quality checks on recorded matches, against fixed thresholds:

  inside geometry   body centre on a non-walkable cell, any alive fly          <= 0.1% of alive time
  against a wall    seeker clearance < 0.35 units, share of its active time    <= 25%
  ground speed      seeker's actual speed / commanded (x1.5 in Final Hide)     >= 75%
  spinning          >= 1 full turn in a 2 s window with < 1 unit of progress   <= 3% of alive active time
  lost in a room    largest share of a match the seeker spends in one room     < 50% in every match
                    with no hider in sight

Also reports, without thresholds: time the body actually touches a wall (exact geometry), long
contacts, bounces per minute, and speed relative to what the near-wall slowdown allows.

    python -m amongusfly.experiments.clean_gate --pattern "showcase_v4_*" --out clean_gate
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from amongusfly.paths import DOCS_DIR, RESULTS_DIR
from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.match import room_grid

WALL_CLEARANCE = 0.35
BOUNCE_DEG, BOUNCE_S = 60.0, 0.4  # a turn this sharp this fast while touching a wall is a bounce
# the near-wall slowdown (motors.body_kinematic): speed x (MIN + (1-MIN) * ramp)
CAREFUL_CLEARANCE, CAREFUL_MIN_FACTOR, BODY_RADIUS = 0.55, 0.45, 0.25
WINDOW_S = 2.0
THRESHOLDS = {"inside_geometry_frac": ("<=", 0.001), "seeker_wall_frac": ("<=", 0.25),
              "seeker_speed_frac": (">=", 0.75), "spin_frac_all": ("<=", 0.03), "spin_frac_seeker": ("<=", 0.03),
              "worst_lost_in_room_frac": ("<", 0.50)}


def exact_clearance(grid: OccupancyGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest blocked cell's square, in world units.

    grid.dist_at snaps to cell centres (quantised at the grid resolution), so near a wall it never
    reports less than one cell and a body can look clear while it is actually in contact.
    """
    h = grid.res / 2
    cy, cx = grid._cell(x, y)
    best = np.full(len(x), 9.9)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            ry, rxx = np.clip(cy + dy, 0, grid.h - 1), np.clip(cx + dx, 0, grid.w - 1)
            blocked = ~grid.walkable[ry, rxx]
            if not blocked.any():
                continue
            ddx = np.maximum(np.abs(x - (grid.x0 + rxx * grid.res)) - h, 0.0)
            ddy = np.maximum(np.abs(y - (grid.y0 + ry * grid.res)) - h, 0.0)
            best = np.where(blocked, np.minimum(best, np.hypot(ddx, ddy)), best)
    return best


def visible(grid: OccupancyGrid, a: np.ndarray, b: np.ndarray, rng: float) -> bool:
    d = float(np.hypot(*(b - a)))
    if d > rng:
        return False
    t = np.linspace(0.0, 1.0, max(2, int(d / (grid.res * 0.5))))
    return bool(grid.walkable_at(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t).all())


def match_metrics(states: np.ndarray, meta: dict, grid: OccupancyGrid, rooms: np.ndarray) -> dict:
    dt = meta["tick_ms"] / 1000.0
    ev = meta["events"]
    seek = next(e["tick"] for e in ev if e["kind"] == "phase" and e["phase"] == "seek")
    fh = next((e["tick"] for e in ev if e["kind"] == "phase" and e["phase"] == "final_hide"), None)
    gc = meta["game_config"]
    base = gc["speed"]["seeker_units_per_s"]
    fh_mult = gc["timers"]["final_hide_speed_multiplier"]
    vis_rng = gc["vision"]["seeker_range_units"] if "vision" in gc else 6.0
    x, y, hd, alive = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 5] > 0.5
    T, n = x.shape
    active = alive.copy()
    active[:seek] = False
    s = 0  # the seeker is agent 0

    inside = ~grid.walkable_at(x, y) & alive
    clear = grid.dist_at(x[:, s], y[:, s])
    exact = exact_clearance(grid, x[:, s], y[:, s])  # true geometry, for contact
    touching = exact < BODY_RADIUS + 1e-6
    act_s = active[:, s]

    v = np.hypot(np.diff(x[:, s]), np.diff(y[:, s])) / dt
    cmd = np.full(T - 1, base)
    if fh is not None:
        cmd[fh:] *= fh_mult
    m = act_s[1:]
    # what the body should manage given our own near-wall slowdown: anything below that is geometry, not the rule
    ramp = np.clip((clear[:-1] - BODY_RADIUS) / (CAREFUL_CLEARANCE - BODY_RADIUS), 0.0, 1.0)
    expect = cmd * (CAREFUL_MIN_FACTOR + (1.0 - CAREFUL_MIN_FACTOR) * ramp)
    grinding = m & (v < 0.6 * expect)

    # bounces: distinct events where the seeker turns sharply while touching a wall
    bw = int(round(BOUNCE_S / dt))
    sharp = np.abs((hd[bw:, s] - hd[:-bw, s] + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(BOUNCE_DEG)
    ev = sharp & touching[:-bw] & act_s[:-bw]
    bounces = int(np.sum(ev[1:] & ~ev[:-1]) + (1 if len(ev) and ev[0] else 0))
    # time spent in contact for more than half a second at a stretch
    run, slide_s = 0, 0.0
    for c in (touching & act_s):
        run = run + 1 if c else 0
        if run * dt >= 0.5:
            slide_s += dt
    w = int(round(WINDOW_S / dt))
    dh = np.abs((np.diff(hd, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    spin = np.zeros(n)
    for a in range(seek, T - w, w):
        turn = dh[a:a + w].sum(axis=0)
        disp = np.hypot(x[a + w] - x[a], y[a + w] - y[a])
        ok = active[a:a + w + 1].all(axis=0)
        spin += ((turn > 2 * np.pi) & (disp < 1.0) & ok) * w * dt

    # seeker lost in one room: ticks (every 0.1 s) in a room with no living hider in sight
    step = max(1, int(round(0.1 / dt)))
    cy, cx = grid._cell(x[:, s], y[:, s])
    lost: dict[str, int] = {}
    n_act = 0
    for t in range(seek, T, step):
        if not act_s[t]:
            continue
        n_act += 1
        a = np.array([x[t, s], y[t, s]])
        if any(alive[t, h] and visible(grid, a, np.array([x[t, h], y[t, h]]), vis_rng) for h in range(1, n)):
            continue
        r = str(rooms[cy[t], cx[t]]) or "(corridor)"
        lost[r] = lost.get(r, 0) + 1
    worst_room, worst_n = max(lost.items(), key=lambda kv: kv[1]) if lost else ("", 0)

    return {
        "alive_s": float(alive.sum() * dt), "active_s": float(active.sum() * dt),
        "seeker_active_s": float(act_s.sum() * dt),
        "inside_geometry_s": float(inside.sum() * dt),
        "seeker_wall_s": float(((clear < WALL_CLEARANCE) & act_s).sum() * dt),
        "seeker_contact_s": float((touching & act_s).sum() * dt),
        "seeker_slide_s": float(slide_s), "seeker_bounces": float(bounces),
        "seeker_dist_units": float((v * dt)[m].sum()), "seeker_cmd_units": float((cmd * dt)[m].sum()),
        "seeker_expected_units": float((expect * dt)[m].sum()), "seeker_grind_s": float(grinding.sum() * dt),
        "spin_s_all": float(spin.sum()), "spin_s_seeker": float(spin[s]),
        "lost_room": worst_room, "lost_in_room_frac": worst_n / max(1, n_act),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="showcase_v4_*")
    ap.add_argument("--out", default=None, help="write docs/<out>.json")
    args = ap.parse_args()
    grid = OccupancyGrid.skeld()
    rooms = room_grid(grid)
    per = {}
    for f in sorted(glob.glob(str(RESULTS_DIR / "replays" / f"{args.pattern}.npz"))):
        name = Path(f).stem
        meta = json.loads(Path(f).with_suffix(".json").read_text(encoding="utf-8"))
        per[name] = match_metrics(np.load(f)["states"], meta, grid, rooms)
    if not per:
        raise SystemExit(f"no replays matching {args.pattern}")
    tot = {k: sum(p[k] for p in per.values()) for k in next(iter(per.values())) if isinstance(next(iter(per.values()))[k], float)}
    summary = {
        "inside_geometry_frac": tot["inside_geometry_s"] / tot["alive_s"],
        "seeker_wall_frac": tot["seeker_wall_s"] / tot["seeker_active_s"],
        "seeker_speed_frac": tot["seeker_dist_units"] / tot["seeker_cmd_units"],
        "spin_frac_all": tot["spin_s_all"] / tot["active_s"],
        "spin_frac_seeker": tot["spin_s_seeker"] / tot["seeker_active_s"],
        "worst_lost_in_room_frac": max(p["lost_in_room_frac"] for p in per.values()),
    }
    reported = {"seeker_contact_frac": tot["seeker_contact_s"] / tot["seeker_active_s"],
                "seeker_slide_frac": tot["seeker_slide_s"] / tot["seeker_active_s"],
                "seeker_bounces_per_min": tot["seeker_bounces"] / (tot["seeker_active_s"] / 60),
                "seeker_speed_vs_expected": tot["seeker_dist_units"] / tot["seeker_expected_units"],
                "seeker_grinding_frac": tot["seeker_grind_s"] / tot["seeker_active_s"]}
    checks = {}
    for k, (op, thr) in THRESHOLDS.items():
        v = summary[k]
        checks[k] = {"value": round(v, 4), "threshold": f"{op} {thr}",
                     "pass": bool(v <= thr if op == "<=" else v >= thr if op == ">=" else v < thr)}
    passed = all(c["pass"] for c in checks.values())
    print(f"clean-match gate on {len(per)} replays ({args.pattern}):")
    for k, c in checks.items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {k:26s} {c['value']:8.4f}  (needs {c['threshold']})")
    print("  per match (seeker wall / speed / spin s / lost in room):")
    for name, p in per.items():
        print(f"    {name:26s} {p['seeker_wall_s'] / max(p['seeker_active_s'], 1e-9):5.0%}  "
              f"{p['seeker_dist_units'] / max(p['seeker_cmd_units'], 1e-9):5.0%}  {p['spin_s_seeker']:5.1f}  "
              f"{p['lost_in_room_frac']:4.0%} {p['lost_room']}")
    print(f"  (reported, not gated) body touching geometry {reported['seeker_contact_frac']:.1%} | "
          f"in contact > 0.5 s at a stretch {reported['seeker_slide_frac']:.1%} | "
          f"bounces {reported['seeker_bounces_per_min']:.1f}/min | "
          f"speed vs what the near-wall rule allows {reported['seeker_speed_vs_expected']:.0%} | "
          f"grinding (under 60% of that) {reported['seeker_grinding_frac']:.1%}")
    print(f"GATE {'PASSED' if passed else 'FAILED'} (numeric checks only; the visual check of every replay is separate)")
    if args.out:
        (DOCS_DIR / f"{args.out}.json").write_text(json.dumps(
            {"pattern": args.pattern, "passed_numeric": passed, "checks": checks, "summary": summary,
             "reported": reported, "per_match": per},
            indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
