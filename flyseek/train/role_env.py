"""
Batched Hide n Seek role episodes for Phase 5 training and evaluation.

M independent matches run in lock-step. Each match has 1 Seeker + H Hiders under the
real rules engine (flyseek/world/rules.py: freeze, kills, auto-vent, Final Hide pings).
The trained role ("seeker" or "hider") is played by one controller type in every match;
the other role is scripted (flyseek/agents/scripted.py).

  role="seeker": agent 0 is the trainee, H scripted hiders.   Batch column = match.
  role="hider":  agents 1..H are trainees, 1 scripted seeker. Batch column = m*H + (h-1).
  role="both":   every agent is a brain (Phase 6 showcase setting); values = {"seeker": .., "hider": ..}.
                 Batch column = m*(1+H) + agent.

Controllers for the trainee:
  "brain"    connectome + role policy + trained decoder (all brain flies share one GPU batch)
  "scripted" the scripted agent of that role (reference)
  "random"   random walk: constant speed, random turning, wall-bump turn (baseline)

Senses for brain flies mirror flyseek/world/match.py: hiders -> LC10a (seeker), seeker ->
LC4/LPLC2 loom (hiders), danger meter -> aversive odor, pings -> attractive odor.

Fitness:
  seeker: sum over hiders of caught * (1 + time left / round) / H        (0..2)
  hider:  survival fraction (death time / round, 1 if alive) + 0.5 * survived, per hider fly
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from flyseek.agents.fly_agent import FlyPopulation
from flyseek.agents.role_policy import RoleController
from flyseek.agents.scripted import ScriptedHider, ScriptedSeeker
from flyseek.brain.lif_torch import load_config
from flyseek.brain.roles import type_idx
from flyseek.motors.body_kinematic import KinematicBody
from flyseek.motors.decoders import load_motor_config
from flyseek.senses.vision import VisualObject
from flyseek.train.adapter import apply_decoder
from flyseek.world.grid import OccupancyGrid
from flyseek.world.match import DANGER_MAX_HZ, PING_DECAY_S, PING_MAX_HZ, room_grid, spawn_positions
from flyseek.world.pathing import GridPaths
from flyseek.world.rules import HideNSeekRules, load_extras, load_game_config

_W = {}


def _world():
    if not _W:
        g = OccupancyGrid.skeld()
        _W.update(grid=g, rooms=room_grid(g), paths=GridPaths(g), nav_paths=GridPaths(g, wall_cost=4.0))
    return _W


def _los_many(grid, x0, y0, x1, y1) -> np.ndarray:
    """Vectorized line of sight for pairs of points."""
    d = np.hypot(x1 - x0, y1 - y0)
    if len(d) == 0:
        return np.zeros(0, bool)
    wall = grid.raycast(x0, y0, np.arctan2(y1 - y0, x1 - x0)[:, None], float(max(d.max(), 1e-3)))[:, 0]
    return (wall >= d - grid.res) | (d < 1e-6)


def run_role_episode(tag: str, role: str, values: dict | None, seeds: list[int], controller: str = "brain",
                     preset: str = "short", n_hiders: int = 3, spawn: str = "default",
                     silence: list[str] | None = None, brain=None, record_positions: bool = False) -> dict:
    """values: decoded adapter values (scalars or arrays of length = number of brain flies)."""
    W = _world()
    grid, rooms, paths = W["grid"], W["rooms"], W["paths"]
    cfg = load_game_config(preset)
    extras = load_extras()
    tick_ms = load_config()["sim"]["brain_ms_per_game_tick"]
    dt = tick_ms / 1000.0
    M, H = len(seeds), n_hiders
    n = 1 + H
    T = cfg["timers"]["round_length_s"]
    speed = np.array([cfg["speed"]["seeker_units_per_s"]] + [cfg["speed"]["hider_units_per_s"]] * H)
    vis_range = np.array([cfg["vision"]["seeker_range_units"]] + [cfg["vision"]["hider_range_units"]] * H)
    R = cfg["body_radius_units"]

    X, Y, HD = np.zeros((M, n)), np.zeros((M, n)), np.zeros((M, n))
    rules, rngs = [], []
    for m, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        p = spawn_positions(grid, n, extras["spawn"]["xy"], rng, spawn, rooms)
        X[m], Y[m], HD[m] = p[:, 0], p[:, 1], rng.uniform(-np.pi, np.pi, n)
        rules.append(HideNSeekRules(["seeker"] + ["hider"] * H, dt, copy.deepcopy(cfg), extras, seed=s))
        rngs.append(rng)

    trainee = [0] if role == "seeker" else (list(range(1, n)) if role == "hider" else list(range(n)))
    # (m, agent) of each trainee batch column
    cols = [(m, a) for m in range(M) for a in trainee]
    A = len(cols)
    cm, ca = np.array([c[0] for c in cols]), np.array([c[1] for c in cols])
    opp = [a for a in range(n) if a not in trainee]

    # scripted opponents (and scripted/random trainees)
    scripted = [(m, a) for m in range(M) for a in opp]
    if controller == "scripted":
        scripted += cols
    sm, sa = np.array([c[0] for c in scripted], int), np.array([c[1] for c in scripted], int)
    spol = [ScriptedSeeker(paths, rngs[m], vis_range[a]) if a == 0 else ScriptedHider(paths, rngs[m], vis_range[a])
            for m, a in scripted]
    sbody = KinematicBody(X[sm, sa], Y[sm, sa], HD[sm, sa], radius=R)
    rbody = KinematicBody(X[cm, ca], Y[cm, ca], HD[cm, ca], radius=R) if controller == "random" else None
    rw_rng = np.random.default_rng(seeds[0] + 104729)

    pop = policy = None
    if controller == "brain":
        col_roles = ["seeker" if a == 0 else "hider" for a in ca]
        role_values = values if role == "both" else {role: values}
        policy = RoleController(col_roles, role_values, grid, rooms, W["nav_paths"], extras["vents"],
                                extras["spawn"]["xy"], seed=int(seeds[0]))
        pop = FlyPopulation(tag, X[cm, ca], Y[cm, ca], HD[cm, ca], grid, seed=int(seeds[0]), brain=brain,
                            channel_gain=policy.channel_gains(odor_ok=tag.startswith("navcore")))
        pop.body.radius = R
        if silence:
            pop.brain.silence([i for t in silence for i in type_idx(t, graph=tag)])
        else:
            pop.brain.keep_mask = None
        dcfg = copy.deepcopy(load_motor_config())
        apply_decoder(dcfg, policy.decoder_values())
        pop.decoder.cfg = dcfg

    death_t = np.full((M, n), np.nan)
    done = np.zeros(M, bool)
    ping_level, ping_bearing = np.zeros(M), np.zeros(M)
    ping_pos: list = [None] * M
    visible = np.ones((M, n), bool)
    mult = np.ones((M, n))
    danger = np.zeros((M, n))
    first_sight = np.full(M, np.nan)
    stuck = np.zeros(A)
    traj = [] if record_positions else None
    t0 = time.perf_counter()
    max_ticks = int(T / dt) + 2

    for tick in range(max_ticks):
        # -------------------------------------------------------------- rules
        new_pings = [None] * M
        for m in range(M):
            if done[m]:
                continue
            r = rules[m]
            out = r.step(X[m], Y[m])
            for a, (tx, ty) in out.teleports.items():
                X[m, a], Y[m, a] = tx, ty
            for ev in out.events:
                if ev["kind"] == "kill":
                    death_t[m, ev["victim"]] = r.t
            mult[m], danger[m], visible[m] = out.speed_mult, out.danger, out.visible
            if out.ping_positions:
                new_pings[m] = out.ping_positions
                p = min(out.ping_positions, key=lambda q: np.hypot(q[0] - X[m, 0], q[1] - Y[m, 0]))
                ping_level[m] = 1.0
                ping_bearing[m] = (np.arctan2(p[1] - Y[m, 0], p[0] - X[m, 0]) - HD[m, 0] + np.pi) % (2 * np.pi) - np.pi
            if r.over:
                done[m] = True
                mult[m] = 0.0
        ping_level *= np.exp(-dt / PING_DECAY_S)
        if done.all():
            break
        alive = np.array([r.alive for r in rules])
        live = ~done

        # teleports must reach the bodies
        if pop is not None:
            pop.body.x, pop.body.y = X[cm, ca].copy(), Y[cm, ca].copy()
        if rbody is not None:
            rbody.x, rbody.y = X[cm, ca].copy(), Y[cm, ca].copy()
        sbody.x, sbody.y = X[sm, sa].copy(), Y[sm, sa].copy()

        # sightings (seeker <-> hiders, range + line of sight + visible), per match
        dx, dy = X[:, 1:] - X[:, :1], Y[:, 1:] - Y[:, :1]
        in_range = (np.hypot(dx, dy) <= vis_range[0]) & visible[:, 1:] & visible[:, :1] & live[:, None]
        mm, hh = np.nonzero(in_range)
        los = np.zeros((M, H), bool)
        if len(mm):
            los[mm, hh] = _los_many(grid, X[mm, 0], Y[mm, 0], X[mm, hh + 1], Y[mm, hh + 1])
        seeking = np.array([r.phase in ("seek", "final_hide") for r in rules])
        newly = np.isnan(first_sight) & los.any(axis=1) & seeking
        first_sight[newly] = tick * dt

        # -------------------------------------------------------------- trainee: brain
        if pop is not None:
            movable = mult[cm, ca] > 0
            sensing = alive[cm, ca] & live[cm]
            sk = ca == 0  # seeker columns
            # hiders -> LC10a target channel of seeker columns; seeker -> looming channel of hider columns
            objects = [VisualObject(X[cm, k], Y[cm, k], R, "target",
                                    visible_to=sk & visible[cm, k]
                                    & (np.hypot(X[cm, k] - X[cm, ca], Y[cm, k] - Y[cm, ca]) <= vis_range[0]))
                       for k in range(1, n)]
            objects.append(VisualObject(X[cm, 0], Y[cm, 0], R, "threat",
                                        visible_to=~sk & visible[cm, 0]
                                        & (np.hypot(X[cm, 0] - X[cm, ca], Y[cm, 0] - Y[cm, ca]) <= vis_range[1])))
            pl, pb = np.where(sk, ping_level[cm], 0.0), ping_bearing[cm]
            dz = np.where(sk, 0.0, danger[cm, ca]) * DANGER_MAX_HZ
            extra = {"ping": {"L": pl * PING_MAX_HZ * (1 + np.sin(pb)) / 2, "R": pl * PING_MAX_HZ * (1 - np.sin(pb)) / 2},
                     "danger": {"L": dz, "R": dz}}
            seen_h, seen_s, pings = [], [], []
            for m, a in cols:
                if a == 0:
                    ks = np.flatnonzero(los[m])
                    if len(ks):
                        k = ks[np.argmin(np.hypot(dx[m, ks], dy[m, ks]))]
                        seen_h.append((float(X[m, k + 1]), float(Y[m, k + 1])))
                    else:
                        seen_h.append(None)
                    pings.append(new_pings[m])
                    seen_s.append(None)
                else:
                    seen_h.append(None)
                    pings.append(None)
                    seen_s.append((float(X[m, 0]), float(Y[m, 0])) if los[m, a - 1] else None)
            goal, sp = policy.step(pop.body.x, pop.body.y, pop.body.heading, dt, movable & sensing, seen_h, pings,
                                   seen_s, danger[cm, ca])
            pop.tick(objects, extra_rates=extra, base_speed=speed[ca] * mult[cm, ca] * sp, movable=movable,
                     sensing=sensing, goal_angle=goal)
            stuck += pop.body.wall_contact * dt * (movable & sensing)
            X[cm, ca], Y[cm, ca], HD[cm, ca] = pop.body.x, pop.body.y, pop.body.heading

        # -------------------------------------------------------------- trainee: random walk
        if rbody is not None:
            movable = mult[cm, ca] > 0
            om = rw_rng.normal(0, 2.0, A)
            rbody.step(speed[ca] * mult[cm, ca], np.where(movable, om, 0.0), dt, grid)
            bump = rbody.wall_contact & movable
            rbody.heading = np.where(bump, rbody.heading + rw_rng.uniform(1.0, 2.6, A) * rw_rng.choice([-1, 1], A),
                                     rbody.heading)
            stuck += bump * dt
            X[cm, ca], Y[cm, ca], HD[cm, ca] = rbody.x, rbody.y, rbody.heading

        # -------------------------------------------------------------- scripted agents
        spd, om = np.zeros(len(scripted)), np.zeros(len(scripted))
        hiders = list(range(1, n))
        for j, (m, a) in enumerate(scripted):
            if done[m] or mult[m, a] <= 0:
                continue
            if a == 0:
                frac, om[j] = spol[j].act(0, X[m], Y[m], HD[m, 0], visible[m], hiders, dt, ping=new_pings[m])
            else:
                frac, om[j] = spol[j].act(a, X[m], Y[m], HD[m, a], 0, bool(visible[m, 0]), dt)
            spd[j] = frac * speed[a] * mult[m, a]
        sbody.step(spd, om, dt, grid)
        X[sm, sa], Y[sm, sa], HD[sm, sa] = sbody.x, sbody.y, sbody.heading
        if traj is not None:
            traj.append(np.stack([X.copy(), Y.copy()]))

    wall = time.perf_counter() - t0
    t_end = np.array([r.t for r in rules])
    caught = ~np.isnan(death_t[:, 1:])
    left = np.where(caught, (T - np.nan_to_num(death_t[:, 1:], nan=T)) / T, 0.0)
    seeker_fit = (caught * (1.0 + left)).sum(axis=1) / H
    surv_t = np.where(caught, death_t[:, 1:], T)
    hider_fit = (surv_t / T + 0.5 * ~caught).reshape(-1)  # [M*H] in column order (m, h)
    per = {"kills": caught.sum(axis=1), "seeker_win": caught.all(axis=1), "survival_s": surv_t.reshape(-1),
           "survived": (~caught).reshape(-1), "fitness_seeker": seeker_fit, "fitness_hider": hider_fit}
    fitness = seeker_fit if role == "seeker" else hider_fit  # "both": hider fitness; see fitness_seeker
    out = {"fitness": fitness, "wall_s": wall, "first_sighting_s": first_sight, "match_end_s": t_end,
           "stuck_s": stuck, **per}
    if traj is not None:
        out["traj"] = np.stack(traj)
    if pop is not None:
        del pop
        torch.cuda.empty_cache()
    return out


def new_pings_for(cm, new_pings):
    return [new_pings[m] for m in cm]
