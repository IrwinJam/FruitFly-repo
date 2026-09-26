"""
Exploration episodes on The Skeld, for training and evaluation.

Each batch column is an independent fly starting near the Cafeteria spawn. The route planner
picks a goal direction every tick, which enters the brain as the FC2 goal bump alongside the
EPG compass bump; steering comes from the brain through the motor decoder.

Fitness per fly = rooms visited + 10 * fraction of 1-unit map bins visited
                  - SPIN_PENALTY * seconds spent spinning on the spot
                  - contact_penalty * seconds blocked by a wall
(spinning: a 2 s window with at least one full turn but under 1 unit of progress).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.agents.goal_policy import ExplorationGoalPolicy
from amongusfly.agents.route_policy import RouteGoalPolicy
from amongusfly.world.pathing import GridPaths
from amongusfly.brain.roles import type_idx
from amongusfly.motors.decoders import load_motor_config
from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.match import Coverage, room_grid, spawn_positions
from amongusfly.world.rules import load_extras
from amongusfly.train.adapter import apply_decoder, policy_values

SPIN_PENALTY = 0.1  # fitness points per second spent spinning on the spot
SPIN_WINDOW = 100  # ticks (2 s)

_GRID = None
_ROOMS = None
_PATHS = None


def _world():
    global _GRID, _ROOMS
    if _GRID is None:
        _GRID = OccupancyGrid.skeld()
        _ROOMS = room_grid(_GRID)
    return _GRID, _ROOMS


def _paths():
    global _PATHS
    if _PATHS is None:
        # wall_cost keeps planned routes near corridor centers (see pathing.GridPaths)
        _PATHS = GridPaths(_world()[0], wall_cost=4.0)
    return _PATHS


def run_episode(tag: str, values: dict, seeds: list[int], seconds: float = 45.0, spawn: str = "default",
                silence: list[str] | None = None, wall_bump: bool = False, brain=None,
                return_traces: bool = False, policy_kind: str = "route", speed: float | None = None,
                contact_penalty: float = 0.0) -> dict:
    """
    values: decoded adapter values, each scalar or an array of length len(seeds) (one fly per seed).
    Returns fitness per fly plus coverage details.
    """
    grid, rooms = _world()
    extras = load_extras()
    n = len(seeds)
    policy_kind = policy_kind.replace("_obs", "")  # route_obs = route + obstacle sense
    xs, ys, hs = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        p = spawn_positions(grid, 1, extras["spawn"]["xy"], rng, spawn, rooms)[0]
        xs[i], ys[i], hs[i] = p[0], p[1], rng.uniform(-np.pi, np.pi)

    pop = FlyPopulation(tag, xs, ys, hs, grid, seed=int(seeds[0]), brain=brain,
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": float(values.get("gain:compass", 1.0)),
                                      "goal": float(values.get("gain:goal", 1.0)),
                                      "obstacle": np.broadcast_to(np.asarray(values.get("gain:obstacle", 0.0), float),
                                                                  (n,)).copy()})
    if silence:
        pop.brain.silence([i for t in silence for i in type_idx(t, graph=tag)])
    else:
        pop.brain.keep_mask = None
    cfg = copy.deepcopy(load_motor_config())
    apply_decoder(cfg, values)
    pop.decoder.cfg = cfg
    if policy_kind == "route":
        policy = RouteGoalPolicy(n, grid, _paths(), rooms, policy_values(values), seed=int(seeds[0]))
    else:
        policy = ExplorationGoalPolicy(n, grid, policy_values(values))
    cov = Coverage(grid, rooms, n)
    dt = pop.tick_ms / 1000
    stuck = np.zeros(n)
    spin_s = np.zeros(n)
    win_turn, win_x, win_y = np.zeros(n), pop.body.x.copy(), pop.body.y.copy()
    prev_hd = pop.body.heading.copy()
    trace = [] if return_traces else None
    rng = np.random.default_rng(seeds[0] + 7919)
    t0 = time.perf_counter()
    # wall_slow: a fly in wall contact cuts forward thrust to this fraction, so it can turn away
    # instead of being pinned against the wall by its own forward speed
    wall_slow = values.get("policy:wall_slow", 1.0)
    # speed: forward speed in units/s; defaults to the motor config. Matches run at the game's
    # role speed (config/game.yaml), so training and evaluation should use that value.
    base_speed_cfg = pop.decoder.cfg["forward"]["base_speed_units_per_s"] if speed is None else speed
    for step_i in range(int(seconds / dt)):
        b = pop.body
        goal = (policy.step(b.x, b.y, b.heading, dt, blocked=b.wall_contact)
                if policy_kind == "route" else policy.step(b.x, b.y, b.heading, dt))
        bs = np.where(b.wall_contact, np.asarray(base_speed_cfg) * np.asarray(wall_slow), base_speed_cfg)
        pop.tick([], goal_angle=goal, base_speed=np.broadcast_to(bs, (n,)).astype(float))
        if wall_bump:
            bump = pop.body.wall_contact
            pop.body.heading = np.where(bump, pop.body.heading + rng.uniform(1.0, 2.6, n) * rng.choice([-1, 1], n),
                                        pop.body.heading)
        stuck += pop.body.wall_contact * dt
        win_turn += np.abs((pop.body.heading - prev_hd + np.pi) % (2 * np.pi) - np.pi)
        prev_hd = pop.body.heading.copy()
        if (step_i + 1) % SPIN_WINDOW == 0:
            moved = np.hypot(pop.body.x - win_x, pop.body.y - win_y)
            spin_s += ((win_turn > 2 * np.pi) & (moved < 1.0)) * SPIN_WINDOW * dt
            win_turn[:] = 0.0
            win_x, win_y = pop.body.x.copy(), pop.body.y.copy()
        cov.update(pop.body.x, pop.body.y, np.ones(n, bool))
        if trace is not None:
            trace.append(np.stack([pop.body.x, pop.body.y, pop.body.heading, goal]))
    wall = time.perf_counter() - t0
    s = cov.summary()
    rooms_n = np.array(s["n_rooms_visited"], dtype=float)
    covf = np.array(s["coverage_frac"], dtype=float)
    # contact_penalty: fitness points lost per second the body is blocked by a wall
    out = {"fitness": rooms_n + 10.0 * covf - SPIN_PENALTY * spin_s - contact_penalty * stuck,
           "rooms": rooms_n, "coverage": covf,
           "stuck_s": stuck, "spin_s": spin_s,
           "wall_s": wall, "rooms_visited": s["rooms_visited"]}
    if trace is not None:
        out["trace"] = np.stack(trace)  # [T, 4, A]
    del pop
    torch.cuda.empty_cache()
    return out
