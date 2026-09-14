"""
Exploration episodes on The Skeld for training and evaluation (Phase 4).

Each batch column is an independent fly (flies don't see each other here). Flies start
near the Cafeteria spawn with random headings. Every tick the exploration goal policy
picks a world goal direction, which enters the connectome as an FC2 bump alongside the
EPG compass bump; steering comes from the connectome via the motor decoder.

Fitness per fly = rooms visited + 10 * fraction of 1-unit map bins visited.
The wall-bump reflex is OFF by default here (training should not lean on it).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from flyseek.agents.fly_agent import FlyPopulation
from flyseek.agents.goal_policy import ExplorationGoalPolicy
from flyseek.agents.route_policy import RouteGoalPolicy
from flyseek.world.pathing import GridPaths
from flyseek.brain.roles import type_idx
from flyseek.motors.decoders import load_motor_config
from flyseek.world.grid import OccupancyGrid
from flyseek.world.match import Coverage, room_grid, spawn_positions
from flyseek.world.rules import load_extras
from flyseek.train.adapter import apply_decoder, policy_values

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
                return_traces: bool = False, policy_kind: str = "route") -> dict:
    """
    values: decoded adapter values, each scalar or an array of length len(seeds) (one fly per seed).
    Returns fitness per fly plus coverage details.
    """
    grid, rooms = _world()
    extras = load_extras()
    n = len(seeds)
    xs, ys, hs = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        p = spawn_positions(grid, 1, extras["spawn"]["xy"], rng, spawn, rooms)[0]
        xs[i], ys[i], hs[i] = p[0], p[1], rng.uniform(-np.pi, np.pi)

    pop = FlyPopulation(tag, xs, ys, hs, grid, seed=int(seeds[0]), brain=brain,
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": 1.0, "goal": 1.0})
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
    trace = [] if return_traces else None
    rng = np.random.default_rng(seeds[0] + 7919)
    t0 = time.perf_counter()
    for _ in range(int(seconds / dt)):
        b = pop.body
        goal = policy.step(b.x, b.y, b.heading, dt)
        pop.tick([], goal_angle=goal)
        if wall_bump:
            bump = pop.body.wall_contact
            pop.body.heading = np.where(bump, pop.body.heading + rng.uniform(1.0, 2.6, n) * rng.choice([-1, 1], n),
                                        pop.body.heading)
        stuck += pop.body.wall_contact * dt
        cov.update(pop.body.x, pop.body.y, np.ones(n, bool))
        if trace is not None:
            trace.append(np.stack([pop.body.x, pop.body.y, pop.body.heading, goal]))
    wall = time.perf_counter() - t0
    s = cov.summary()
    rooms_n = np.array(s["n_rooms_visited"], dtype=float)
    covf = np.array(s["coverage_frac"], dtype=float)
    out = {"fitness": rooms_n + 10.0 * covf, "rooms": rooms_n, "coverage": covf, "stuck_s": stuck,
           "wall_s": wall, "rooms_visited": s["rooms_visited"]}
    if trace is not None:
        out["trace"] = np.stack(trace)  # [T, 4, A]
    del pop
    torch.cuda.empty_cache()
    return out
