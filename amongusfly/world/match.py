"""
Run a Hide n Seek match on The Skeld with any mix of brain-driven and scripted flies, and
record a replay.

Agent 0 is the seeker, agents 1..H the hiders; `--brains` chooses which are brain-driven.
Everything runs in 20 ms ticks of brain time. Brain senses: the seeker sees hiders through
LC10a; hiders see a looming seeker through LC4/LPLC2. The danger meter and Final Hide pings
are passed to the role policies.

    python -m amongusfly.world.match --preset full --brains all --seed 6000 --name demo
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch

from amongusfly.agents.fly_agent import FlyPopulation
from amongusfly.agents.scripted import ScriptedHider, ScriptedSeeker, line_of_sight
from amongusfly.brain.lif_torch import load_config
from amongusfly.motors.body_kinematic import KinematicBody
from amongusfly.paths import CACHE_DIR, DOCS_DIR, RESULTS_DIR
from amongusfly.senses.vision import VisualObject
from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.pathing import GridPaths
from amongusfly.world.replay import ReplayRecorder
from amongusfly.world.rules import HideNSeekRules, load_extras, load_game_config

PING_MAX_HZ = 20.0
DANGER_MAX_HZ = 20.0
PING_DECAY_S = 0.8


def room_grid(grid: OccupancyGrid) -> np.ndarray:
    """Room name per grid cell ('' for walls), aligned with OccupancyGrid.skeld() (1-cell pad)."""
    g = np.load(CACHE_DIR / "skeld_grid.npz", allow_pickle=True)
    rooms = np.pad(g["room_grid"].astype(str), 1, constant_values="")
    assert rooms.shape == grid.walkable.shape
    return rooms


def spawn_positions(grid: OccupancyGrid, n: int, center, rng, mode: str = "default",
                    rooms: np.ndarray | None = None) -> np.ndarray:
    """
    default: everyone within 2.5 units of the Cafeteria spawn (the Among Us rule).
    spread:  each agent in a different random room (a demo/experiment preset, not an Among Us rule).
    """
    ok = grid.walkable & (grid.dist >= 0.4)
    cand = np.argwhere(ok)
    xy = np.stack([grid.x0 + cand[:, 1] * grid.res, grid.y0 + cand[:, 0] * grid.res], axis=1)
    if mode == "default":
        near = xy[np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1]) < 2.5]
        return near[rng.choice(len(near), n, replace=False)]
    if mode == "spread":
        cell_rooms = rooms[cand[:, 0], cand[:, 1]]
        names = sorted({r for r in cell_rooms if r and r != "Hallway"})
        chosen = rng.choice(names, n, replace=False)
        out = []
        for r in chosen:
            pts = xy[cell_rooms == r]
            out.append(pts[rng.integers(len(pts))])
        return np.array(out)
    raise ValueError(f"unknown spawn mode {mode!r}")


class Coverage:
    """Per-agent rooms visited and 1-unit map-bin coverage while alive."""

    def __init__(self, grid: OccupancyGrid, rooms: np.ndarray, n: int, bin_units: float = 1.0):
        self.grid, self.rooms, self.n = grid, rooms, n
        self.bin = bin_units
        walk = np.argwhere(grid.walkable)
        wx = grid.x0 + walk[:, 1] * grid.res
        wy = grid.y0 + walk[:, 0] * grid.res
        self.bx0, self.by0 = wx.min(), wy.min()
        bins = {(int((a - self.bx0) // bin_units), int((b - self.by0) // bin_units)) for a, b in zip(wx, wy)}
        self.n_bins = len(bins)
        self.all_rooms = sorted({r for r in np.unique(rooms) if r})
        self.visited_bins = [set() for _ in range(n)]
        self.visited_rooms = [set() for _ in range(n)]

    def update(self, x, y, alive):
        cy, cx = self.grid._cell(x, y)
        room = self.rooms[cy, cx]
        for a in range(self.n):
            if alive[a]:
                self.visited_bins[a].add((int((x[a] - self.bx0) // self.bin), int((y[a] - self.by0) // self.bin)))
                if room[a]:
                    self.visited_rooms[a].add(str(room[a]))

    def summary(self) -> dict:
        return {
            "n_rooms_total": len(self.all_rooms),
            "rooms_visited": [sorted(r) for r in self.visited_rooms],
            "n_rooms_visited": [len(r) for r in self.visited_rooms],
            "coverage_frac": [round(len(b) / self.n_bins, 4) for b in self.visited_bins],
        }


def run_match(n_hiders: int, brain_agents: list[int], graph: str, preset: str | None, seed: int,
              name: str, record_spikes: bool = True, max_seconds: float | None = None,
              spawn: str = "default", nav: str | None = None, seeker_adapter: str | None = None,
              hider_adapter: str | None = None) -> dict:
    """
    nav: name of an exploration training run (results/train/<run>/best.json). When set,
         brain agents get the trained motor decoder plus the route goal policy, whose world goal
         direction enters the connectome through the EPG compass / FC2 goal bumps. The wall-bump
         reflex is then off (the navigator was trained without it).
    seeker_adapter / hider_adapter: role adapters, "<kind>:<run>" with kind init | best | mean
         (see amongusfly/train/eval_role.py). When given, brain agents are driven by the same
         RoleController as the batched evaluation (amongusfly/agents/role_policy.py), so a recorded
         match behaves like the evaluated ones. Takes precedence over `nav`.
    """
    rng = np.random.default_rng(seed)
    cfg = load_game_config(preset)
    extras = load_extras()
    grid = OccupancyGrid.skeld()
    paths = GridPaths(grid)
    rooms = room_grid(grid)
    tick_ms = load_config()["sim"]["brain_ms_per_game_tick"]
    dt = tick_ms / 1000.0

    n = 1 + n_hiders
    roles = ["seeker"] + ["hider"] * n_hiders
    role_speed = np.array([cfg["speed"]["seeker_units_per_s"]] + [cfg["speed"]["hider_units_per_s"]] * n_hiders)
    vis_range = np.array([cfg["vision"]["seeker_range_units"]] + [cfg["vision"]["hider_range_units"]] * n_hiders)
    start = spawn_positions(grid, n, extras["spawn"]["xy"], rng, spawn, rooms)
    coverage = Coverage(grid, rooms, n)
    first_sighting_s = None
    start_heading = rng.uniform(-np.pi, np.pi, n)

    brain_agents = sorted(brain_agents)
    scripted_agents = [a for a in range(n) if a not in brain_agents]
    odor_ok = graph.startswith("navcore")

    pop = None
    if brain_agents:
        gain = {"target": 1.0, "loom": 1.0, "photo": 1.0, "danger": 1.0 if odor_ok else 0.0, "ping": 1.0 if odor_ok else 0.0}
        pop = FlyPopulation(graph, start[brain_agents, 0], start[brain_agents, 1], start_heading[brain_agents], grid,
                            seed=seed, channel_gain=gain)
        pop.body.radius = cfg["body_radius_units"]
    nav_policy = role_ctrl = None
    if pop is not None and (seeker_adapter or hider_adapter):
        from amongusfly.agents.role_policy import RoleController
        from amongusfly.motors.decoders import load_motor_config
        from amongusfly.train.adapter import PARAM_SETS, apply_decoder
        from amongusfly.train.eval_role import load_values
        role_vals = {r: load_values(*spec.split(":", 1), PARAM_SETS[r])
                     for r, spec in (("seeker", seeker_adapter), ("hider", hider_adapter)) if spec}
        col_roles = [roles[a] for a in sorted(brain_agents)]
        role_ctrl = RoleController(col_roles, role_vals, grid, rooms, GridPaths(grid, wall_cost=4.0), extras["vents"],
                                   extras["spawn"]["xy"], seed=seed)
        pop.channel_gain = role_ctrl.channel_gains(odor_ok=odor_ok)
        dcfg = copy.deepcopy(load_motor_config())
        apply_decoder(dcfg, role_ctrl.decoder_values())
        pop.decoder.cfg = dcfg
        cfg["reflexes"]["wall_bump_turn"] = False
    elif pop is not None and nav:
        from amongusfly.agents.route_policy import RouteGoalPolicy
        from amongusfly.motors.decoders import load_motor_config
        from amongusfly.train.adapter import apply_decoder, policy_values
        nav_values = json.loads((RESULTS_DIR / "train" / nav / "best.json").read_text())["values"]
        pop.channel_gain.update({"compass": 1.0, "goal": 1.0})
        dcfg = copy.deepcopy(load_motor_config())
        apply_decoder(dcfg, nav_values)
        pop.decoder.cfg = dcfg
        nav_policy = RouteGoalPolicy(len(brain_agents), grid, GridPaths(grid, wall_cost=4.0), rooms,
                                     policy_values(nav_values), seed=seed)
        cfg["reflexes"]["wall_bump_turn"] = False
    sbody = KinematicBody(start[scripted_agents, 0], start[scripted_agents, 1], start_heading[scripted_agents],
                          radius=cfg["body_radius_units"]) if scripted_agents else None
    policies = {}
    for a in scripted_agents:
        policies[a] = (ScriptedSeeker(paths, rng, vis_range[a]) if roles[a] == "seeker"
                       else ScriptedHider(paths, rng, vis_range[a]))

    rules = HideNSeekRules(roles, dt, cfg, extras, seed=seed)
    meta = {
        "experiment": "hide_n_seek_match", "name": name, "seed": seed, "preset": preset, "graph": graph, "spawn": spawn,
        "roles": roles, "brain_agents": brain_agents, "scripted_agents": scripted_agents, "tick_ms": tick_ms,
        "odor_channels_enabled": odor_ok and bool(brain_agents),
        "game_config": cfg, "vents": extras["vents"], "spawn": extras["spawn"],
        "map": {"x0": grid.x0, "y0": grid.y0, "res": grid.res, "shape": list(grid.walkable.shape)},
        "nav": nav, "seeker_adapter": seeker_adapter, "hider_adapter": hider_adapter,
        "engineered": ["forward speed constant", "auto-vent rule", "wall-bump turn reflex" if cfg["reflexes"]["wall_bump_turn"] else None,
                       "side-level vision encoder",
                       (f"role policies + trained adapters (seeker {seeker_adapter}, hider {hider_adapter}); goal enters via "
                        "EPG/FC2 bumps with an idealized compass; hider camping scales forward speed")
                       if role_ctrl is not None else
                       f"motor decoder + route goal policy trained in run {nav} (goal enters via EPG/FC2 bumps; idealized compass)"
                       if nav_policy is not None else "hand-set motor decoder weights (untrained)"],
        "disclaimer": "Connectome-constrained LIF model with engineered sensory/motor mappings; not a validated fly brain.",
    }
    rec = ReplayRecorder(n, meta)
    n_neurons = pop.brain.n_neurons if pop else 0
    brain_col = {a: i for i, a in enumerate(brain_agents)}

    ping_level = np.zeros(n)
    ping_bearing = np.zeros(n)
    prev_visible = np.ones(n, dtype=bool)
    heading_all = start_heading.copy()
    x, y = start[:, 0].copy(), start[:, 1].copy()
    mult = np.where(np.arange(n) == 0, 0.0, 1.0)
    danger = np.zeros(n)
    t0 = time.perf_counter()
    last_print = t0
    max_ticks = int((max_seconds or cfg["timers"]["round_length_s"]) / dt) + 2

    for tick in range(max_ticks):
        # ------------------------------------------------ rules (uses last positions)
        out = rules.step(x, y)
        for ev in out.events:
            rec.event(tick, **ev)
        for a, (tx, ty) in out.teleports.items():
            if a in brain_col:
                pop.body.x[brain_col[a]], pop.body.y[brain_col[a]] = tx, ty
            else:
                j = scripted_agents.index(a)
                sbody.x[j], sbody.y[j] = tx, ty
            x[a], y[a] = tx, ty
        mult, danger, visible = out.speed_mult, out.danger, out.visible
        if out.ping_positions:
            for a in range(n):
                if roles[a] == "seeker" and out.ping_positions:
                    p = min(out.ping_positions, key=lambda q: np.hypot(q[0] - x[a], q[1] - y[a]))
                    ping_level[a] = 1.0
                    ping_bearing[a] = (np.arctan2(p[1] - y[a], p[0] - x[a]) - heading_all[a] + np.pi) % (2 * np.pi) - np.pi
        ping_level *= np.exp(-dt / PING_DECAY_S)
        if rules.over:
            break

        # ------------------------------------------------ brain agents
        counts_all = None
        if pop is not None:
            nb = len(brain_agents)
            viewer_alive = rules.alive[np.array(brain_agents)]  # caught flies stop sensing
            objects = []
            for j in range(n):
                others = np.array([a != j for a in brain_agents]) & viewer_alive
                dist = np.hypot(x[j] - pop.body.x, y[j] - pop.body.y)
                in_range = dist <= vis_range[brain_agents]
                is_seeker_viewer = np.array([roles[a] == "seeker" for a in brain_agents])
                base_vis = others & in_range & bool(visible[j])
                jx, jy = np.full(nb, x[j]), np.full(nb, y[j])
                if roles[j] == "hider":
                    objects.append(VisualObject(jx, jy, cfg["body_radius_units"], "target", visible_to=base_vis & is_seeker_viewer))
                    objects.append(VisualObject(jx, jy, cfg["body_radius_units"], "threat", visible_to=np.zeros(nb, bool)))
                else:
                    objects.append(VisualObject(jx, jy, cfg["body_radius_units"], "target", visible_to=np.zeros(nb, bool)))
                    objects.append(VisualObject(jx, jy, cfg["body_radius_units"], "threat", visible_to=base_vis & ~is_seeker_viewer))
            ba = np.array(brain_agents)
            extra = {
                "danger": {"L": danger[ba] * DANGER_MAX_HZ, "R": danger[ba] * DANGER_MAX_HZ},
                "ping": {"L": ping_level[ba] * PING_MAX_HZ * (1 + np.sin(ping_bearing[ba])) / 2,
                         "R": ping_level[ba] * PING_MAX_HZ * (1 - np.sin(ping_bearing[ba])) / 2},
            }
            goal, sp = None, np.ones(nb)
            if role_ctrl is not None:
                seen_h, pings_c, seen_s = [], [], []
                for a in ba:
                    if roles[a] == "seeker":
                        cand = [h for h in range(n) if roles[h] == "hider" and visible[h]
                                and np.hypot(x[h] - x[a], y[h] - y[a]) <= vis_range[a]
                                and line_of_sight(grid, x[a], y[a], x[h], y[h])]
                        h = min(cand, key=lambda q: np.hypot(x[q] - x[a], y[q] - y[a])) if cand else None
                        seen_h.append(None if h is None else (float(x[h]), float(y[h])))
                        pings_c.append(out.ping_positions)
                        seen_s.append(None)
                    else:
                        s0 = int(np.flatnonzero(np.array(roles) == "seeker")[0])
                        ok = bool(visible[s0]) and np.hypot(x[s0] - x[a], y[s0] - y[a]) <= vis_range[a]                             and line_of_sight(grid, x[a], y[a], x[s0], y[s0])
                        seen_s.append((float(x[s0]), float(y[s0])) if ok else None)
                        seen_h.append(None)
                        pings_c.append(None)
                goal, sp = role_ctrl.step(pop.body.x, pop.body.y, pop.body.heading, dt,
                                          viewer_alive & (mult[ba] > 0), seen_h, pings_c, seen_s, danger[ba],
                                          blocked=pop.body.wall_contact)
            elif nav_policy is not None:
                goal = nav_policy.step(pop.body.x, pop.body.y, pop.body.heading, dt,
                                       active=viewer_alive & (mult[ba] > 0), blocked=pop.body.wall_contact)
            res, counts = pop.tick(objects, record_all_spikes=record_spikes, extra_rates=extra,
                                   base_speed=role_speed[ba] * mult[ba] * sp, movable=mult[ba] > 0,
                                   sensing=viewer_alive, goal_angle=goal)
            if cfg["reflexes"]["wall_bump_turn"]:
                bump = pop.body.wall_contact & (mult[ba] > 0)
                lo, hi = np.deg2rad(cfg["reflexes"]["wall_bump_turn_deg"])
                turn = rng.uniform(lo, hi, nb) * rng.choice([-1, 1], nb)
                pop.body.heading = np.where(bump, pop.body.heading + turn, pop.body.heading)
            if record_spikes:
                counts_all = np.zeros((n_neurons, n), dtype=np.int32)
                counts_all[:, ba] = counts
            x[ba], y[ba], heading_all[ba] = pop.body.x, pop.body.y, pop.body.heading

        # ------------------------------------------------ scripted agents
        if sbody is not None:
            sp = np.zeros(len(scripted_agents))
            om = np.zeros(len(scripted_agents))
            hiders = [a for a in range(n) if roles[a] == "hider"]
            for j, a in enumerate(scripted_agents):
                if mult[a] <= 0:
                    continue
                if roles[a] == "seeker":
                    frac, om[j] = policies[a].act(a, x, y, heading_all[a], visible, hiders, dt, ping=out.ping_positions)
                else:
                    frac, om[j] = policies[a].act(a, x, y, heading_all[a], 0, bool(visible[0]), dt)
                sp[j] = frac * role_speed[a] * mult[a]
            sbody.step(sp, om, dt, grid)
            sa = np.array(scripted_agents)
            x[sa], y[sa], heading_all[sa] = sbody.x, sbody.y, sbody.heading

        speed_all = np.zeros(n)
        if pop is not None:
            speed_all[np.array(brain_agents)] = pop.body.speed
        if sbody is not None:
            speed_all[np.array(scripted_agents)] = sbody.speed
        state = {"x": x, "y": y, "heading": heading_all, "speed": speed_all, "omega": np.zeros(n),
                 "alive": rules.alive.astype(float)}
        rec.record(state, counts_all)
        coverage.update(x, y, rules.alive)
        if first_sighting_s is None and rules.phase in ("seek", "final_hide"):
            for h in range(1, n):
                if visible[h] and np.hypot(x[h] - x[0], y[h] - y[0]) <= vis_range[0] \
                        and line_of_sight(grid, x[0], y[0], x[h], y[h]):
                    first_sighting_s = round(rules.t, 2)
                    break

        now = time.perf_counter()
        if now - last_print > 15:
            print(f"  t={rules.t:6.1f}s phase={rules.phase:10s} alive={int(rules.alive[1:].sum())}/{n_hiders} "
                  f"| {tick / (now - t0):.1f} ticks/s", flush=True)
            last_print = now

    wall = time.perf_counter() - t0
    info = rec.save(RESULTS_DIR / "replays" / name)
    summary = {"name": name, "spawn": spawn, "nav": nav, "seeker_adapter": seeker_adapter,
               "hider_adapter": hider_adapter, "brain_agents": brain_agents, "graph": graph, "seed": seed,
               "winner": rules.winner, "sim_seconds": round(rules.t, 2), "wall_seconds": round(wall, 1),
               "survivors": [int(h) for h in range(1, n) if rules.alive[h]],
               "kills": [e for e in rec.events if e["kind"] == "kill"],
               "vents": sum(1 for e in rec.events if e["kind"] == "vent_enter"),
               "pings": sum(1 for e in rec.events if e["kind"] == "ping"),
               "first_sighting_s": first_sighting_s, **coverage.summary(), **info}
    if pop is not None:
        del pop
        torch.cuda.empty_cache()
    return summary


def parse_brains(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    if spec == "none":
        return []
    if spec == "seeker":
        return [0]
    if spec == "hiders":
        return list(range(1, n))
    return [int(s) for s in spec.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiders", type=int, default=3)
    ap.add_argument("--brains", default="all", help="all | none | seeker | hiders | comma-separated agent ids")
    ap.add_argument("--graph", default="navcore")
    ap.add_argument("--preset", default="short")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-spikes", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--spawn", default="default", choices=["default", "spread"])
    ap.add_argument("--nav", default=None, help="trained exploration run for brain agents, e.g. explore_navcore")
    ap.add_argument("--seeker-adapter", default=None, help="seeker adapter, e.g. init:explore_v5")
    ap.add_argument("--hider-adapter", default=None, help="hider adapter, e.g. mean:hider_v5")
    args = ap.parse_args()
    brains = parse_brains(args.brains, 1 + args.hiders)
    name = args.name or f"match_{args.brains}_{args.graph}_{args.preset}_{args.spawn}_s{args.seed}"
    s = run_match(args.hiders, brains, args.graph, args.preset, args.seed, name, not args.no_spikes, args.max_seconds,
                  args.spawn, args.nav, args.seeker_adapter, args.hider_adapter)
    print(json.dumps(s, indent=2, default=str))
