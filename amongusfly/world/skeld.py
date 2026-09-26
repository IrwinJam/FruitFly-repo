"""
Build The Skeld's occupancy grid, room labels and vents from skeld_map.json.

- The file samples walkable positions, so the floor is dilated by one cell to recover corridor
  width; a guard asserts that no two rooms merge through a wall.
- Two tiny disconnected islands (rasterisation noise) are dropped.
- Room names are normalised (e.g. "UpperEngine" / "Upper Engine", "LifeSupp" -> "O2"), and
  "Unknown" cells take the nearest labelled room.
- Vents use the real in-game coordinates and links of all 14 vents (6 groups) from SkeldJS's
  generated map data (MIT, https://github.com/SkeldJS/SkeldJS), each snapped to the nearest cell
  with at least MIN_VENT_CLEARANCE of wall clearance. Snap distances go to config/skeld_extras.json.
- Tasks are not modelled (Hide n Seek disables them; config/game.yaml).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from amongusfly.paths import CACHE_DIR, CONFIG_DIR, REPO_ROOT

MAP_PATH = REPO_ROOT / "skeld_map.json"
EXTRAS_OUT = CONFIG_DIR / "skeld_extras.json"

ROOM_ALIASES = {
    "UpperEngine": "Upper Engine",
    "LowerEngine": "Lower Engine",
    "Nav": "Navigation",
    "LifeSupp": "O2",
}

# Real Skeld vent positions and links (source in the module docstring): game name,
# x, y, and the indices of the vents it connects to.
SKELD_VENTS: list[tuple[str, float, float, tuple[int, ...]]] = [
    ("AdminVent", 2.544, -9.955201, (2, 1)),
    ("BigYVent", 9.384, -6.438, (0, 2)),
    ("CafeVent", 4.2588, -0.27600002, (0, 1)),
    ("ElecVent", -9.7764, -8.034, (5, 6)),
    ("LEngineVent", -15.288, 2.52, (11,)),
    ("SecurityVent", -12.534, -6.949, (3, 6)),
    ("MedVent", -10.608, -4.176, (3, 5)),
    ("WeaponsVent", 8.82, 3.324, (12,)),
    ("ReactorVent", -20.796, -6.953, (9,)),
    ("REngineVent", -15.2508, -13.656, (8,)),
    ("ShieldsVent", 9.5232, -14.338, (13,)),
    ("UpperReactorVent", -21.876, -3.052, (4,)),
    ("NavVentNorth", 16.008, -3.168, (7,)),
    ("NavVentSouth", 16.008, -6.384, (10,)),
]

# A fly has body radius 0.25 (config/game.yaml) and can only move into cells whose
# distance to the nearest wall is at least that, so a vent must sit somewhere it can
# stand and leave from. 0.30 matches the route planner's clearance (pathing.BODY_CLEARANCE),
# so every vent is also a cell the planner will route to, and it keeps 13 of the 14 vents
# on their exact published coordinate instead of snapping them away from it.
MIN_VENT_CLEARANCE = 0.30


def load_points() -> pd.DataFrame:
    raw = json.loads(MAP_PATH.read_text())
    df = pd.DataFrame(raw["points"])
    df["room"] = df["room"].replace(ROOM_ALIASES)
    return df, raw["grid_res"]


def drop_islands(df: pd.DataFrame, grid_res: float) -> pd.DataFrame:
    """8-connected component analysis on snapped grid cells; keep only the largest."""
    ix = np.round(df["x"] / grid_res).astype(int)
    iy = np.round(df["y"] / grid_res).astype(int)
    cells = set(zip(ix, iy))

    seen = set()
    comps = []
    for c in cells:
        if c in seen:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (cur[0] + dx, cur[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    keep = set(comps[0])

    mask = np.array([(x, y) in keep for x, y in zip(ix, iy)])
    dropped = len(df) - mask.sum()
    if dropped:
        print(f"Dropped {dropped} cells in {len(comps)-1} disconnected island(s).")
    return df[mask].reset_index(drop=True)


def fill_unknown(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["room"] != "Unknown"]
    unknown = df[df["room"] == "Unknown"]
    if len(unknown) == 0:
        return df

    known_xy = known[["x", "y"]].to_numpy()
    for idx, row in unknown.iterrows():
        d2 = (known_xy[:, 0] - row["x"]) ** 2 + (known_xy[:, 1] - row["y"]) ** 2
        nearest = known.iloc[d2.argmin()]
        df.loc[idx, "room"] = nearest["room"]
    print(f"Relabeled {len(unknown)} 'Unknown' cells to their nearest labeled neighbor.")
    return df


def _room_adjacencies(mask: np.ndarray, rooms: np.ndarray) -> set:
    """Pairs of different rooms whose cells touch (8-connected), i.e. with no wall between them."""
    lab = rooms.copy()
    lab[~mask] = ""
    pairs = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lab[1:-1, 1:-1]
        b = np.roll(np.roll(lab, -dy, axis=0), -dx, axis=1)[1:-1, 1:-1]
        m = (a != "") & (b != "") & (a != b)
        pairs.update(tuple(sorted((str(u), str(v)))) for u, v in zip(a[m], b[m]))
    return pairs


def build_grid(df: pd.DataFrame, grid_res: float) -> dict:
    ix = np.round(df["x"] / grid_res).astype(int)
    iy = np.round(df["y"] / grid_res).astype(int)
    x_min, x_max = ix.min(), ix.max()
    y_min, y_max = iy.min(), iy.max()
    w, h = x_max - x_min + 1, y_max - y_min + 1

    walkable = np.zeros((h + 2, w + 2), dtype=bool)  # 1 cell of slack for the reconstruction below
    room_grid = np.full((h + 2, w + 2), "", dtype=object)
    walkable[iy - y_min + 1, ix - x_min + 1] = True
    room_grid[iy - y_min + 1, ix - x_min + 1] = df["room"].to_numpy()
    x_min, y_min, h, w = x_min - 1, y_min - 1, h + 2, w + 2

    # Reconstruct the floor: each sampled point stands for a cell of floor, so the walkable area
    # extends about half a cell past the outermost samples. Dilating by one cell restores corridor
    # width; the guard below ensures it never connects two rooms through a wall.
    grown = ndimage.binary_dilation(walkable, structure=np.ones((3, 3), dtype=bool))
    nearest = ndimage.distance_transform_edt(~walkable, return_distances=False, return_indices=True)
    grown_rooms = room_grid[tuple(nearest)]
    grown_rooms[~grown] = ""
    before, after = _room_adjacencies(walkable, room_grid), _room_adjacencies(grown, grown_rooms)
    assert not (after - before), f"floor reconstruction merged rooms through a wall: {sorted(after - before)}"
    walkable, room_grid = grown, grown_rooms

    dist = ndimage.distance_transform_edt(walkable) * grid_res  # distance-to-wall, in world units

    return {
        "grid_res": grid_res,
        "x_min": int(x_min), "y_min": int(y_min),
        "width": int(w), "height": int(h),
        "walkable": walkable,
        "room_grid": room_grid,
        "dist_to_wall": dist,
    }


def room_centroid(df: pd.DataFrame, room: str) -> np.ndarray:
    sub = df[df["room"] == room]
    return sub[["x", "y"]].to_numpy().mean(axis=0)


def nearest_point_to(df: pd.DataFrame, room: str, target_xy: np.ndarray) -> np.ndarray:
    sub = df[df["room"] == room][["x", "y"]].to_numpy()
    d2 = ((sub - target_xy) ** 2).sum(axis=1)
    return sub[d2.argmin()]


def vent_groups() -> list[list[int]]:
    """Connected components of the real vent link graph (6 groups, 14 openings)."""
    adj = {i: set() for i in range(len(SKELD_VENTS))}
    for i, (_, _, _, links) in enumerate(SKELD_VENTS):
        for j in links:
            adj[i].add(j)
            adj[j].add(i)
    seen, groups = set(), []
    for i in range(len(SKELD_VENTS)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack.extend(adj[k])
        groups.append(sorted(comp))
    return groups


def build_vents(grid: dict, grid_res: float) -> list[dict]:
    """Real vent coordinates snapped to the nearest standable walkable node."""
    walkable, rooms, dist = grid["walkable"], grid["room_grid"], grid["dist_to_wall"]
    free = np.argwhere(walkable & (dist >= MIN_VENT_CLEARANCE))
    fx = (grid["x_min"] + free[:, 1]) * grid_res
    fy = (grid["y_min"] + free[:, 0]) * grid_res
    groups = vent_groups()
    group_of = {i: gi for gi, comp in enumerate(groups) for i in comp}
    vents = []
    for i, (name, gx, gy, links) in enumerate(SKELD_VENTS):
        k = int(np.argmin((fx - gx) ** 2 + (fy - gy) ** 2))
        row, col = free[k]
        vents.append({
            "group": group_of[i],
            "index": i,
            "game_name": name,
            "room": str(rooms[row, col]),
            "links_to": [SKELD_VENTS[j][0] for j in links],
            "x": float(fx[k]),
            "y": float(fy[k]),
            "game_x": gx,
            "game_y": gy,
            "snap_distance": round(float(np.hypot(fx[k] - gx, fy[k] - gy)), 3),
            "clearance": round(float(dist[row, col]), 3),
            "approximate": False,
        })
    return vents


def main():
    df, grid_res = load_points()
    print(f"Loaded {len(df)} raw points, grid_res={grid_res}")
    print("Room counts before cleaning:", df["room"].value_counts().to_dict())

    df = drop_islands(df, grid_res)
    df = fill_unknown(df)
    print("Room counts after cleaning:", df["room"].value_counts().to_dict())

    grid = build_grid(df, grid_res)
    vents = build_vents(grid, grid_res)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_DIR / "skeld_grid.npz",
        walkable=grid["walkable"],
        room_grid=grid["room_grid"].astype(str),
        dist_to_wall=grid["dist_to_wall"],
        x_min=grid["x_min"], y_min=grid["y_min"], grid_res=grid_res,
    )
    print(f"Saved occupancy grid ({grid['width']}x{grid['height']}) to {CACHE_DIR / 'skeld_grid.npz'}")

    extras = {
        "source": "walkable grid derived from skeld_map.json by amongusfly/world/skeld.py; vent coordinates from SkeldJS generated map data (MIT), snapped to the nearest standable node",
        "vents": vents,
        "spawn": {
            "note": "Standard Among Us behavior: all players (including the Seeker) spawn in Cafeteria.",
            "room": "Cafeteria",
            "xy": room_centroid(df, "Cafeteria").tolist(),
        },
        "tasks_modeled": False,
        "tasks_note": (
            "Hide n Seek's task rules are ambiguous across sources (Innersloth's own "
            "page describes tasks reducing the Final Hide timer via a 'Common Tasks' "
            "setting; other summaries state tasks are disabled entirely). Not resolved "
            "by further searching, so task stations are not modeled (config/game.yaml)."
        ),
    }
    EXTRAS_OUT.write_text(json.dumps(extras, indent=2))
    print(f"Saved {len(vents)} vents ({len(vent_groups())} groups, max snap "
          f"{max(v['snap_distance'] for v in vents):.2f} units) to {EXTRAS_OUT}")


if __name__ == "__main__":
    main()
