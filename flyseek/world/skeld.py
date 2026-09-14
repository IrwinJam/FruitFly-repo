"""
Clean skeld_map.json into a usable occupancy grid + room graph, and derive
approximate vent positions from the real map geometry.

Findings from cleaning the raw map (see docs/skeld_map_preview.png for the
pre-cleaning render):
  - 13,563 walkable cells at grid_res=0.15. 13,557 of them (99.96%) form one
    connected region (8-connectivity); the other 6 are two tiny islands (5 cells
    in Reactor, 1 in Storage) that are almost certainly rasterization noise, not
    real disconnected spaces -- dropped.
  - Room names are inconsistently cased/spaced: "UpperEngine"/"Upper Engine",
    "LowerEngine"/"Lower Engine", "Nav"/"Navigation" are the same room under two
    labels; merged. "O2"/"LifeSupp" are also merged (LifeSupp is the pre-release
    Innersloth internal name for the room shipped as "O2").
  - 38 cells are labeled "Unknown" -- relabeled to the nearest labeled cell.

Vent connectivity is the real, documented Skeld vent graph (cross-checked: 6
groups summing to exactly 14 individual vent openings, matching the commonly
cited "14 vents" total): Reactor(north)<->UpperEngine, Reactor(south)<->
LowerEngine, {Electrical,MedBay,Security}, {Admin,Cafeteria,Hallway},
Navigation(north)<->Weapons, Navigation(south)<->Shields.

Vent *positions* are NOT available in skeld_map.json (it has no vent data at
all) and there's no published pixel/coordinate list calibrated to this specific
file's coordinate system. So each vent's position is derived directly from our
own map data: for a room in a 2-room group, its vent sits at the point in that
room's cell cloud closest to the other room's centroid; for a 3-room group, at
the point closest to the centroid of the other two rooms. This is a documented
approximation, not ground truth -- see the `approximate` flag on every vent
in the exported extras file.

Hide n Seek notably **disables crewmate tasks** per Innersloth's own mode
description, though other sources describe a "Common Tasks" setting that
reduces the Final Hide timer -- this is a genuine ambiguity in what's publicly
documented (not something we could resolve further by more searching), so
task stations are NOT modeled here; `config/game.yaml` defaults
`tasks_enabled: false` and documents the ambiguity rather than guessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

MAP_PATH = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\skeld_map.json")
EXTRAS_OUT = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\config\skeld_extras.json")
CACHE_DIR = Path(r"C:\flyseek-data\cache")

ROOM_ALIASES = {
    "UpperEngine": "Upper Engine",
    "LowerEngine": "Lower Engine",
    "Nav": "Navigation",
    "LifeSupp": "O2",
}

# (roomA, roomB, ...) -- one vent opening per room listed, all mutually linked.
VENT_GROUPS: list[list[str]] = [
    ["Reactor", "Upper Engine"],
    ["Reactor", "Lower Engine"],
    ["Electrical", "MedBay", "Security"],
    ["Admin", "Cafeteria", "Hallway"],
    ["Navigation", "Weapons"],
    ["Navigation", "Shields"],
]


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


def build_grid(df: pd.DataFrame, grid_res: float) -> dict:
    ix = np.round(df["x"] / grid_res).astype(int)
    iy = np.round(df["y"] / grid_res).astype(int)
    x_min, x_max = ix.min(), ix.max()
    y_min, y_max = iy.min(), iy.max()
    w, h = x_max - x_min + 1, y_max - y_min + 1

    walkable = np.zeros((h, w), dtype=bool)
    room_grid = np.full((h, w), "", dtype=object)
    walkable[iy - y_min, ix - x_min] = True
    room_grid[iy - y_min, ix - x_min] = df["room"].to_numpy()

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


def build_vents(df: pd.DataFrame) -> list[dict]:
    vents = []
    for gid, rooms in enumerate(VENT_GROUPS):
        centroids = {r: room_centroid(df, r) for r in rooms}
        for room in rooms:
            others = [c for r, c in centroids.items() if r != room]
            target = np.mean(others, axis=0)
            pos = nearest_point_to(df, room, target)
            vents.append({
                "group": gid,
                "room": room,
                "links_to": [r for r in rooms if r != room],
                "x": float(pos[0]),
                "y": float(pos[1]),
                "approximate": True,
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
    vents = build_vents(df)

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
        "source": "derived from skeld_map.json by flyseek/world/skeld.py",
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
            "by further searching, so task stations are not modeled -- see config/game.yaml."
        ),
    }
    EXTRAS_OUT.write_text(json.dumps(extras, indent=2))
    print(f"Saved {len(vents)} vents (6 groups) to {EXTRAS_OUT}")


if __name__ == "__main__":
    main()
