"""
Export a recorded match (RESULTS_DIR/replays/<name>.npz/.json) into browser files under
viewer/public/replays/<name>/ and refresh viewer/public/replays/index.json.

Files:
  meta.json     roles, timing, map info, vents, events, disclosures
  states.bin    Float32 [n_ticks, n_agents, n_fields]
  walkable.bin  Uint8 [H * W] walkable mask (row 0 = lowest y)
  spike_idx.bin Uint32 neuron indices in FULL-brain layout space (so panels use soma_xy.bin)
  spike_cnt.bin Uint8 spike counts
  spike_off.bin Uint32 [n_ticks * n_agents + 1] block offsets
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from flyseek.brain.roles import full_idx_of
from flyseek.paths import RESULTS_DIR, VIEWER_DATA_DIR
from flyseek.world.grid import OccupancyGrid

REPLAY_OUT = VIEWER_DATA_DIR.parent / "replays"


def export(name: str) -> dict:
    src = RESULTS_DIR / "replays" / name
    data = np.load(src.with_suffix(".npz"))
    meta = json.loads(src.with_suffix(".json").read_text())
    out = REPLAY_OUT / name
    out.mkdir(parents=True, exist_ok=True)

    states = data["states"].astype(np.float32)
    states.tofile(out / "states.bin")

    idx = data["spike_idx"].astype(np.int64)
    full = full_idx_of(meta.get("graph")) if meta.get("brain_agents") else None
    if full is not None and len(idx):
        idx = full[idx]
    idx.astype(np.uint32).tofile(out / "spike_idx.bin")
    data["spike_cnt"].astype(np.uint8).tofile(out / "spike_cnt.bin")
    data["offsets"].astype(np.uint32).tofile(out / "spike_off.bin")

    grid = OccupancyGrid.skeld()
    grid.walkable.astype(np.uint8).tofile(out / "walkable.bin")

    n_sim = len(full) if full is not None else (165122 if meta.get("brain_agents") else 0)
    view_meta = {
        "name": name,
        "roles": meta["roles"],
        "brain_agents": meta["brain_agents"],
        "graph": meta["graph"],
        "neurons_simulated": int(n_sim),
        "tick_ms": meta["tick_ms"],
        "n_ticks": int(states.shape[0]),
        "n_agents": int(states.shape[1]),
        "state_fields": meta["state_fields"],
        "map": {"x0": grid.x0, "y0": grid.y0, "res": grid.res, "height": int(grid.walkable.shape[0]),
                "width": int(grid.walkable.shape[1])},
        "vents": meta["vents"],
        "events": meta["events"],
        "timers": meta["game_config"]["timers"],
        "vision_range": meta["game_config"]["vision"],
        "danger_range": meta["game_config"]["danger_meter"]["max_range_units"],
        "engineered": [e for e in meta["engineered"] if e],
        "odor_channels_enabled": meta.get("odor_channels_enabled"),
        "disclaimer": meta["disclaimer"],
        "seed": meta["seed"],
        "preset": meta["preset"],
    }
    (out / "meta.json").write_text(json.dumps(view_meta))

    index_path = REPLAY_OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [e for e in index if e["name"] != name]
    winner = next((e.get("winner") for e in meta["events"] if e["kind"] == "over"), None)
    index.insert(0, {"name": name, "brain_agents": meta["brain_agents"], "graph": meta["graph"],
                     "seconds": round(states.shape[0] * meta["tick_ms"] / 1000, 1), "winner": winner})
    index_path.write_text(json.dumps(index, indent=2))
    total = sum(f.stat().st_size for f in out.iterdir())
    return {"name": name, "dir": str(out), "bytes": total, "ticks": int(states.shape[0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    for nm in args.names:
        print(export(nm))
