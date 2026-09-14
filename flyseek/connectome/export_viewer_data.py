"""
Export the layout + group data the browser viewer needs, as small binary/JSON files
under viewer/public/data/. Run after `layout.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flyseek.connectome.layout import DISPLAY_GROUPS, build_layout

CACHE_DIR = Path(r"C:\flyseek-data\cache")
OUT_DIR = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\viewer\public\data")


def export():
    n = pd.read_parquet(CACHE_DIR / "neurons.parquet")
    soma_xy = np.load(CACHE_DIR / "soma_xy.npy")  # [N,2] float32
    groups = json.loads((CACHE_DIR / "groups.json").read_text())

    group_names = list(DISPLAY_GROUPS.keys())
    name_to_id = {g: i for i, g in enumerate(group_names)}
    group_ids = np.array([name_to_id[g] for g in groups["idx_to_group"]], dtype=np.uint8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    soma_xy.astype(np.float32).tofile(OUT_DIR / "soma_xy.bin")
    group_ids.tofile(OUT_DIR / "group_ids.bin")

    meta = {
        "n_neurons": int(len(n)),
        "n_edges_full": None,
        "group_names": group_names,
        "group_colors": [DISPLAY_GROUPS[g]["color"] for g in group_names],
        "group_counts": {g: int((group_ids == i).sum()) for g, i in name_to_id.items()},
    }
    stats_path = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\docs\bench_results.json")
    build_stats_path = CACHE_DIR / "build_stats.json"
    if build_stats_path.exists():
        stats = json.loads(build_stats_path.read_text())
        full = next((s for s in stats if s["tag"] == "full"), None)
        if full:
            meta["n_edges_full"] = full["n_edges"]

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {len(n):,} neurons to {OUT_DIR}")
    print(f"  soma_xy.bin: {(OUT_DIR/'soma_xy.bin').stat().st_size/1024:.1f} KB")
    print(f"  group_ids.bin: {(OUT_DIR/'group_ids.bin').stat().st_size/1024:.1f} KB")
    print(f"  meta: {meta}")


if __name__ == "__main__":
    export()
