"""
2D soma layout for the brain panel.

Screen x is soma_x (left-right) and screen y is soma_z (head to tail), which puts the brain
above the nerve cord. Neurons without a soma position are placed deterministically inside
their region's point cloud. Also defines the display groups and colours the viewer uses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from amongusfly.paths import CACHE_DIR, DOCS_DIR

# display group -> (role names from celltypes.py, colour for the viewer); one hue per circuit
DISPLAY_GROUPS = {
    "vision": {
        "roles": ["photoreceptor_achromatic", "photoreceptor_color"],
        "color": "#79b3a2",  # sage
    },
    "pursuit": {
        "roles": ["target_motion_detector", "pursuit_relay", "arousal_gain", "pursuit_descending"],
        "color": "#c9899f",  # dusty rose
    },
    "looming_escape": {
        "roles": ["looming_expansion", "looming_size", "giant_fiber"],
        "color": "#cbb173",  # sand
    },
    "steering": {
        "roles": ["steering_high_gain", "steering_low_gain", "steering_secondary"],
        "color": "#8c9fd4",  # periwinkle
    },
    "locomotion": {
        "roles": ["forward_drive", "backward_drive"],
        "color": "#7eaec6",  # slate blue
    },
    "navigation": {
        "roles": ["heading_compass", "goal_direction", "goal_steering"],
        "color": "#a595cf",  # heather
    },
    "olfaction": {
        "roles": ["aversive_odor", "attractive_odor"],
        "color": "#6aaeac",  # teal
    },
    "other": {"roles": [], "color": "#8b8f97"},  # grey base point cloud
}


def _seeded_jitter(body_id: int, scale: float) -> tuple[float, float]:
    h = hashlib.sha256(str(int(body_id)).encode()).digest()
    a = int.from_bytes(h[:4], "little") / 2**32 - 0.5
    b = int.from_bytes(h[4:8], "little") / 2**32 - 0.5
    return a * 2 * scale, b * 2 * scale


def region_of(superclass: str) -> str:
    if not isinstance(superclass, str):
        return "other"
    if superclass.startswith("vnc_") or superclass in ("efferent_ascending", "efferent_descending"):
        return "vnc"
    return "brain"


def _donor_pools(with_soma: pd.DataFrame, keys: list[str]) -> dict:
    """Groups with_soma rows by `keys`, returning {key_tuple: (n_donors, xy ndarray)}."""
    pools = {}
    for key_val, sub in with_soma.groupby(keys):
        pools[key_val] = sub[["soma_x", "soma_z"]].to_numpy(dtype=np.float64)
    return pools


def build_layout(jitter_scale: float = 800.0) -> pd.DataFrame:
    n = pd.read_parquet(CACHE_DIR / "neurons.parquet").copy()
    n["region"] = n["superclass"].apply(region_of)

    screen_x = n["soma_x"].to_numpy(dtype=np.float64).copy()
    screen_y = n["soma_z"].to_numpy(dtype=np.float64).copy()
    has_soma = n["has_soma"].to_numpy()

    with_soma = n[has_soma]
    pool_type_side = _donor_pools(with_soma, ["type", "somaSide"])
    pool_sc_side = _donor_pools(with_soma, ["superclass", "somaSide"])
    pool_region = _donor_pools(with_soma, ["region"])
    global_xy = with_soma[["soma_x", "soma_z"]].to_numpy(dtype=np.float64)

    missing_idx = np.where(~has_soma)[0]
    n_fallback_used = {"type_side": 0, "superclass_side": 0, "region": 0, "global": 0}

    for i in missing_idx:
        row = n.iloc[i]
        key_ts = (row["type"], row["somaSide"])
        key_sc = (row["superclass"], row["somaSide"])
        key_r = (row["region"],)

        if key_ts in pool_type_side:
            donors, tag = pool_type_side[key_ts], "type_side"
        elif key_sc in pool_sc_side:
            donors, tag = pool_sc_side[key_sc], "superclass_side"
        elif key_r in pool_region:
            donors, tag = pool_region[key_r], "region"
        else:
            donors, tag = global_xy, "global"
        n_fallback_used[tag] += 1

        # deterministic donor pick, seeded from bodyId so it's stable across runs
        h = int.from_bytes(hashlib.sha256(str(int(row["bodyId"])).encode()).digest()[8:12], "little")
        donor = donors[h % len(donors)]

        jx, jy = _seeded_jitter(row["bodyId"], jitter_scale)
        screen_x[i] = donor[0] + jx
        screen_y[i] = donor[1] + jy

    # normalize to a fixed canvas, origin at center, y flipped so brain (small soma_z) is up
    x = screen_x
    y = -screen_y  # negate: smaller soma_z (brain) -> larger y -> rendered up if using +y=up
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    n["layout_x"] = x.astype(np.float32)
    n["layout_y"] = y.astype(np.float32)

    # assign display group by role membership (precedence in dict order; "other" default)
    role_body_ids = json.loads((CACHE_DIR / "role_body_ids.json").read_text())
    body_to_group = {}
    for group, spec in DISPLAY_GROUPS.items():
        for role in spec["roles"]:
            for bid in role_body_ids.get(role, []):
                body_to_group.setdefault(bid, group)  # first match wins
    n["display_group"] = n["bodyId"].map(body_to_group).fillna("other")

    print("Fallback layout usage:", n_fallback_used)
    return n


def export_for_viewer(n: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    soma_xy = n[["layout_x", "layout_y"]].to_numpy(dtype=np.float32)
    np.save(CACHE_DIR / "soma_xy.npy", soma_xy)

    groups_payload = {
        "idx_to_group": n["display_group"].tolist(),  # index-aligned with neurons.parquet 'idx'
        "group_colors": {g: spec["color"] for g, spec in DISPLAY_GROUPS.items()},
    }
    (CACHE_DIR / "groups.json").write_text(json.dumps(groups_payload))
    print(f"Saved soma_xy.npy ({soma_xy.shape}) and groups.json to {CACHE_DIR}")


def render_preview(n: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10), facecolor="black")
    ax.set_facecolor("black")

    base = n[n["display_group"] == "other"]
    ax.scatter(base["layout_x"], base["layout_y"], s=0.6, c="#d4d4d8", alpha=0.35, linewidths=0)

    for group, spec in DISPLAY_GROUPS.items():
        if group == "other":
            continue
        sub = n[n["display_group"] == group]
        if len(sub) == 0:
            continue
        ax.scatter(sub["layout_x"], sub["layout_y"], s=10, c=spec["color"], alpha=0.9,
                   linewidths=0, label=f"{group} (n={len(sub)})")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, facecolor="black", labelcolor="white", framealpha=0.5)
    ax.set_title(f"{len(n):,} neurons \u00b7 MaleCNS v1.0 \u00b7 dorsal schematic", color="white", fontsize=10)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="black")
    print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    n = build_layout()
    export_for_viewer(n)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    render_preview(n, DOCS_DIR / "brain_panel_preview.png")
