"""
Build `navcore`: the training subgraph between FlySeek's sensory roles and motor
descending neurons (GAMEPLAN.md Phase 1).

A neuron is kept if it lies within `k_down` hops downstream of any sensory-role
neuron AND within `k_up` hops upstream of any motor-DN neuron (on the pruned5
graph), plus all role neurons themselves. Edges are the pruned5 edges among kept
neurons.

Plain hop counts don't work on this small-world graph (1 hop upstream of the DNs
= 7.4k neurons, 2 hops = 73k). So a hop only follows STRONG edges: an edge counts
for the upstream expansion if it supplies >= `min_frac` of the postsynaptic
neuron's total input synapses, and for the downstream expansion if it is >=
`min_frac` of the presynaptic neuron's total output synapses. Neuron indices are REMAPPED to 0..M-1; the mapping to full-graph `idx`
is saved alongside, so roles can be looked up.

Run with --sizes to print subgraph sizes for several k before choosing.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp

from flyseek.brain.roles import role_idx
from flyseek.paths import CACHE_DIR

SOURCE_ROLES = [
    "photoreceptor_achromatic", "photoreceptor_color", "target_motion_detector", "looming_expansion",
    "looming_size", "aversive_odor", "attractive_odor", "arousal_gain",
]
SINK_ROLES = [
    "steering_high_gain", "steering_low_gain", "steering_secondary", "pursuit_descending",
    "giant_fiber", "forward_drive", "backward_drive",
]
KEEP_ROLES = SOURCE_ROLES + SINK_ROLES + ["pursuit_relay", "heading_compass", "goal_direction", "goal_steering"]


def hops_mask(adj: sp.csr_matrix, seeds: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of nodes reachable within k hops along adj (row -> col)."""
    n = adj.shape[0]
    reached = np.zeros(n, dtype=bool)
    reached[seeds] = True
    frontier = reached.copy()
    for _ in range(k):
        nxt = adj.T @ frontier.astype(np.float32) > 0  # nodes with an in-edge from frontier
        nxt &= ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt
    return reached


def load_adj(tag: str):
    import pyarrow.parquet as pq

    ei = np.load(CACHE_DIR / f"edge_index_{tag}.npy")
    syn = np.abs(np.load(CACHE_DIR / f"edge_weight_{tag}.npy"))  # proportional to synapse count
    n = pq.ParquetFile(CACHE_DIR / "neurons.parquet").metadata.num_rows
    total_out = np.bincount(ei[0], weights=syn, minlength=n)
    total_in = np.bincount(ei[1], weights=syn, minlength=n)
    frac_out = syn / np.maximum(total_out[ei[0]], 1e-9)
    frac_in = syn / np.maximum(total_in[ei[1]], 1e-9)
    return ei, (frac_out, frac_in), n


def _adj(ei, mask, n):
    return sp.csr_matrix((np.ones(int(mask.sum()), dtype=np.float32), (ei[0][mask], ei[1][mask])), shape=(n, n))


# navcore-v2 (Phase 4): the compass circuit. EPG (heading) and FC2 (goal) become
# sources, and the pathway they use to reach steering DNs is force-kept: Delta7 (the
# main EPG->PFL3 route), PFL3, and PFL3's strongest postsynaptic partner types from
# docs/phase4_cx_map.json (their per-edge input share is too small to pass min_frac).
CX_SOURCE_ROLES = ["heading_compass", "goal_direction"]
CX_KEEP_TYPES = ["EPG", "FC2A", "FC2B", "FC2C", "Delta7", "PFL3", "LAL121", "AOTU042", "AOTU019", "VES054",
                 "LAL126", "LAL083", "LAL040", "LAL141", "CRE041", "LAL076", "DNb01", "VES005"]


def select(ei, fracs, n, k_down: int, k_up: int, min_frac: float,
           source_roles: list[str] | None = None, keep_types: list[str] | None = None) -> np.ndarray:
    frac_out, frac_in = fracs
    sources = SOURCE_ROLES + (source_roles or [])
    src = np.array(sorted({i for r in sources for i in role_idx(r)}))
    snk = np.array(sorted({i for r in SINK_ROLES for i in role_idx(r)}))
    down = hops_mask(_adj(ei, frac_out >= min_frac, n), src, k_down)
    up = hops_mask(_adj(ei, frac_in >= min_frac, n).T.tocsr(), snk, k_up)
    keep = down & up
    keep[[i for r in KEEP_ROLES for i in role_idx(r)]] = True
    if keep_types:
        from flyseek.brain.roles import type_idx

        keep[[i for t in keep_types for i in type_idx(t)]] = True
    return keep


def build(tag: str, k_down: int, k_up: int, min_frac: float, out_tag: str = "navcore",
          source_roles: list[str] | None = None, keep_types: list[str] | None = None) -> dict:
    ei, fracs, n = load_adj(tag)
    keep = select(ei, fracs, n, k_down, k_up, min_frac, source_roles, keep_types)
    new_of_old = -np.ones(n, dtype=np.int64)
    kept = np.where(keep)[0]
    new_of_old[kept] = np.arange(len(kept))

    ew = np.load(CACHE_DIR / f"edge_weight_{tag}.npy")
    m = keep[ei[0]] & keep[ei[1]]
    sub_ei = np.stack([new_of_old[ei[0][m]], new_of_old[ei[1][m]]]).astype(np.int32)
    np.save(CACHE_DIR / f"edge_index_{out_tag}.npy", sub_ei)
    np.save(CACHE_DIR / f"edge_weight_{out_tag}.npy", ew[m])
    np.save(CACHE_DIR / f"{out_tag}_full_idx.npy", kept.astype(np.int32))
    meta = {"tag": out_tag, "source_graph": tag, "k_down": k_down, "k_up": k_up, "min_frac": min_frac,
            "extra_source_roles": source_roles or [], "keep_types": keep_types or [],
            "n_neurons": int(len(kept)), "n_edges": int(m.sum())}
    (CACHE_DIR / f"{out_tag}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pruned5")
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--k-down", type=int, default=6)
    ap.add_argument("--k-up", type=int, default=4)
    ap.add_argument("--min-frac", type=float, default=0.02)
    args = ap.parse_args()

    if args.sizes:
        ei, fracs, n = load_adj(args.tag)
        for f in (0.01, 0.02, 0.05):
            for kd, ku in ((4, 2), (6, 3), (6, 4), (8, 4), (8, 6)):
                keep = select(ei, fracs, n, kd, ku, f)
                m = keep[ei[0]] & keep[ei[1]]
                print(f"min_frac={f} k_down={kd} k_up={ku}: neurons {int(keep.sum()):7d}  edges {int(m.sum()):9d}", flush=True)
    else:
        print(build(args.tag, args.k_down, args.k_up, args.min_frac))
