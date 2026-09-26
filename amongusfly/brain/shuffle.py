"""
Shuffled-connectome control.

Keeps the `pre` column and the weights exactly as they are, and randomly permutes
the `post` column across all edges. This preserves:
  - every neuron's out-degree, total outgoing synapse count, and sign (its NT),
  - every neuron's in-degree,
  - the global weight distribution,
while destroying WHO connects to WHOM. Not preserved: each neuron's total incoming
weight and its E/I input mix. A permutation can create a few self-loops and
duplicate pairs; duplicates are summed when the sparse matrix is built. Counts
are reported.

If a behavior shows up just as strongly in shuffled graphs, the specific wiring
isn't what produces it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from amongusfly.paths import CACHE_DIR


def shuffle_edges(edge_index: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    pre, post = edge_index[0], edge_index[1].copy()
    rng.shuffle(post)
    shuffled = np.stack([pre, post])

    n = int(max(pre.max(), post.max())) + 1
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    stats = {
        "seed": seed,
        "n_edges": int(len(pre)),
        "self_loops": int((pre == post).sum()),
        "duplicate_pairs": int(len(key) - len(np.unique(key))),
    }
    return shuffled, stats


def build(tag: str, seed: int) -> dict:
    edge_index = np.load(CACHE_DIR / f"edge_index_{tag}.npy")
    edge_weight = np.load(CACHE_DIR / f"edge_weight_{tag}.npy")
    shuffled, stats = shuffle_edges(edge_index, seed)

    out_tag = f"{tag}_shuf{seed}"
    np.save(CACHE_DIR / f"edge_index_{out_tag}.npy", shuffled)
    np.save(CACHE_DIR / f"edge_weight_{out_tag}.npy", edge_weight)

    # sanity: degree sequences identical
    n = int(edge_index.max()) + 1
    assert np.array_equal(np.bincount(edge_index[0], minlength=n), np.bincount(shuffled[0], minlength=n))
    assert np.array_equal(np.bincount(edge_index[1], minlength=n), np.bincount(shuffled[1], minlength=n))

    stats["tag"] = out_tag
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pruned5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    all_stats = [build(args.tag, s) for s in args.seeds]
    for s in all_stats:
        print(json.dumps(s))
