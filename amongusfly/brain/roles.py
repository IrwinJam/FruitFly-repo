"""
Shared lookups: resolved role -> neuron indices (optionally by side), neuron table.

Indices are in the full-graph `idx` space unless `graph` names a subgraph (e.g.
"navcore", "navcore_shuf0"), in which case they're translated to that subgraph's
local indices and neurons not in the subgraph are dropped.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from amongusfly.paths import CACHE_DIR


@lru_cache(maxsize=1)
def neurons() -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / "neurons.parquet")


@lru_cache(maxsize=1)
def _role_body_ids() -> dict:
    return json.loads((CACHE_DIR / "role_body_ids.json").read_text())


def subgraph_base(graph: str | None) -> str | None:
    """'navcore_shuf1' -> 'navcore' if navcore_full_idx.npy exists, else None (full-graph space)."""
    if graph is None:
        return None
    base = re.sub(r"_shuf\d+$", "", graph)
    return base if (CACHE_DIR / f"{base}_full_idx.npy").exists() else None


@lru_cache(maxsize=8)
def full_idx_of(graph: str | None) -> np.ndarray | None:
    base = subgraph_base(graph)
    return None if base is None else np.load(CACHE_DIR / f"{base}_full_idx.npy")


def _to_graph(idx: list[int], graph: str | None) -> list[int]:
    full = full_idx_of(graph)
    if full is None:
        return idx
    local = {int(f): i for i, f in enumerate(full)}
    return sorted(local[i] for i in idx if i in local)


def role_idx(role: str, side: str | None = None, graph: str | None = None) -> list[int]:
    df = neurons()
    bids = set(_role_body_ids()[role])
    sub = df[df["bodyId"].isin(bids)]
    if side is not None:
        sub = sub[sub["somaSide"] == side]
    return _to_graph(sorted(sub["idx"].astype(int).tolist()), graph)


def type_idx(type_name: str, side: str | None = None, graph: str | None = None) -> list[int]:
    df = neurons()
    sub = df[df["type"] == type_name]
    if side is not None:
        sub = sub[sub["somaSide"] == side]
    return _to_graph(sorted(sub["idx"].astype(int).tolist()), graph)
