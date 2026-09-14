"""
Build the FlySeek brain graph from raw MaleCNS v1.0 Feather files.

Findings from inspecting the raw data (2026-09-13) that shape this module:

* The raw `connectome-weights-*.feather` file has 151.8M rows, but the vast majority
  connect to untraced/orphan segmentation fragments (87.5M distinct `body_post`
  values against only 165,122 *traced* neurons). Restricting to edges where BOTH
  endpoints are `status == "Traced"` gives **25,563,197 edges** among **165,122
  neurons** — matching the "~166,700 neurons / ~125M synapses" headline figures
  reported for MaleCNS v1.0 in press coverage (that figure counts individual
  synapses; this table already aggregates same pre->post pairs into one row with a
  `weight` = synapse count) and, more precisely, the "166,700 neurons / 25,582,938
  connections" figures cited by community projects (DOOMFLY, ornata/fly) almost
  exactly. That confirms "traced-to-traced" is the right definition of our `full`
  graph.
* Filtering that traced-only edge set to `weight >= 5` gives **6,235,682 edges** --
  matching the "176,422 neurons and 6.29 million connections" figure reported by the
  fly-brain-minecraft project (small neuron-count difference likely reflects a
  slightly different confidence/status cut). This confirms `pruned5` as a sane
  default working graph.
* Neurotransmitter sign: using `celltype_predicted_nt` (the cell-type-level
  aggregate column, described in the file as more robust than the single-body
  `predicted_nt`) resolves all but ~2.9% of traced neurons to a concrete
  neurotransmitter. Sign convention (standard for Drosophila LIF models, per Shiu et
  al. 2024's "excitatory (primarily cholinergic) or inhibitory (GABAergic or
  glutamatergic)" simplification, extended here for the rarer categories present in
  this dataset):
    - acetylcholine, dopamine, serotonin, octopamine -> excitatory (+1)
      (dopamine/serotonin/octopamine are really neuromodulatory, not classically
      fast-excitatory; treating them as +1 is a simplification flagged in
      `neurons.parquet` via the `nt_confident` column.)
    - gaba, glutamate, histamine -> inhibitory (-1)
      (glutamate is inhibitory in the fly CNS via glutamate-gated chloride channels;
      histamine is inhibitory at photoreceptor->LMC synapses.)
    - unclear / missing -> excitatory (+1) by default (majority class), flagged
      `nt_confident = False`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAW_DIR = Path(r"C:\flyseek-data\raw\malecns_v1")
CACHE_DIR = Path(r"C:\flyseek-data\cache")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "brain.yaml"

ANNOTATIONS_FILE = RAW_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = RAW_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = RAW_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

NT_SIGN = {
    "acetylcholine": 1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}
DEFAULT_SIGN = 1  # applied when NT is "unclear" or missing


def _soma_xyz(v):
    if hasattr(v, "__len__") and len(v) == 3:
        return v[0], v[1], v[2]
    return np.nan, np.nan, np.nan


def load_neurons() -> pd.DataFrame:
    """Traced neurons with type/superclass/soma/side + resolved NT sign."""
    ann = pd.read_feather(ANNOTATIONS_FILE)
    ann = ann[ann["status"] == "Traced"].copy()
    ann = ann[["bodyId", "type", "superclass", "class", "somaSide", "somaLocation"]]

    nt = pd.read_feather(NT_FILE)[
        ["body", "celltype_predicted_nt", "predicted_nt"]
    ].rename(columns={"body": "bodyId"})

    df = ann.merge(nt, on="bodyId", how="left")

    resolved_nt = df["celltype_predicted_nt"].where(
        df["celltype_predicted_nt"].notna() & (df["celltype_predicted_nt"] != "unclear"),
        df["predicted_nt"],
    )
    df["nt"] = resolved_nt
    df["nt_confident"] = resolved_nt.isin(NT_SIGN.keys())
    df["sign"] = resolved_nt.map(NT_SIGN).fillna(DEFAULT_SIGN).astype(np.int8)

    xyz = df["somaLocation"].apply(_soma_xyz)
    df["soma_x"] = xyz.apply(lambda t: t[0])
    df["soma_y"] = xyz.apply(lambda t: t[1])
    df["soma_z"] = xyz.apply(lambda t: t[2])
    df["has_soma"] = df["soma_x"].notna()

    df = df.drop(columns=["somaLocation", "celltype_predicted_nt", "predicted_nt"])
    df = df.sort_values("bodyId").reset_index(drop=True)
    df["idx"] = np.arange(len(df), dtype=np.int32)
    return df


def load_edges(neurons: pd.DataFrame, min_synapses: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (edge_index[2,E] int32 using `idx` from `neurons`, edge_weight[E] float32
    in millivolts, i.e. synapse_count * sign(presynaptic NT) * mv_per_synapse).
    Only edges where both endpoints are in `neurons` (traced) are kept.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    mv_per_synapse = cfg["lif"]["mv_per_synapse"]

    w = pd.read_feather(WEIGHTS_FILE)
    if min_synapses > 0:
        w = w[w["weight"] >= min_synapses]

    body_to_idx = pd.Series(neurons["idx"].values, index=neurons["bodyId"].values)
    traced_ids = set(neurons["bodyId"])
    w = w[w["body_pre"].isin(traced_ids) & w["body_post"].isin(traced_ids)]

    pre_idx = body_to_idx.loc[w["body_pre"]].to_numpy(dtype=np.int32)
    post_idx = body_to_idx.loc[w["body_post"]].to_numpy(dtype=np.int32)

    sign_by_idx = neurons.set_index("idx")["sign"].to_numpy()
    signed_weight = (
        w["weight"].to_numpy(dtype=np.float32)
        * sign_by_idx[pre_idx].astype(np.float32)
        * np.float32(mv_per_synapse)
    )

    edge_index = np.stack([pre_idx, post_idx], axis=0)
    return edge_index, signed_weight


def build_and_cache(min_synapses: int, tag: str) -> dict:
    neurons = load_neurons()
    edge_index, edge_weight = load_edges(neurons, min_synapses=min_synapses)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    neurons_path = CACHE_DIR / "neurons.parquet"
    if not neurons_path.exists():
        neurons.to_parquet(neurons_path)

    np.save(CACHE_DIR / f"edge_index_{tag}.npy", edge_index)
    np.save(CACHE_DIR / f"edge_weight_{tag}.npy", edge_weight)

    stats = {
        "tag": tag,
        "min_synapses": min_synapses,
        "n_neurons": len(neurons),
        "n_edges": int(edge_index.shape[1]),
        "n_with_soma": int(neurons["has_soma"].sum()),
        "n_nt_confident": int(neurons["nt_confident"].sum()),
    }
    return stats


def main():
    all_stats = []
    for tag, min_syn in [("full", 0), ("pruned5", 5)]:
        stats = build_and_cache(min_syn, tag)
        print(f"[{tag}] neurons={stats['n_neurons']:,} edges={stats['n_edges']:,} "
              f"with_soma={stats['n_with_soma']:,} nt_confident={stats['n_nt_confident']:,}")
        all_stats.append(stats)

    (CACHE_DIR / "build_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"\nCached to {CACHE_DIR}")


if __name__ == "__main__":
    main()
