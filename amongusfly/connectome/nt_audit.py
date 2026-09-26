"""
Audit: how would switching the neurotransmitter source to the dataset's
`consensus_nt` column change the model? Counts traced neurons whose NT label or
excitatory/inhibitory SIGN would change, weighted by outgoing synapses.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from amongusfly.connectome.build_graph import DEFAULT_SIGN, NT_FILE, NT_SIGN
from amongusfly.paths import CACHE_DIR, DOCS_DIR


def main():
    neu = pd.read_parquet(CACHE_DIR / "neurons.parquet")[["idx", "bodyId", "type", "nt", "sign"]]
    nt = pd.read_feather(NT_FILE)[["body", "consensus_nt"]].rename(columns={"body": "bodyId"})
    df = neu.merge(nt, on="bodyId", how="left").sort_values("idx").reset_index(drop=True)

    cons = df["consensus_nt"].where(df["consensus_nt"].notna() & (df["consensus_nt"] != "unclear"))
    new_nt = cons.fillna(df["nt"])  # consensus first, fall back to the current label
    new_sign = new_nt.map(NT_SIGN).fillna(DEFAULT_SIGN).astype(np.int8)

    ei = np.load(CACHE_DIR / "edge_index_full.npy")
    ew = np.abs(np.load(CACHE_DIR / "edge_weight_full.npy"))
    out_w = np.bincount(ei[0], weights=ew, minlength=len(df))

    label_changed = (new_nt != df["nt"]).to_numpy()
    sign_changed = (new_sign != df["sign"]).to_numpy()
    res = {
        "traced_neurons": int(len(df)),
        "consensus_available_not_unclear": int(cons.notna().sum()),
        "label_changed_neurons": int(label_changed.sum()),
        "sign_changed_neurons": int(sign_changed.sum()),
        "sign_changed_frac_of_output_weight": float(out_w[sign_changed].sum() / out_w.sum()),
        "label_transitions_top": (
            pd.Series(list(zip(df["nt"][label_changed], new_nt[label_changed]))).value_counts().head(12)
            .rename(lambda t: f"{t[0]} -> {t[1]}").to_dict()
        ),
        "sign_change_types_top": df.loc[sign_changed, "type"].value_counts().head(15).to_dict(),
    }
    (DOCS_DIR / "results" / "neurotransmitter_audit.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
