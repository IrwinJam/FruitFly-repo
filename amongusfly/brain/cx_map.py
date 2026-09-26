"""
Map the central-complex compass circuit onto MaleCNS neurons.

Parses each EPG, FC2 and PFL3 neuron's bridge glomerulus and fan-shaped-body column from its
instance name (e.g. EPG(PB08)_R2, FC2A_C2_L, PFL3(PB12c)_R2_C1), assigns a nominal preferred
angle, and measures from the connectome which side each PFL3 steers. Writes CACHE_DIR/cx_map.json.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from amongusfly.brain.roles import neurons
from amongusfly.paths import CACHE_DIR, DOCS_DIR

PB_RE = re.compile(r"_([LR])(\d)")
COL_RE = re.compile(r"_C(\d)")


def parse(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        inst = str(r["instance"])
        pb = PB_RE.search(inst.split(")")[-1]) if "(PB" in inst else None
        col = COL_RE.search(inst)
        out.append({
            "idx": int(r["idx"]), "bodyId": int(r["bodyId"]), "type": r["type"], "instance": inst,
            "somaSide": r["somaSide"],
            "pb_side": pb.group(1) if pb else None, "pb_glom": int(pb.group(2)) if pb else None,
            "fb_col": int(col.group(1)) if col else None,
            "irregular": "irreg" in inst,
        })
    p = pd.DataFrame(out)
    p["pb_phase_deg"] = (p["pb_glom"] - 1) * 45.0
    p["fb_phase_deg"] = (p["fb_col"] - 1) * 40.0
    return p


def main():
    df = neurons().sort_values("idx").reset_index(drop=True)
    ann = pd.read_feather(__import__("amongusfly.paths", fromlist=["RAW_DIR"]).RAW_DIR /
                          "body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=["bodyId", "instance"])
    df = df.merge(ann, on="bodyId", how="left")
    cx = df[df["type"].isin(["EPG", "FC2A", "FC2B", "FC2C", "PFL3"])]
    p = parse(cx)

    ei = np.load(CACHE_DIR / "edge_index_full.npy")
    w = np.abs(np.load(CACHE_DIR / "edge_weight_full.npy")) / 0.275  # synapse counts
    n = len(df)
    type_of = df["type"].fillna("").to_numpy()
    side_of = df["somaSide"].fillna("").to_numpy()

    def syn_between(pre_mask, post_mask):
        m = pre_mask[ei[0]] & post_mask[ei[1]]
        return float(w[m].sum())

    is_type = lambda *ts: np.isin(type_of, ts)
    epg, fc2, pfl3 = is_type("EPG"), is_type("FC2A", "FC2B", "FC2C"), is_type("PFL3")
    totals = {
        "EPG->PFL3": syn_between(epg, pfl3),
        "FC2->PFL3": syn_between(fc2, pfl3),
        "PFL3->DNa02": syn_between(pfl3, is_type("DNa02")),
        "PFL3->DNa03": syn_between(pfl3, is_type("DNa03")),
        "EPG->Delta7": syn_between(epg, is_type("Delta7")),
        "Delta7->PFL3": syn_between(is_type("Delta7"), pfl3),
    }

    # per-PFL3 steering side, measured from output synapses
    lal = np.char.startswith(type_of.astype(str), "LAL")
    dn = is_type("DNa02", "DNa03")
    rows = []
    for _, r in p[p["type"] == "PFL3"].iterrows():
        m = ei[0] == r["idx"]
        post, ww = ei[1][m], w[m]
        def side_w(mask, side):
            sel = mask[post] & (side_of[post] == side)
            return float(ww[sel].sum())
        rows.append({"idx": r["idx"], "instance": r["instance"],
                     "dn_L": side_w(dn, "L"), "dn_R": side_w(dn, "R"),
                     "lal_L": side_w(lal, "L"), "lal_R": side_w(lal, "R")})
    steer = pd.DataFrame(rows)
    steer["steer_side"] = np.where(steer["lal_L"] + steer["dn_L"] >= steer["lal_R"] + steer["dn_R"], "L", "R")
    p = p.merge(steer[["idx", "steer_side", "dn_L", "dn_R", "lal_L", "lal_R"]], on="idx", how="left")

    # top postsynaptic partner types of PFL3 (full graph), to see the real downstream path
    m = pfl3[ei[0]]
    top_post = (pd.Series(w[m]).groupby(type_of[ei[1][m]]).sum().sort_values(ascending=False).head(15).round(0).to_dict())

    pfl3p = p[p["type"] == "PFL3"]
    summary = {
        "counts": p["type"].value_counts().to_dict(),
        "synapse_totals": totals,
        "pfl3_top_postsynaptic_types": top_post,
        "pfl3_pb_side_vs_steer_side": pd.crosstab(pfl3p["pb_side"], pfl3p["steer_side"]).to_dict(),
        "pfl3_soma_side_vs_steer_side": pd.crosstab(pfl3p["somaSide"], pfl3p["steer_side"]).to_dict(),
        "unparsed": p[(p["pb_glom"].isna()) & (p["type"].isin(["EPG", "PFL3"])) |
                      (p["fb_col"].isna()) & (p["type"].isin(["FC2A", "FC2B", "FC2C", "PFL3"]))]["instance"].tolist(),
    }
    (CACHE_DIR / "cx_map.json").write_text(p.to_json(orient="records"))
    (DOCS_DIR / "results" / "compass_map.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(p[p["type"] == "PFL3"][["instance", "somaSide", "pb_side", "pb_glom", "fb_col", "steer_side", "dn_L", "dn_R", "lal_L", "lal_R"]].to_string())


if __name__ == "__main__":
    main()
