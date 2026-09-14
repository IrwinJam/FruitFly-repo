"""
Resolve the logical sensory/motor "roles" used by FlySeek (see PROJECT_PLAN.md section
3.6) against the actual `type` names in the MaleCNS v1.0 annotations table.

This module is deliberately strict: every role must resolve to at least one real
`type` value present in the annotations table, or the loader raises. Silent guessing
is exactly what the project plan says to avoid.

Findings baked in here (from the M1 resolution pass, 2026-09-13, MaleCNS v1.0):

* Most single descending neurons (DNa01, DNa02, DNa03, DNp01, DNp09, DNp10, DNg13,
  DNg100, MDN, MN9) exist as exactly 2 traced bodies (one per side, as expected for a
  single command-neuron pair), each with a soma position.
* **P1 is not a `type` in MaleCNS.** Cross-referencing the `synonyms` column (which
  records prior literature names) shows `type == "pIP1"` carries the synonym
  "Yu 2010: pIP1" and "Cachero 2010: pIP-a" — the classic P1/pIP-a courtship-arousal
  cluster. `pIP10` is a distinct, separately-traced descending neuron that shares the
  same historical reference and is the more plausible *descending* arm of the P1
  pathway. We treat `pIP1` as the arousal population (tonic P1-style drive) and
  `pIP10` as a secondary descending readout candidate. Flagged uncertain — verify
  against Cachero et al. 2010 / Yu et al. 2010 / recent MaleCNS papers before relying
  on this for anything beyond a demo.
* **`oDN1` does not exist in MaleCNS.** It appears in the Eon Systems writeup, whose
  brain model is built from the *female* FlyWire connectome. oDN1 ("oviposition
  descending neuron 1") is a female egg-laying circuit neuron with no male
  counterpart — it should not have been in the original plan's target list for a
  male-CNS project. We substitute **DNg100** (used by the Fly64/Mario project on this
  same MaleCNS data) as the primary forward-drive descending neuron, with DNp09 as a
  secondary contributor.
* **Photoreceptors have no single "R1"..."R8" type.** R1-R6 (the achromatic/motion
  channel) are pooled into one type, `R1-R6` (n=3,377). Color/UV photoreceptors R7 and
  R8 are split by rhabdomere subtype into pale/yellow/dorsal-rim variants: `R7d`,
  `R7p`, `R7y` and `R8d`, `R8p`, `R8y`.
* **Soma positions are missing for ~85% of traced neurons overall, and for nearly
  ALL photoreceptors and antennal ORNs** (13/3,377 for R1-R6; 0/896 for R7*;
  11/887 for R8*; 0 for every `ORN_*` type checked) — their somas sit in the retina
  or antenna, outside typical segmentation coverage. The brain-panel layout step
  (`layout.py`) must fall back to a synapse-centroid or deterministic placement for
  these, not assume soma_location is always present.
* **No gustatory (sugar/bitter) receptor neurons resolve by name at minconf 0.5.**
  Patterns for `Gr\\d`, `gustGRN`, `sugGRN`, `bitGRN`, `pharyng` all return zero rows;
  the only `*GRN` hits (`claw_tpGRN`, `dorsal_tpGRN`) are leg tactile-placode
  mechanosensory neurons, not gustatory ones. This isn't a problem for FlySeek
  (Among Us has no feeding behavior to model) but means the "danger" and "ping"
  odor channels use genuine **antennal ORNs** (olfactory, not gustatory) chosen by
  their well-established glomerulus identity from the literature: `ORN_DA2`
  (Or56a, geosmin — a strong innate aversive) and `ORN_V` (Gr21a/Gr63a, CO2 —
  aversive) for danger; `ORN_VA2` (Or92a) and `ORN_DM1` (Or42b) — both classic
  food-odor/vinegar attractants — for the ping-direction channel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"C:\flyseek-data\raw\malecns_v1")
ANNOTATIONS_FILE = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

# role -> (list of `type` values, human note)
# "uncertain": True marks roles that need biological verification before being
# trusted for anything beyond a visual demo (see module docstring).
ROLES: dict[str, dict] = {
    # --- vision: photoreceptors (sensory input) ---
    "photoreceptor_achromatic": {"types": ["R1-R6"], "uncertain": False},
    "photoreceptor_color": {
        "types": ["R7d", "R7p", "R7y", "R8d", "R8p", "R8y"],
        "uncertain": False,
    },
    # --- pursuit circuit (seeker) ---
    "target_motion_detector": {"types": ["LC10a"], "uncertain": False},
    "pursuit_relay": {"types": ["AOTU019", "AOTU025"], "uncertain": False},
    "arousal_gain": {
        "types": ["pIP1"],
        "uncertain": True,
        "note": "P1/pIP-a courtship-arousal cluster by synonym cross-reference, not an exact 'P1' type label in MaleCNS.",
    },
    "pursuit_descending": {"types": ["DNa03", "DNa02"], "uncertain": False},
    # --- looming / escape circuit (hider) ---
    "looming_expansion": {"types": ["LC4"], "uncertain": False},
    "looming_size": {"types": ["LPLC2"], "uncertain": False},
    "giant_fiber": {"types": ["DNp01"], "uncertain": False},
    # --- steering ---
    "steering_high_gain": {"types": ["DNa02"], "uncertain": False},
    "steering_low_gain": {"types": ["DNa01"], "uncertain": False},
    "steering_secondary": {"types": ["DNg13"], "uncertain": False},
    # --- forward / backward walking ---
    "forward_drive": {
        "types": ["DNg100", "DNp09"],
        "uncertain": True,
        "note": "oDN1 (used by Eon's female-FlyWire model) has no male-CNS counterpart. Substituted DNg100 (used by the Fly64/Mario MaleCNS project) + DNp09.",
    },
    "backward_drive": {"types": ["MDN"], "uncertain": False},
    # --- central complex / goal navigation (stretch) ---
    "heading_compass": {"types": ["EPG"], "uncertain": False},
    "goal_direction": {"types": ["FC2A", "FC2B", "FC2C"], "uncertain": False},
    "goal_steering": {"types": ["PFL3"], "uncertain": False},
    # --- feeding / misc reference ---
    "proboscis_motor": {"types": ["MN9"], "uncertain": False},
    # --- olfaction: danger meter / ping direction (glomerulus identity from literature,
    # not a gustatory or valence label in the dataset itself; see module docstring) ---
    "aversive_odor": {
        "types": ["ORN_DA2", "ORN_V"],
        "uncertain": True,
        "note": "ORN_DA2=Or56a (geosmin) and ORN_V=Gr21a/Gr63a (CO2) are innate-aversive glomeruli by established literature, not a labeled 'aversive' role in MaleCNS itself.",
    },
    "attractive_odor": {
        "types": ["ORN_VA2", "ORN_DM1"],
        "uncertain": True,
        "note": "ORN_VA2=Or92a and ORN_DM1=Or42b are classic food-odor/vinegar attractant glomeruli by established literature, not a labeled 'attractive' role in MaleCNS itself.",
    },
}


@dataclass
class ResolvedRole:
    role: str
    types: list[str]
    body_ids: list[int]
    n_bodies: int
    n_with_soma: int
    sides: dict
    superclass: str | None
    uncertain: bool
    note: str | None


def _has_soma(v) -> bool:
    return hasattr(v, "__len__") and len(v) == 3


def load_annotations(path: Path = ANNOTATIONS_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"MaleCNS annotations not found at {path}. Run "
            "`python -m flyseek.connectome.download` first."
        )
    return pd.read_feather(path)


def resolve_roles(df: pd.DataFrame | None = None) -> dict[str, ResolvedRole]:
    if df is None:
        df = load_annotations()

    resolved: dict[str, ResolvedRole] = {}
    missing: list[str] = []

    for role, spec in ROLES.items():
        types = spec["types"]
        rows = df[df["type"].isin(types)]
        if len(rows) == 0:
            missing.append(role)
            continue
        resolved[role] = ResolvedRole(
            role=role,
            types=types,
            body_ids=rows["bodyId"].tolist(),
            n_bodies=len(rows),
            n_with_soma=int(rows["somaLocation"].apply(_has_soma).sum()),
            sides=rows["somaSide"].value_counts().to_dict(),
            superclass=(
                rows["superclass"].mode().iloc[0]
                if rows["superclass"].notna().any()
                else None
            ),
            uncertain=spec.get("uncertain", False),
            note=spec.get("note"),
        )

    if missing:
        raise ValueError(
            f"Failed to resolve required roles against MaleCNS v1.0 annotations: {missing}. "
            "This dataset's naming may have changed — update flyseek/connectome/celltypes.py."
        )

    return resolved


def write_report(out_path: Path, resolved: dict[str, ResolvedRole]) -> None:
    lines = [
        "# Cell-type resolution report",
        "",
        "Generated by `flyseek.connectome.celltypes` against MaleCNS v1.0 annotations.",
        "",
        "| role | type(s) | n bodies | with soma | sides | superclass | uncertain |",
        "|---|---|---|---|---|---|---|",
    ]
    for role, r in resolved.items():
        flag = "⚠️" if r.uncertain else ""
        lines.append(
            f"| {role} | {', '.join(r.types)} | {r.n_bodies} | {r.n_with_soma} | "
            f"{r.sides} | {r.superclass} | {flag} |"
        )
    lines.append("")
    lines.append("## Notes on uncertain / substituted roles")
    lines.append("")
    for role, r in resolved.items():
        if r.uncertain and r.note:
            lines.append(f"- **{role}** ({', '.join(r.types)}): {r.note}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_body_id_map(out_path: Path, resolved: dict[str, ResolvedRole]) -> None:
    payload = {role: r.body_ids for role, r in resolved.items()}
    out_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    resolved = resolve_roles()
    report_path = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\docs\celltypes_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, resolved)

    cache_path = Path(r"C:\flyseek-data\cache\role_body_ids.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_body_id_map(cache_path, resolved)

    print(f"Resolved {len(resolved)}/{len(ROLES)} roles.")
    print(f"Report: {report_path}")
    print(f"Body-ID map cache: {cache_path}")
