"""
Download the MaleCNS v1.0 connectome data files needed to build the brain graph.

Source: https://male-cns.janelia.org/download/
All files are public (CC-BY 4.0), served over plain HTTPS with no auth required.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

DATA_DIR = Path(r"C:\flyseek-data\raw\malecns_v1")


@dataclass
class FileSpec:
    name: str
    expected_size: int  # bytes, from a verified HEAD request
    required: bool


FILES = [
    FileSpec(
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        14_483_314,
        required=True,
    ),
    FileSpec(
        "body-neurotransmitters-male-cns-v1.0.feather",
        43_282_834,
        required=True,
    ),
    FileSpec(
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1_051_241_946,
        required=True,
    ),
]


def download_file(spec: FileSpec, dest_dir: Path) -> Path:
    dest = dest_dir / spec.name
    url = f"{BASE_URL}/{spec.name}"

    if dest.exists() and dest.stat().st_size == spec.expected_size:
        print(f"[skip] {spec.name} already present ({dest.stat().st_size:,} bytes)")
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = spec.expected_size
        mode = "ab" if resume_from else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=spec.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    actual_size = tmp.stat().st_size
    if actual_size != spec.expected_size:
        raise RuntimeError(
            f"{spec.name}: downloaded {actual_size:,} bytes, expected {spec.expected_size:,}"
        )

    tmp.rename(dest)
    return dest


def main() -> int:
    print(f"Downloading MaleCNS v1.0 connectome data to {DATA_DIR}")
    ok = True
    for spec in FILES:
        try:
            path = download_file(spec, DATA_DIR)
            print(f"[ok] {path} ({path.stat().st_size:,} bytes)")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {spec.name}: {e}")
            ok = not spec.required and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
