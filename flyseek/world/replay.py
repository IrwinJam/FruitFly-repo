"""
Replay recorder: per-tick agent states plus sparse per-fly spike counts, saved as
one .npz (arrays) + one .json (metadata). The viewer plays these back; the brain
simulation itself is slower than real time (docs/PHASE1_REPORT.md section 5).

Spikes are stored as, for every (tick, fly): the local graph indices of neurons
that spiked at least once in that tick (uint32) and their spike counts (uint8,
clipped at 255). offsets[t * n_flies + f] marks where each block starts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATE_FIELDS = ["x", "y", "heading", "speed", "omega", "alive"]


class ReplayRecorder:
    def __init__(self, n_flies: int, meta: dict):
        self.n = n_flies
        self.meta = dict(meta)
        self.states: list[np.ndarray] = []
        self.spike_idx: list[np.ndarray] = []
        self.spike_cnt: list[np.ndarray] = []
        self.offsets = [0]
        self.events: list[dict] = []

    def record(self, state: dict, counts_nb: np.ndarray | None):
        """state: field -> [A]; counts_nb: [N, A] spike counts this tick (or None)."""
        self.states.append(np.stack([np.asarray(state[f], dtype=np.float32) for f in STATE_FIELDS], axis=1))
        for f in range(self.n):
            if counts_nb is None:
                idx = np.empty(0, np.uint32)
                cnt = np.empty(0, np.uint8)
            else:
                col = counts_nb[:, f]
                idx = np.flatnonzero(col).astype(np.uint32)
                cnt = np.minimum(col[idx], 255).astype(np.uint8)
            self.spike_idx.append(idx)
            self.spike_cnt.append(cnt)
            self.offsets.append(self.offsets[-1] + len(idx))

    def event(self, tick: int, kind: str, **data):
        self.events.append({"tick": tick, "kind": kind, **data})

    def save(self, path: Path) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            states=np.stack(self.states) if self.states else np.empty((0, self.n, len(STATE_FIELDS)), np.float32),
            spike_idx=np.concatenate(self.spike_idx) if self.spike_idx else np.empty(0, np.uint32),
            spike_cnt=np.concatenate(self.spike_cnt) if self.spike_cnt else np.empty(0, np.uint8),
            offsets=np.asarray(self.offsets, dtype=np.int64),
        )
        meta = {**self.meta, "n_flies": self.n, "n_ticks": len(self.states), "state_fields": STATE_FIELDS,
                "events": self.events}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        size = path.with_suffix(".npz").stat().st_size
        return {"npz_bytes": size, "n_ticks": len(self.states), "total_spike_entries": self.offsets[-1]}
