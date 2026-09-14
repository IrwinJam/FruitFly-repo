"""
Batched leaky integrate-and-fire (LIF) simulation of the MaleCNS connectome on GPU.

One sparse weight matrix is shared across all flies in the batch; only the membrane
state (v, g, refractory countdown, spikes) has a batch dimension. This is what lets
adding more flies cost GPU memory, not a second copy of the graph (see
PROJECT_PLAN.md section 4.2).

Model: current-based (single-exponential) synapses, no explicit synaptic delay yet
(known simplification vs. the Shiu et al. / flypoke reference model's 1.8ms delay --
see module docstring in `flyseek/connectome/build_graph.py` for related notes; delay
will be added in Milestone M3 only if the W0 validation circuits need it to match
published behavior).

    g[t+dt] = g[t] * exp(-dt/tau_syn) + (incoming weighted spikes this step)
    v[t+dt] = v[t] + dt/tau_m * ( -(v[t] - v_rest) + g[t] )
    spike when v >= v_threshold -> v = v_reset, refractory for `refractory_ms`
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

CACHE_DIR = Path(r"C:\flyseek-data\cache")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "brain.yaml"


@dataclass
class LIFParams:
    v_rest: float
    v_threshold: float
    v_reset: float
    tau_m_ms: float
    tau_syn_ms: float
    refractory_ms: float

    @classmethod
    def from_yaml(cls, cfg: dict) -> "LIFParams":
        p = cfg["lif"]
        return cls(
            v_rest=p["v_rest_mv"],
            v_threshold=p["v_threshold_mv"],
            v_reset=p["v_reset_mv"],
            tau_m_ms=p["tau_m_ms"],
            tau_syn_ms=p["tau_syn_ms"],
            refractory_ms=p["refractory_ms"],
        )


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_graph(tag: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """
    Loads edge_index_{tag}.npy / edge_weight_{tag}.npy and returns a CSR sparse
    tensor A of shape [N,N] with A[post,pre] = weight (note the transpose vs. the
    stored (pre,post) edge_index — this orientation lets us compute
    `current[N,B] = torch.sparse.mm(A, spikes[N,B])` directly), plus N.
    """
    edge_index = np.load(CACHE_DIR / f"edge_index_{tag}.npy")  # [2, E] = (pre, post)
    edge_weight = np.load(CACHE_DIR / f"edge_weight_{tag}.npy")  # [E]

    n_neurons = int(edge_index.max()) + 1
    pre, post = edge_index[0], edge_index[1]

    indices = torch.tensor(np.stack([post, pre]), dtype=torch.int64, device=device)
    values = torch.tensor(edge_weight, dtype=torch.float32, device=device)

    A = torch.sparse_coo_tensor(indices, values, size=(n_neurons, n_neurons))
    A = A.coalesce().to_sparse_csr()
    return A, n_neurons


class LIFBrain:
    def __init__(self, tag: str = "pruned5", device: str | None = None):
        cfg = load_config()
        self.params = LIFParams.from_yaml(cfg)
        self.device = torch.device(device or cfg.get("device", "cpu"))
        self.tag = tag

        self.A, self.n_neurons = load_graph(tag, self.device)

        self.decay_syn = math.exp(-cfg["sim"]["dt_ms"] / self.params.tau_syn_ms)
        self.dt_ms = cfg["sim"]["dt_ms"]

        self.batch_size = 0
        self.v: torch.Tensor | None = None
        self.g: torch.Tensor | None = None
        self.refrac: torch.Tensor | None = None

    def reset(self, batch_size: int):
        self.batch_size = batch_size
        n, dev = self.n_neurons, self.device
        self.v = torch.full((n, batch_size), self.params.v_rest, device=dev)
        self.g = torch.zeros((n, batch_size), device=dev)
        self.refrac = torch.zeros((n, batch_size), device=dev)

    @torch.no_grad()
    def step(self, ext_current: torch.Tensor | None = None) -> torch.Tensor:
        """
        ext_current: optional [N, B] tensor of external drive (mV-equivalent) added
        to the synaptic trace this step (e.g. Poisson sensory input).
        Returns spikes: [N, B] bool tensor for this step.
        """
        p = self.params
        dt = self.dt_ms

        # synaptic current from network spikes of the *previous* step is applied via
        # self.g accumulation below; here we decay g then inject new spikes' effect.
        self.g = self.g * self.decay_syn

        # Integrate membrane potential for neurons not in refractory period.
        active = self.refrac <= 0
        dv = (dt / p.tau_m_ms) * (-(self.v - p.v_rest) + self.g)
        self.v = torch.where(active, self.v + dv, self.v)

        if ext_current is not None:
            self.g = self.g + ext_current

        # Spike detection + reset
        spiking = active & (self.v >= p.v_threshold)
        self.v = torch.where(spiking, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.where(
            spiking, torch.full_like(self.refrac, p.refractory_ms), self.refrac
        )
        self.refrac = torch.clamp(self.refrac - dt, min=0.0)

        # Propagate this step's spikes through the network -> next step's synaptic input.
        spikes_f = spiking.float()
        network_current = torch.sparse.mm(self.A, spikes_f)  # [N,B]
        self.g = self.g + network_current

        return spiking

    def num_edges(self) -> int:
        return self.A._nnz()
