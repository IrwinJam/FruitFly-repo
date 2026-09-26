"""
Batched leaky integrate-and-fire simulation of the MaleCNS connectome on GPU, following the
Shiu et al. (2024) reference model:

    dv/dt = (v_0 - v + g) / t_mbr     (unless refractory)
    dg/dt = -g / tau
    on presynaptic spike, after t_dly:  g_post += w      (w = n_synapses * w_syn * sign)
    stimulated neurons: Poisson input at rate r, each input spike adds w_syn * f_poi

One weight matrix is shared by the batch; state has a batch dimension, so each column can be
a different fly or condition. The (v, g) update is exact over a step, the synaptic delay is a
ring buffer and the refractory period a step counter. Silencing masks a neuron's spikes before
propagation. Propagation uses a sparse matmul or event-driven updates, whichever is cheaper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from amongusfly.paths import CACHE_DIR, CONFIG_DIR

CONFIG_PATH = CONFIG_DIR / "brain.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class LIFParams:
    v_rest: float
    v_threshold: float
    v_reset: float
    tau_m_ms: float
    tau_syn_ms: float
    refractory_ms: float
    delay_ms: float
    mv_per_synapse: float
    poisson_weight_factor: float

    @classmethod
    def from_config(cls, cfg: dict) -> "LIFParams":
        p = cfg["lif"]
        return cls(
            v_rest=p["v_rest_mv"],
            v_threshold=p["v_threshold_mv"],
            v_reset=p["v_reset_mv"],
            tau_m_ms=p["tau_m_ms"],
            tau_syn_ms=p["tau_syn_ms"],
            refractory_ms=p["refractory_ms"],
            delay_ms=p["synaptic_delay_ms"],
            mv_per_synapse=p["mv_per_synapse"],
            poisson_weight_factor=p["poisson_weight_factor"],
        )

    @property
    def poisson_kick_mv(self) -> float:
        return self.mv_per_synapse * self.poisson_weight_factor


MODULATORY_NTS = ("dopamine", "serotonin", "octopamine")


def load_edges(
    tag: str,
    weight_scale: float | None = None,
    modulatory_sign: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (edge_index [2,E] = (pre, post), edge_weight [E] in mV, n_neurons).
    Applies the dataset calibration from brain.yaml (connectome_weight_scale,
    modulatory_sign) unless overridden by the arguments.
    """
    cfg = load_config()
    scale = cfg.get("connectome_weight_scale", 1.0) if weight_scale is None else weight_scale
    mod_sign = cfg.get("modulatory_sign", 1) if modulatory_sign is None else modulatory_sign

    edge_index = np.load(CACHE_DIR / f"edge_index_{tag}.npy")
    edge_weight = np.load(CACHE_DIR / f"edge_weight_{tag}.npy")

    import pandas as pd

    from amongusfly.brain.roles import full_idx_of

    nt = pd.read_parquet(CACHE_DIR / "neurons.parquet", columns=["idx", "nt"]).sort_values("idx")
    is_mod_full = nt["nt"].isin(MODULATORY_NTS).to_numpy()
    full = full_idx_of(tag)  # subgraph (e.g. navcore) -> full-graph idx, or None
    if full is not None:
        n_neurons = len(full)
        is_mod_node = is_mod_full[full]
    else:
        n_neurons = len(nt)
        is_mod_node = is_mod_full
    if mod_sign != 1:
        is_mod = is_mod_node[edge_index[0]]
        # cached weights carry sign +1 for modulatory NTs; replace with mod_sign
        edge_weight = np.where(is_mod, np.abs(edge_weight) * mod_sign, edge_weight)
    if scale != 1.0:
        edge_weight = edge_weight * scale
    return edge_index, edge_weight.astype(np.float32), n_neurons


class LIFBrain:
    def __init__(
        self,
        tag: str = "pruned5",
        device: str | None = None,
        dt_ms: float | None = None,
        propagation: str | None = None,
        delay: bool = True,
        edges: tuple[np.ndarray, np.ndarray, int] | None = None,
    ):
        cfg = load_config()
        self.params = p = LIFParams.from_config(cfg)
        self.device = torch.device(device or cfg.get("device", "cpu"))
        self.tag = tag
        self.dt_ms = dt = float(dt_ms if dt_ms is not None else cfg["sim"]["dt_ms"])
        self.propagation = propagation or cfg["sim"].get("propagation", "auto")

        edge_index, edge_weight, n = edges if edges is not None else load_edges(tag)
        self.n_neurons = n
        self.n_edges = int(edge_index.shape[1])
        self._build_graph(edge_index, edge_weight)

        # exact one-step propagators for the linear (v, g) system
        self.a_v = math.exp(-dt / p.tau_m_ms)
        self.b_g = math.exp(-dt / p.tau_syn_ms)
        self.c_g = p.tau_syn_ms / (p.tau_syn_ms - p.tau_m_ms) * (self.b_g - self.a_v)
        # Brian2 holds a neuron refractory while (t - lastspike) <= t_rfc, i.e. for
        # floor(t_rfc/dt) steps after the spike step. Min ISI = (steps + 1) * dt.
        self.refractory_steps = max(1, math.floor(p.refractory_ms / dt + 1e-9))
        self.delay_steps = max(1, round(p.delay_ms / dt)) if delay else 1

        self.batch_size = 0
        self.keep_mask: torch.Tensor | None = None  # [N] 1 = normal, 0 = silenced
        self.clear_stimulus()

    # ------------------------------------------------------------------ graph
    def _build_graph(self, edge_index: np.ndarray, edge_weight: np.ndarray):
        dev, n = self.device, self.n_neurons
        pre = torch.as_tensor(edge_index[0], dtype=torch.int64)
        post = torch.as_tensor(edge_index[1], dtype=torch.int64)
        w = torch.as_tensor(edge_weight, dtype=torch.float32)

        # CSR keyed by presynaptic neuron, for event-driven propagation
        order = torch.argsort(pre, stable=True)
        pre_sorted = pre[order]
        self.out_post = post[order].to(dev)
        self.out_w = w[order].to(dev)
        out_deg = torch.bincount(pre_sorted, minlength=n)
        self.out_deg = out_deg.to(dev)
        self.out_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(out_deg, 0)]).to(dev)

        # CSR A[post, pre] for whole-graph sparse matmul
        A = torch.sparse_coo_tensor(torch.stack([post, pre]), w, size=(n, n)).coalesce()
        self.A = A.to_sparse_csr().to(dev)

    # --------------------------------------------------------------- stimulus
    def clear_stimulus(self):
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        self._stim_neuron = empty
        self._stim_col = empty
        self._stim_prob = torch.empty(0, device=self.device)

    def set_stimulus(self, neurons, cols, rates_hz):
        """
        Poisson stimulation, Shiu-style. neurons/cols/rates are 1-D, same length:
        neuron idx, batch column, rate in Hz. Replaces any previous stimulus.
        """
        dev = self.device
        neurons = np.asarray(neurons, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if len(neurons) and (neurons.min() < 0 or neurons.max() >= self.n_neurons):
            raise IndexError(
                f"stimulus neuron index out of range for graph '{self.tag}' ({self.n_neurons} neurons). "
                "Look up roles with graph=<tag> when using a subgraph such as navcore."
            )
        if len(cols) and self.batch_size and cols.max() >= self.batch_size:
            raise IndexError(f"stimulus column {cols.max()} >= batch size {self.batch_size}; call reset() first")
        self._stim_neuron = torch.as_tensor(neurons, dtype=torch.int64, device=dev)
        self._stim_col = torch.as_tensor(cols, dtype=torch.int64, device=dev)
        rates = torch.as_tensor(rates_hz, dtype=torch.float32, device=dev)
        self._stim_prob = torch.clamp(rates * (self.dt_ms / 1000.0), 0.0, 1.0)

    def silence(self, neurons):
        mask = torch.ones(self.n_neurons, device=self.device)
        if len(neurons):
            mask[torch.as_tensor(neurons, dtype=torch.int64, device=self.device)] = 0.0
        self.keep_mask = mask

    # ------------------------------------------------------------------ state
    def reset(self, batch_size: int, seed: int | None = None):
        n, dev, p = self.n_neurons, self.device, self.params
        self.batch_size = batch_size
        self.v = torch.full((n, batch_size), p.v_rest, device=dev)
        self.g = torch.zeros((n, batch_size), device=dev)
        self.refrac = torch.zeros((n, batch_size), dtype=torch.int16, device=dev)
        self.delay_buf = torch.zeros((self.delay_steps, n, batch_size), device=dev)
        self.buf_ptr = 0
        self.gen = torch.Generator(device=dev)
        if seed is not None:
            self.gen.manual_seed(seed)

    # ------------------------------------------------------------------- step
    def _propagate(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: [N,B] bool -> postsynaptic current [N,B]."""
        n, B = self.n_neurons, self.batch_size
        src = spikes if self.keep_mask is None else spikes & (self.keep_mask[:, None] > 0)

        mode = self.propagation
        if mode != "spmm":
            pre_idx, col_idx = src.nonzero(as_tuple=True)
            if len(pre_idx) == 0:
                return torch.zeros((n, B), device=self.device)
            deg = self.out_deg[pre_idx]
            total = int(deg.sum())
            if mode == "auto" and total > 0.3 * self.n_edges * B:
                mode = "spmm"
            else:
                out = torch.zeros(n * B, device=self.device)
                if total == 0:
                    return out.view(n, B)
                which = torch.repeat_interleave(torch.arange(len(pre_idx), device=self.device), deg, output_size=total)
                starts = self.out_ptr[pre_idx]
                first_of = torch.cumsum(deg, 0) - deg
                edge = starts[which] + (torch.arange(total, device=self.device) - first_of[which])
                out.index_add_(0, self.out_post[edge] * B + col_idx[which], self.out_w[edge])
                return out.view(n, B)

        return torch.sparse.mm(self.A, src.float())

    @torch.no_grad()
    def step(self, ext_current: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt. ext_current [N,B] (mV added to g) is optional. Returns spikes [N,B] bool."""
        p = self.params

        # 1. delayed synaptic input arriving now, plus stimulus spikes
        self.g += self.delay_buf[self.buf_ptr]
        if len(self._stim_neuron):
            # no boolean masking here: masking forces a GPU->CPU sync every step
            fire = torch.rand(len(self._stim_prob), device=self.device, generator=self.gen) < self._stim_prob
            self.g.index_put_(
                (self._stim_neuron, self._stim_col),
                fire.to(self.g.dtype) * p.poisson_kick_mv,
                accumulate=True,
            )
        if ext_current is not None:
            self.g += ext_current

        # 2. exact integration over dt for non-refractory neurons
        active = self.refrac <= 0
        v_new = p.v_rest + (self.v - p.v_rest) * self.a_v + self.g * self.c_g
        self.v = torch.where(active, v_new, self.v)
        self.g *= self.b_g

        # 3. threshold, reset, refractory
        spikes = active & (self.v >= p.v_threshold)
        self.v = torch.where(spikes, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.where(
            spikes,
            torch.full_like(self.refrac, self.refractory_steps),
            torch.clamp(self.refrac - 1, min=0),
        )

        # 4. schedule this step's spikes to arrive after the synaptic delay
        self.delay_buf[self.buf_ptr] = self._propagate(spikes)
        self.buf_ptr = (self.buf_ptr + 1) % self.delay_steps
        return spikes

    @torch.no_grad()
    def run(self, n_steps: int, count_neurons: torch.Tensor | None = None, bin_steps: int = 0):
        """
        Run n_steps. Returns dict with:
          counts:      [len(count_neurons), B] spike counts (or [N,B] if count_neurons is None)
          pop_rate_hz: [n_bins, B] mean firing rate across all neurons per bin (if bin_steps > 0)
        """
        idx = count_neurons
        counts = torch.zeros((self.n_neurons if idx is None else len(idx), self.batch_size),
                             dtype=torch.int32, device=self.device)
        bins = []
        bin_acc = torch.zeros(self.batch_size, device=self.device)
        for t in range(n_steps):
            s = self.step()
            counts += (s if idx is None else s[idx]).to(torch.int32)
            if bin_steps:
                bin_acc += s.sum(0).float()
                if (t + 1) % bin_steps == 0:
                    bins.append(bin_acc / (self.n_neurons * bin_steps * self.dt_ms / 1000.0))
                    bin_acc = torch.zeros(self.batch_size, device=self.device)
        out = {"counts": counts}
        if bin_steps:
            out["pop_rate_hz"] = torch.stack(bins) if bins else torch.empty(0, self.batch_size)
        return out

    def num_edges(self) -> int:
        return self.n_edges
