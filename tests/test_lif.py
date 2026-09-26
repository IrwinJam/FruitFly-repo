"""
Unit tests for the LIF simulator on tiny hand-built graphs with known answers.
Run: .venv/Scripts/python -m pytest tests -q
"""
import math

import numpy as np
import pytest
import torch

from amongusfly.brain.lif_torch import LIFBrain

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def chain_edges(weight_mv: float, n: int = 3):
    """0 -> 1 -> 2 ... with a single weight."""
    pre = np.arange(n - 1)
    post = np.arange(1, n)
    return np.stack([pre, post]), np.full(n - 1, weight_mv, dtype=np.float32), n


def make(edges, **kw):
    return LIFBrain(tag="test", device=DEV, edges=edges, **kw)


def test_rest_is_stable():
    b = make(chain_edges(1.0))
    b.reset(2)
    out = b.run(2000)
    assert int(out["counts"].sum()) == 0
    assert torch.allclose(b.v, torch.full_like(b.v, b.params.v_rest))


def test_single_kick_matches_analytic_peak():
    """One kick of g0 at t=0: v(t) - v0 = g0*tau/(tau-tm)*(e^{-t/tau} - e^{-t/tm})."""
    g0 = 5.0
    b = make(chain_edges(0.0), dt_ms=0.1)
    b.reset(1)
    ext = torch.zeros(b.n_neurons, 1, device=DEV)
    ext[0, 0] = g0
    vs = []
    for t in range(600):
        b.step(ext if t == 0 else None)
        vs.append(b.v[0, 0].item() - b.params.v_rest)
    tau, tm = b.params.tau_syn_ms, b.params.tau_m_ms
    t_peak = tau * tm / (tm - tau) * math.log(tm / tau)  # argmax of the curve
    v_peak = g0 * tau / (tau - tm) * (math.exp(-t_peak / tau) - math.exp(-t_peak / tm))
    assert max(vs) == pytest.approx(v_peak, rel=1e-3)


def test_dt_05_tracks_dt_01_for_single_kick():
    peaks = {}
    for dt in (0.1, 0.5):
        b = make(chain_edges(0.0), dt_ms=dt)
        b.reset(1)
        ext = torch.zeros(b.n_neurons, 1, device=DEV)
        ext[0, 0] = 5.0
        vmax = -1e9
        for t in range(int(60 / dt)):
            b.step(ext if t == 0 else None)
            vmax = max(vmax, b.v[0, 0].item())
        peaks[dt] = vmax
    assert peaks[0.5] == pytest.approx(peaks[0.1], abs=0.02)


def test_synaptic_delay_steps():
    """A spike in neuron 0 must reach neuron 1's g exactly delay_steps after it fires."""
    b = make(chain_edges(3.0), dt_ms=0.1)
    assert b.delay_steps == 18
    b.reset(1)
    ext = torch.zeros(b.n_neurons, 1, device=DEV)
    ext[0, 0] = 1000.0  # force an immediate spike
    spike_step, arrive_step = None, None
    for t in range(60):
        s = b.step(ext if t == 0 else None)
        if spike_step is None and s[0, 0]:
            spike_step = t
        if arrive_step is None and b.g[1, 0].item() > 0:
            arrive_step = t
    # the kick enters g first, so v needs a step to cross threshold; only the
    # spike -> arrival gap is the delay under test
    assert spike_step is not None
    assert arrive_step - spike_step == b.delay_steps


def test_refractory_caps_rate():
    b = make(chain_edges(0.0), dt_ms=0.1)
    b.reset(1)
    b.set_stimulus([0], [0], [10000.0])  # every step
    out = b.run(10000)  # 1 s
    rate = int(out["counts"][0, 0])
    expected = 1000.0 / ((b.refractory_steps + 1) * b.dt_ms)  # ~434.8 Hz at t_rfc=2.2, dt=0.1
    assert rate == pytest.approx(expected, rel=0.01)


def test_poisson_rate_drives_at_or_slightly_above_rate():
    """
    Shiu-style stimulation (68.75 mV kicks) makes a neuron fire at roughly the input
    rate, but slightly ABOVE it: a kick still carries ~42 mV of g after the refractory
    period, so two kicks close together can add an extra spike (measured ~1.13x at
    50 Hz). This is a property of the reference model's equations, not a bug, so
    experiments must report measured stimulated-neuron rates, not nominal ones.
    """
    b = make(chain_edges(0.0))
    b.reset(8, seed=0)
    b.set_stimulus([0] * 8, list(range(8)), [50.0] * 8)
    out = b.run(int(4000 / b.dt_ms))  # 4 s
    ratio = out["counts"][0].float().mean().item() / 4.0 / 50.0
    assert 1.0 <= ratio <= 1.3


def test_event_and_spmm_propagation_agree():
    rng = np.random.default_rng(0)
    n, e = 400, 8000
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    w = rng.normal(1.5, 1.0, e).astype(np.float32)
    edges = (np.stack([pre, post]), w, n)
    runs = {}
    for mode in ("event", "spmm"):
        b = make(edges, propagation=mode)
        b.reset(3, seed=1)
        b.set_stimulus(list(range(20)) * 3, [c for c in range(3) for _ in range(20)], [80.0] * 60)
        runs[mode] = b.run(2000)["counts"].cpu()
    assert torch.equal(runs["event"], runs["spmm"])


def test_silencing_blocks_transmission():
    b = make(chain_edges(40.0))
    b.reset(1)
    b.set_stimulus([0], [0], [100.0])
    assert int(b.run(4000)["counts"][1, 0]) > 0
    b.reset(1)
    b.silence([0])
    assert int(b.run(4000)["counts"][1, 0]) == 0
