# Phase 1 report — making the brain trustworthy

*2026-09-13. All numbers come from files in `docs/` produced by the scripts named below. Nothing is estimated unless it's labeled that way.*

## TL;DR

1. **The uncalibrated model was broken for our purposes.** At Shiu et al.'s 0.275 mV per synapse, a 50 ms pulse on a *single* neuron ignited a self-sustaining ~11 Hz state across ~17k neurons. Descending-neuron (DN) readouts then stopped tracking input. This explains last session's saturated or meaningless M3 results.
2. **Cause: a dataset calibration mismatch.** The MaleCNS brain has ~698 input synapses per neuron vs ~359 in FlyWire, where w_syn was calibrated. Scaling weights by 359/698 = **0.514**, chosen from dataset statistics before looking at behavioral effects, removes ignition.
3. **With calibration, the wiring carries side-specific signals that shuffled wiring does not:**
   - LC10a (target detector) drive flips the DNa02/DNa03 left–right difference with side.
   - LC4/LPLC2 (looming) drive strongly and laterally drives the giant fiber DNp01.
   - This holds on `full`, `pruned5`, and the 22.7k-neuron `navcore` training graph, and at dt 0.1 ms as well as 0.5 ms.
4. **Limit found:** against 100 shuffled graphs, pursuit lateralization is wiring-specific at 10–25 Hz (p = 0.01), marginal at 50 Hz, and not specific at 100 Hz. Pursuit encoders should target **≤ 25 Hz**.
5. **Open problem:** odor (ORN) input still ignites a self-sustained Kenyon-cell / mushroom-body state on `pruned5` and `full`. navcore is unaffected, so training isn't blocked; the full-graph showcase is (§6.3). Also found: the dataset labels all Kenyon cells as dopaminergic.

## 1. Simulator now matches the reference model

`flyseek/brain/lif_torch.py`, checked against `philshiu/Drosophila_brain_model/model.py`:

| Parameter | Reference | Ours |
|---|---|---|
| v_rest / v_reset / v_th | −52 / −52 / −45 mV | same |
| τ_m / τ_syn | 20 / 5 ms | same |
| Refractory | 2.2 ms (Brian2 `t − lastspike > t_rfc`) | same semantics (4 steps at dt 0.5 → min ISI 2.5 ms) |
| Synaptic delay | 1.8 ms | ring buffer, round(1.8/dt) steps = **2.0 ms at dt 0.5**, 1.8 ms at dt 0.1 |
| Stimulus | Poisson, weight w_syn × f_poi = 68.75 mV | same |
| Integration | Brian2 at 0.1 ms | exact (v, g) propagator per step |

**Unit tests** (`tests/test_lif.py`, 8 passing):
- analytic single-kick peak
- dt 0.5 vs 0.1 agreement
- exact synaptic delay
- refractory rate cap
- stimulus rate
- event-driven and sparse-matmul propagation give bit-identical spikes
- silencing blocks transmission
- rest stability

**Note:** a 50 Hz stimulus drives the target neuron at ~56 Hz, not 50, because a residual kick can re-trigger it after the refractory period. The reference model has the same property. All reports use **measured** stimulated rates.

## 2. Ignition at the uncalibrated scale

`ignition.py`, `ignition_core.py` → `phase1_ignition*.json`

| 50 ms pulse on… | Real pruned5 | Shuffled pruned5 |
|---|---|---|
| All photoreceptors (20 Hz) | small, dies out | same |
| LC10a‑L (50 Hz) | **self-sustained 11.2 Hz, 11% of neurons** | nothing |
| One DNa02‑L neuron (100 Hz) | **self-sustained 11.2 Hz, 12% of neurons** | nothing |

- **What the ignited core is:** 16,704 neurons at a mean of 111 Hz, many at the 400 Hz ceiling. It sits almost entirely in the central brain (MBONs, antennal-lobe LNs, dopamine neurons); silencing the whole VNC changes nothing.
- **Structure, not overall balance:** real and shuffled graphs have identical excitatory/inhibitory totals (E/I weight ratio 1.63).
- **Neuromodulators are not the main cause:** zeroing dopamine/serotonin/octopamine synapses lowers activity (11.2 → 2.0 Hz) but doesn't remove ignition.

## 3. Calibration

| | FlyWire (reference calibration) | MaleCNS brain |
|---|---|---|
| Synapses | ~50M | 101.1M |
| Neurons | 139,255 | 144,760 |
| Input synapses / neuron | ~359 | 698 |

**Regime sweep** (`regime_sweep.py` → `phase1_regime_sweep.json`, pruned5, one seed):
- **Scale 1.0:** ignition; DNa02 left–right difference ≈ 0 under LC10a drive.
- **Scale 0.75:** LC10a pulse still ignites.
- **Scale 0.514:** no ignition; DNa02 left–right flips with drive side; DNp01 386/222 Hz under looming.
- **Smaller scales** (0.4, 0.3) give larger effects, but were **not** chosen, to avoid tuning the model to the result.
- **Neuromodulator sign** (+1 vs 0) makes little difference once calibrated. Kept at +1 to match the reference; 0 is a sensitivity check.

## 4. Sanity checks with statistics (calibrated)

`sanity_checks.py`: 10 independent replicates per condition, 200 ms warmup, 1 s measurement. Welch t-test between left-drive and right-drive replicates; d = Cohen's d.

### 4.1 Pursuit: LC10a drive → DNa02 (L−R, Hz), left drive vs right drive

| Graph | 10 Hz | 25 Hz | 50 Hz | 100 Hz |
|---|---|---|---|---|
| full | +2.6 / −2.2, p=1e‑6 | +10.5 / −16.5, p=6e‑7 | +7.0 / −24.3, p=0.002 | +19.2 / −9.2, p=5e‑11 |
| pruned5 | +2.1 / −3.6, p=5e‑7 | +8.8 / −16.7, p=3e‑8 | +5.6 / −38.6, p=2e‑5 | +23.8 / −8.1, p=9e‑13 |
| navcore | +2.1 / −3.6, p=5e‑7 | +10.1 / −16.9, p=4e‑8 | +4.6 / −34.2, p=2e‑4 | +20.0 / −5.2, p=2e‑11 |
| pruned5, dt 0.1 ms | — | +9.6 / −18.6, p=0.001 | +5.0 / −40.6, p=8e‑4 | — |

**DNa03** flips with side at 10–50 Hz on every real graph (p ≤ 0.003), and **not** at 100 Hz (p = 0.07–0.34).

### 4.2 Looming: LC4+LPLC2 drive → DNp01 rate vs no input (Hz)

| Graph | 10 Hz | 25 Hz | 50 Hz | 100 Hz |
|---|---|---|---|---|
| full | +71.7 | +212.1 | +293.9 | +264.8 |
| pruned5 | +71.9 | +194.3 | +300.7 | +320.6 |
| navcore | +66.2 | +161.3 | +247.8 | +272.9 |

All p < 1e‑11 vs baseline. The response is lateralized: left drive raises left DNp01 relative to right, and vice versa (p ≤ 4e‑7 at every rate on real graphs).

### 4.3 Ignition (calibrated)

After a 50 ms pulse on DNa02‑L or LC10a‑L, the fraction of neurons still above 1 Hz is **≤ 0.0012** on every graph (vs 0.11–0.12 uncalibrated).

### 4.4 Real vs shuffled wiring

Shuffle = degree-preserving permutation of postsynaptic targets (`shuffle.py`). In- and out-degree, output weights and signs are identical to the real graph.

- **full vs 1 shuffle, pruned5 vs 3 shuffles:** shuffled DNp01 never responds to looming. Shuffled DNa02 sometimes shows a one-sided chance response at ≥ 50 Hz, never a side-specific flip.
- **navcore vs 10 shuffles** (`phase1_sanity_navcore_shuffle_compare.json`). Lateralization index LI = (L−R | left drive) − (L−R | right drive):

| Rate | DNa02 LI real (shuffle max) | DNa03 LI real (shuffle max) | DNp01 LI real (shuffle max) |
|---|---|---|---|
| 10 Hz | +5.7 (+0.6) | +1.1 (+0.1) | +128.5 (+0.0) |
| 25 Hz | +27.0 (+9.7) | +9.8 (+1.8) | +262.9 (+0.0) |
| 50 Hz | +38.8 (+38.4) | +12.2 (+9.0) | +345.4 (+2.5) |
| 100 Hz | +25.2 (**+95.4**) | +0.1 (**+55.3**) | +344.5 (+169.2) |

- At 10–50 Hz the real graph exceeds **all 10** shuffles on every metric (empirical p = 1/11, the minimum possible with 10 shuffles).
- At 50 Hz, DNa02's margin over the best shuffle is thin (+38.8 vs +38.4).
- At 100 Hz, pursuit LI is **inside the shuffle distribution** (p ≈ 0.55–0.64).
- **100-shuffle null:** see §6.

### 4.5 Other descending-neuron candidates (same runs; mean Hz, full graph / navcore)

| Readout | No input | LC10a‑L 50 Hz | LC10a‑L 100 Hz | Loom‑L 50 Hz | Loom‑L 100 Hz | Loom‑R 50 Hz |
|---|---|---|---|---|---|---|
| DNg100 L/R (forward candidate) | 0/0 | 0/0 · 0.6/0.6 | 0/0 · 0/0 | 0/0 | 0/0 | 0/0 |
| DNp09 L/R (forward candidate) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| MDN L/R (backward) | 0/0 | 0/0 | 0/0 | 0/0.1 | 0/0 | 0.3/0 · 1.0/0.1 |
| DNa01 L/R | 0/0 | 0/0 | 0/0 | **0/25.6 · 0/2.3** | **0/75.9 · 0/49.1** | 1.4/0 · 0.7/0 |
| DNg13 L/R | 0/0 | 0/0 · 2.3/3.1 | **15.8/0 · 26.9/0** | 0/0 | 0/0 | 0/0 |
| pIP1 L/R ("P1" candidate) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **0/97.0 · 0/84.3** |

**Findings:**
- **Forward and backward DNs are not reached by these sensory inputs.** DNg100, DNp09 and MDN stay at ≈ 0 Hz, so gameplay can't get forward walking from vision alone. **Phase 2 decision:** base forward speed is an **explicit engineered constant in the motor decoder**, labeled as not brain-generated. Connectome outputs modulate it: DNa02/DNa03/DNg13/DNa01 steer, and DNp01/MDN trigger dash or back-off. Driving DNg100 and reading it back as speed would be circular, like the old self-readout checks, so it isn't done.
- **DNa01 responds contralaterally to looming** (left loom → right DNa01). This is a candidate *turn-away-from-threat* signal for Hiders. It needs a proper left/right statistical test before it's relied on.
- **DNg13 responds ipsilaterally to strong LC10a drive** (100 Hz).
- **pIP1 responds strongly to right-side looming.** That doesn't fit the courtship-arousal P1 identity it was assigned by synonym, which strengthens the "uncertain" flag.

## 5. Performance (active network, `bench.py` → `bench_results.json`)

| Graph | Flies (batch) | ms/step | Fly-seconds simulated per wall second |
|---|---|---|---|
| navcore | 64 | 2.06 | 15.6 |
| navcore | 256 | 6.18 | 20.7 |
| pruned5 | 6 | 1.49 | 2.0 |
| full | 6 | 2.48 | 1.2 |

Removing two GPU→CPU syncs from the stimulus code gave ~1.7× speedup. Per-step Python overhead (~0.9 ms) now dominates small batches; CUDA graphs are a later optimization.

**Implications (estimates):**
- A 5-minute, 6-fly match on `full` takes ~25 min to compute.
- A 90 s × 16-fly training generation on navcore takes ~90 s.

## 6. Final batch: 100-shuffle null, stability, and an odor-driven mushroom-body problem

### 6.1 navcore vs 100 shuffles (`phase1_sanity_navcore100_shuffle_compare.json`)

Empirical p = (1 + #shuffles ≥ real)/101. The minimum possible is 0.010.

| Rate | DNa02 LI: real / shuffle max / p | DNa03 LI: real / max / p | DNp01 LI: real / max / p | DNp01 drive: real / max / p |
|---|---|---|---|---|
| 10 Hz | +5.7 / +2.2 / **0.010** | +1.1 / +0.5 / **0.010** | +128.5 / +0.2 / **0.010** | +75.3 / +0.1 / **0.010** |
| 25 Hz | +27.0 / +23.6 / **0.010** | +9.8 / +7.8 / **0.010** | +262.9 / +5.7 / **0.010** | +178.6 / +1.4 / **0.010** |
| 50 Hz | +38.8 / +108.3 / **0.030** | +12.2 / +31.6 / 0.059 | +345.4 / +32.4 / **0.010** | +282.5 / +8.1 / **0.010** |
| 100 Hz | +25.2 / +206.3 / 0.366 | +0.1 / +198.5 / 0.515 | +344.5 / +169.2 / **0.010** | +313.9 / +114.8 / **0.010** |

**Conclusions:**
- **Looming → giant fiber** is wiring-specific at every rate tested.
- **Pursuit lateralization** is wiring-specific at 10–25 Hz and marginal at 50 Hz: DNa02 p = 0.03, DNa03 p = 0.06, and some shuffles produce larger one-sided responses.
- **Pursuit encoder rates should target ≤ 25 Hz**, tightened from the earlier ≤ 50 Hz.

### 6.2 Stability under sustained input (`phase1_stability.json`, 5 s)

**No condition grows** over the 5 s; late/early ratio ≈ 1.0 everywhere.

| Graph | Population rate at 5 / 20 / 50 Hz input | Non-stimulated neurons active |
|---|---|---|
| navcore | 0.77 / 3.21 / 8.65 Hz (graded; mostly the stimulated neurons) | ≤ 0.2% |
| pruned5 shuffled | 0.11 / 0.44 / 1.20 Hz | ≤ 0.2% |
| pruned5 | 4.8 / 5.1 / 6.0 Hz (**nearly input-independent**) | 6.4% |
| full | 10.4 / 10.7 / 11.6 Hz (**nearly input-independent**) | 8.6% |

### 6.3 Open problem: odor input ignites the mushroom body, even when calibrated

`ignition_calibrated.py`, `kc_test.py` → `phase1_ignition_calibrated.json`, `phase1_kc_test.json`

**What happens:**
- 500 ms of 20 Hz drive on the danger/ping **ORNs** leaves activity running after offset: pruned5 4.7 Hz / 9.2k neurons; full 10.2 Hz / 12.5k neurons.
- **Photoreceptor** drive alone does not.
- The self-sustained set is dominated by **Kenyon cells** (MB), e.g. KCg‑m: 1,342 cells at 247 Hz on full.
- navcore contains no MB and shows **no** after-activity.

**Data-quality finding:** MaleCNS `celltype_predicted_nt` labels **all 4,062 KCs as dopamine**. KCs are cholinergic in the literature. With `modulatory_sign: 1` the sign was still excitatory, so the main results are unaffected. However, the §3 "modulatory_zero" sensitivity check was **confounded**: it also silenced every KC.

**KC→KC synapses:**
- They are 23% (pruned5) and 55% (full) of KC input weight.
- Removing them lowers full-graph after-activity 10.2 → 5.8 Hz but **does not remove it**.
- Other MB loops sustain it. Leading hypothesis: APL feedback inhibition is too weak in a single-compartment LIF, where one APL neuron saturates.

**Impact and plan:**
- **Not a blocker for Phases 2–5.** Training and closed-loop work run on navcore, which is unaffected, and vision-only input doesn't trigger this.
- **It is a blocker for the full-graph showcase** if odor channels (danger meter, pings) are used.
- **Fixes to test before Phase 6**, each reported with its effect:
  1. Override KC NT to acetylcholine (literature ground truth).
  2. Remove KC→KC synapses.
  3. Model APL as graded, non-saturating inhibition.
  4. As a disclosed fallback, drop odor channels from the full-graph showcase.

## 7. Decisions carried forward

- `connectome_weight_scale: 0.514`, `modulatory_sign: 1`, dt 0.5 ms
- **Direct LC10a injection works**, so `bypass_downstream` stays **off**
- Pursuit encoder rates target **≤ 25 Hz** (§6.1); looming ≤ 100 Hz
- Odor channels are used on navcore only until the MB ignition fix (§6.3) is done
- `navcore` (22,686 neurons, 985k edges) is the training graph; it reproduces the full-graph effects
- Base forward speed is an engineered, disclosed decoder constant, not brain output (§4.5). The connectome supplies steering (DNa02/DNa03/DNg13 for pursuit, DNa01 as a candidate turn-away signal for escape) and dash/back-off (DNp01/MDN)

## 8. Limitations to disclose

- **One-sided shuffle control:** the shuffle preserves degree and output weight but not each neuron's incoming weight or E/I input mix.
- **Stimulation is optogenetic-style**, Poisson on whole populations. Retinotopic, naturalistic input comes in Phase 2.
- **Guessed cell types:** DNg100/DNp09 as forward drive and pIP1 as P1 are still unverified. §4.5 shows DNg100/DNp09 get no sensory drive from these inputs, and pIP1's responses don't fit a P1 identity.
- **Asymmetry in the real graph:** right-side drive gives larger DNa02 responses than left (e.g. 50 Hz: +5.6 vs −38.6). Not investigated.
