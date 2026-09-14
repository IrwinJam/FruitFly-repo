# FlySeek Gameplan — from foundation to a real match

*Written 2026-09-13, after the first build session. This file turns what's left in
[STATUS.md](STATUS.md) into an ordered plan. [PROJECT_PLAN.md](PROJECT_PLAN.md) still holds the research and design.*

---

## 1. Where we actually are

| Area | State | Honest note |
|---|---|---|
| Data (MaleCNS v1.0) | ✅ Done | Downloaded, byte-verified, and the graph matches published edge counts |
| Brain simulator | ✅ Runs | GPU LIF. **No synaptic delay yet.** (An earlier note here claimed a refractory countdown bug; tracing it step by step showed it wasn't one.) |
| Benchmark | ✅ Done | pruned5: 0.46× real time with 1 fly. full: 0.048× with 6 flies. **The network was silent during the benchmark** (0 spikes), so these are speeds for a quiet network |
| Cell types | ✅ Resolved | P1 → `pIP1` and forward drive → `DNg100`/`DNp09` are both **unverified guesses** |
| Brain panel viewer | ✅ Looks right | **Fake activity only.** Not connected to the simulator |
| Sanity checks (M3) | ⚠️ **Weaker than claimed** | See §1.1. Treat M3 as not done |
| Skeld map + vents | ✅ Done | Vent positions are approximate |
| Game rules | 🟡 Config only | `config/game.yaml` exists. No rules engine |
| Senses, motors, body | ❌ None | `senses.yaml` describes a design; no code reads it. `motors.yaml` doesn't exist |
| Agents, training, replays | ❌ None | — |
| Viewer map view | ❌ None | — |

### 1.1 Re-review of the M3 sanity checks

| Check | What it really showed |
|---|---|
| steering_readout | Drove DNa02‑L and read DNa02‑L. **Tests the spike counter, not the brain.** |
| backward_readout | Drove MDN and read MDN. **Same problem.** |
| pursuit_pathway | 16 spikes on the left vs 6 on the right (2 neurons per side, 300 ms, 1 seed, no baseline). **Too few spikes to separate from noise.** |
| looming_pathway | 196 spikes on 2 neurons in 200 ms ≈ 490 Hz, right at the refractory ceiling. **Saturated.** The drive was far too strong, or the network ran away. Nobody checked what else fired. |

**Conclusion:** we have only weak evidence that sensory signals reach the steering neurons. This is still the biggest risk in the project, so Phase 1 comes first.

---

## 2. Strategy: build a crude full game first, then train

The plan's original order was: verify → walk → navigate → train roles → finally make a watchable match. The first watchable result would only arrive after days of GPU training.

**Change of approach:** build an end-to-end, watchable match as early as possible, with **untrained brains and hand-tuned gains**. Then improve it in stages. This has three benefits:
- Every piece (senses, motors, rules, replay, viewer) gets exercised against real data early, so integration bugs show up in week 1, not week 4.
- You get something fun to look at quickly, even if the flies wander.
- Training then becomes *improving a working game*, not a gamble that everything fits together at the end.

**Two simplifications to the plan:**
1. **Replay files instead of live streaming.** The brain runs slower than real time even in the best case, so "live" means stuttering. Record each match to a file, then play it back smoothly in the viewer. Live WebSocket mode moves to the stretch list.
2. **Kinematic body only.** NeuroMechFly stays a stretch goal.

---

## 3. Phases

Effort is in **working sessions** (roughly one sitting like today). GPU time is **unattended** run time; most of it can run overnight. Every GPU estimate is a guess until the Phase 1 benchmark on real, active networks replaces it.

### Phase 1 — Make the brain trustworthy · ~1–2 sessions
*Goal: know, with real statistics, whether senses reach motor neurons at realistic drive levels.*

- [ ] **Stability test.** Put sustained, moderate drive (5–20 Hz) on the sensory populations for 5 s. Record the total firing rate over time. Pass: activity settles instead of exploding. If it runs away, add global inhibition, lower `mv_per_synapse`, or use `pruned5`.
- [x] **Match the Shiu reference model**: refractory 2.2 ms (Brian2 semantics), 1.8 ms synaptic delay buffer, exact integration, 68.75 mV stimulus kicks. Covered by unit tests (`tests/test_lif.py`).
- [ ] **Rewrite the sanity checks properly:**
  - [ ] An undriven baseline for every check, with the **same random seed**.
  - [ ] 20 trials per condition with different seeds. Report mean ± spread and a simple test (e.g. a t-test on L−R).
  - [ ] Longer windows (1–2 s) so spike counts reach the hundreds.
  - [ ] Sweep the drive level (e.g. 10, 25, 50, 100 Hz) and find the lowest level that gives a reliable effect. **That level becomes the encoder gain.**
  - [ ] Delete the two self-readout checks, or relabel them as tests of the counter.
  - [ ] Add the reverse-side test: drive LC10a‑R and expect the asymmetry to flip sign.
  - [ ] Check DNg100/DNp09 (forward candidates) and pIP1 (arousal): does sensory drive move them at all?
- [ ] **Build `navcore`**, the small training subgraph: neurons within k hops downstream of the sensory roles *and* upstream of the motor DNs. Re-run the checks on it and confirm the effects survive. Benchmark it with 16 flies.
- [ ] **Benchmark on active networks** (real spiking load) and update `docs/bench_results.json`.
- [ ] **Build the shuffled-connectome generator** (degree-preserving rewiring, sign kept per presynaptic neuron) and run the same checks on it. If shuffled graphs show the same L/R asymmetries, that's a red flag to investigate now, not after training.

**Done when:** a table with statistics shows which pathways work at which drive levels on `full`, `pruned5` and `navcore`, and there's a go/no-go on direct injection vs `bypass_downstream` for each pathway.

**If pursuit doesn't carry a signal:** use `bypass_downstream` (inject at AOTU019/025). If even that fails, inject at DNa03. This lowers the "the connectome did it" factor, but the game still works. The UI already has a flag to disclose it.

### Phase 2 — One fly, one closed loop · ~2 sessions
*Goal: a single brain drives a body around an empty arena using its own vision.*

- [ ] `flyseek/motors/body_kinematic.py`: position, heading, speed, collisions against a grid (sliding along walls).
- [ ] `config/motors.yaml` + `flyseek/motors/decoders.py`: DN rates (EMA smoothed) → forward/turn/dash, with deadband and clamping (formula in PROJECT_PLAN §4.6).
- [ ] `flyseek/senses/vision.py`: ray-cast panorama → per-ray brightness/color → photoreceptors, plus a direct target-feature channel into LC10a, using the Phase 1 gains.
- [ ] `flyseek/agents/fly_agent.py`: one tick = sense → 40 brain steps → decode → move.
- [ ] Arena test: a bright target on the left or right of the fly.
  - Pass: the fly turns toward it more often than a zero-gain control.
  - Hand-tune the gains. No training yet.
- [ ] A first **replay recorder** that saves positions plus spikes to disk.

**Done when:** a recorded run shows the fly turning toward the target in ≥ 70% of 20 trials, compared with ~50% for the control.

### Phase 3 — First watchable match (vertical slice) · ~2–3 sessions
*Goal: 1 Seeker + 3 Hiders play Hide n Seek on The Skeld, recorded and replayed in the viewer with real spikes.*

- [ ] `flyseek/world/rules.py`: hide phase → seek → Final Hide, kill radius, vents (limited uses and time), pings, win check.
- [ ] Scripted agents (`scripted_seeker.py`, `scripted_hider.py`) as opponents and baselines.
- [ ] Senses for the game: looming for Hiders, danger → aversive odor, pings → attractive odor L/R, wall touch.
- [ ] **Replay format.** Per tick: all agent states, plus spike indices per fly (delta-encoded, compressed). Measure the file size per minute of match. If a 5-minute, 4-fly match is over ~500 MB, fall back to per-neuron heat quantized to 8 bits at 10 fps.
- [ ] **Viewer:**
  - [ ] A Skeld map panel in the center: walls, players, vision radius, vents, timer, danger ring.
  - [ ] Replay loading and playback controls (play/pause, speed, scrub).
  - [ ] **Remove the synthetic activity** and drive the brain panels from the replay's real spikes.
  - [ ] Change the disclaimer from "DEMO" to the plan's model disclaimer.
- [ ] Run matches of untrained brains vs scripted opponents, and brains vs brains.

**Done when:** you can load a replay and watch a full match, with brain panels lighting up from real simulated activity and the game ending on a real win condition.

> 🎉 **This is the point where the project "exists."** Everything after this makes the flies play better.

### Phase 4 — Training infrastructure + walking/navigation · ~2 sessions + ~1–2 nights of GPU
- [ ] `flyseek/train/es.py`: CMA-ES over the adapter parameters (encoder gains, biases, decoder weights). The population runs as the GPU batch.
- [ ] Rewards, curricula, checkpoints, and evaluation against baselines (random walk, zero-gain, scripted).
- [ ] **W1 (arena phototaxis)** on navcore. Rough estimate: 1–2 h of GPU.
- [ ] **W2 (Skeld navigation curriculum)**: Cafeteria → room + corridor → full map. Rough estimate: 7–15 h of GPU.
- [ ] Check transfer: the best adapter from navcore evaluated on `pruned5`.
- [ ] **Portability for free compute:** configurable data paths (no hard-coded `C:\`), headless CLI, checkpoint/resume, and a Kaggle notebook that pulls the repo plus a private Kaggle dataset.
- [ ] Run W2 on the shuffled connectome too, with the same budget, for the §6.2 control.

**Done when:** a trained fly reaches a random room within 90 s in ≥ 60% of trials and beats a random walk.

### Phase 5 — Role training · ~2 sessions + ~2–4 nights of GPU
- [ ] **R1:** train the Seeker adapter against scripted Hiders, and the Hider adapter against a scripted Seeker. Rough estimate: 15–30 h of GPU per role. Use shorter episodes (90 s) first to halve that.
- [ ] **R2:** self-play league with a hall of fame of past adapters. Open-ended; stop when win rates plateau.
- [ ] Win-rate curves and a comparison table against the baselines.
- [ ] **Controls for publication:** the shuffled connectome trained with the same budget, adapter-on-noise, and circuit ablations (LC10a silenced for the Seeker; LC4/LPLC2/DNp01 silenced for Hiders) evaluated on a held-out seed set.

**Done when:** the trained Seeker beats scripted Hiders more often than a random-walk Seeker does, and trained Hiders survive longer than a random baseline.

### Phase 6 — Showcase · ~1–2 sessions + a few hours of GPU
- [ ] **6a — 1 Seeker + 3 Hiders** on the `full` graph with trained role adapters. Recorded match: ~20–40 min of compute per 5-minute match on the 2060 Super (estimate).
- [ ] **6b — 1 Seeker + 5 Hiders**, reusing the same role adapters (no retraining). Evaluate briefly first; fine-tune the Hider adapter only if 5 Hiders crowd or collapse. ~30–60 min of compute per match.
- [ ] Methods table and a results write-up (§6.2) alongside the video.
- [ ] Viewer polish: IMG_0897 bloom, named-circuit labels ("LC10a — target spotted"), a highlight when the fly is selected, title card.
- [ ] Capture a 60–90 s video.

### Stretch (any order after Phase 3)
- Live WebSocket mode on navcore
- Mushroom-body dopamine learning
- Central-complex goal memory
- NeuroMechFly 3D inset for one hero fly
- Flashlight mode

---

## 4. Timeline at a glance

| Phase | Sessions | Unattended GPU | Result you can see |
|---|---|---|---|
| 1 Brain trust | 1–2 | minutes | Stats table: which pathways work |
| 2 Closed loop | 2 | minutes | Recorded fly turning toward a target |
| 3 **First match** | 2–3 | minutes per match | **Watchable replay with real spikes** |
| 4 Walk/navigate | 2 | 1–2 nights | Flies that actually get around The Skeld |
| 5 Roles | 2 | 2–4 nights | Seekers that hunt, hiders that flee |
| 6 Showcase | 1–2 | a few hours | 3-Hider video, then 5-Hider video, plus write-up |
| **Total** | **~10–13 sessions** | **~4–7 nights** | |

No paid compute (§6.1): the nights run on the local 2060 Super, offloaded to Kaggle's free ~30 h/week where they fit in 12 h chunks (no HPCC access). The publication controls in §6.2 (shuffled connectome, ablations) add roughly **+50–100% training compute** on top of these numbers.

---

## 5. Top risks, reordered by what we know now

| # | Risk | Status | Plan |
|---|---|---|---|
| 1 | Sensory signals don't reach steering at realistic drive | **Unresolved** (M3 evidence was weak) | Phase 1 statistics, then fall back to bypass injection |
| 2 | Recurrent runaway on the full graph under sustained input | **Untested** (looming check saturated) | Phase 1 stability test |
| 3 | Training too slow on the 2060 Super | Likely | navcore, shorter episodes, overnight runs, optional cloud |
| 4 | Guessed cell-type identities are wrong (pIP1 as P1, DNg100 as forward) | Open | Phase 1 checks whether they respond at all. Swap candidates if not |
| 5 | Replay files too large | Unmeasured | Measure in Phase 3, with a quantized-heat fallback |
| 6 | Two copies of the repo drift apart | Resolved | `FruitFly/` is canonical, backed up to GitHub |
| 7 | Free compute is tight (local GPU + Kaggle's 30 h/week, 12 h sessions; no HPCC) | Likely | Resumable training; navcore for all training; shorter episodes; keep the `full` graph for evaluation/showcase only |
| 8 | The shuffled connectome performs just as well | Unknown, **and a legitimate result** | Test early in Phase 1; report it either way |

---

## 6. Decisions (resolved 2026-09-13)

| # | Decision | What it means for the plan |
|---|---|---|
| 1 | **`FruitFly/` is the canonical folder**, backed up to a **private GitHub repo** | All work happens in `FruitFly/`. `FruitFly-repo/` is redundant and can be deleted. The data (~1.4 GB) isn't pushed; [README.md](README.md) documents how to rebuild it |
| 2 | **No paid compute** | See §6.1. Azure's free and student tiers don't allow GPU VMs, so the plan uses the local 2060 Super plus Kaggle's free tier (the user has no TTU HPCC access) |
| 3 | **Showcase with 3 Hiders first, then 5** | Adapters are trained **per role, not per fly**, so the 5-Hider match reuses the same trained Hider adapter. Scaling up costs compute, not retraining. A short fine-tune may be needed if 5 Hiders behave differently (e.g. crowding) |
| 4 | **Aim for publishable results** | See §6.2. This adds controls and ablations to Phases 1, 4 and 5, and keeps bypasses **visibly labeled** |
| — | "Feel" values (speed, vision radius, kill distance) | Picked by eye in Phase 3, then **frozen and reported** so results can be reproduced |

### 6.1 Free compute plan

| Option | What you get | Verdict |
|---|---|---|
| Azure free trial / Azure for Students | GPU VM series (NC etc.) not available on trial or student subscriptions; student quota is 3 vCPUs | ❌ Doesn't work for GPU |
| **Local RTX 2060 Super** | 8 GB, always available, overnight runs | ✅ Default for Phases 1–3 and short training runs |
| **Kaggle Notebooks** | ~30 GPU-hours/week free (P100 16 GB or 2× T4), 12 h max per session; runs continue after closing the tab | ✅ Offload for Phase 4–5 training. Needs checkpoint/resume (a 12 h cap per run) and the data uploaded as a private Kaggle dataset (~1.4 GB) |
| TTU HPCC | Free only with faculty sponsorship | ❌ Not available to the user |

**Implications for the code:**
- Training scripts must be **headless and resumable**, checkpointing every N generations.
- Data paths must be **configurable**. They're hard-coded to `C:\flyseek-data` today, which breaks on Kaggle (Linux).
- Both changes are added to Phase 4.

### 6.2 What "publishable" adds

A result like "the connectome makes the fly hunt" only holds up against the right comparisons. Each of these goes into the phases below:

1. **Shuffled-connectome control.** A degree-preserving rewired graph, trained with the same adapter budget. If it plays just as well, the wiring isn't doing the work. **This is the single most important control.**
2. **Baselines.** Zero-gain brain, random walk, scripted agents, and an adapter-only model (the same readout on noise instead of a brain).
3. **Circuit ablations.** Silence LC10a → does pursuit drop? Silence LC4/LPLC2/DNp01 → does escape drop?
4. **Statistics.** Multiple seeds, confidence intervals, fixed evaluation sets held out from training.
5. **Disclosure.** Every bypass injection, guessed cell-type identity (pIP1 as P1, DNg100 as forward), and hand-picked constant is listed in a methods table.
6. **Reproducibility.** Code, configs, checkpoints and the exact data version are pinned. This is a small preregistration: write down success criteria *before* running each evaluation.
7. **Attribution and IP.** MaleCNS is CC-BY 4.0 with a citation. Use no Among Us art; use own sprites in any published video. Remove the reference screenshots from any public release.

---

## 7. Next session, concretely

Start Phase 1:
1. Fix the refractory countdown.
2. Write `flyseek/brain/stability.py` and run it on pruned5 and full.
3. Rewrite `sanity_checks.py` with baselines, seeds, longer windows, drive sweeps, and the reverse-side test.
4. Build navcore and benchmark it under real load.
