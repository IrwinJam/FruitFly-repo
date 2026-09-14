# FlySeek Gameplan — from foundation to a real match

*Written 2026-09-13, after the first build session. This file turns what's left in
[STATUS.md](STATUS.md) into an ordered plan. [PROJECT_PLAN.md](PROJECT_PLAN.md) still holds the research and design.*

---

## 1. Where we actually are

| Area | State | Honest note |
|---|---|---|
| Data (MaleCNS v1.0) | ✅ Done | Downloaded, byte-verified, and the graph matches published edge counts |
| Brain simulator | ✅ Runs | GPU LIF. **No synaptic delay yet.** The effective refractory period is 1.5 ms, not 2 ms (it counts down in the same step it's set) |
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
- [ ] **Fix the refractory countdown** (1.5 ms → the configured 2 ms). Decide whether synaptic delay matters: run the checks with and without a fixed 2-step delay buffer.
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

**Done when:** a trained fly reaches a random room within 90 s in ≥ 60% of trials and beats a random walk.

### Phase 5 — Role training · ~2 sessions + ~2–4 nights of GPU
- [ ] **R1:** train the Seeker adapter against scripted Hiders, and the Hider adapter against a scripted Seeker. Rough estimate: 15–30 h of GPU per role. Use shorter episodes (90 s) first to halve that.
- [ ] **R2:** self-play league with a hall of fame of past adapters. Open-ended; stop when win rates plateau.
- [ ] Win-rate curves and a comparison table against the baselines.

**Done when:** the trained Seeker beats scripted Hiders more often than a random-walk Seeker does, and trained Hiders survive longer than a random baseline.

### Phase 6 — Showcase · ~1–2 sessions + ~1 h of GPU
- [ ] 1 Seeker + 5 Hiders on the `full` graph with trained adapters. Recorded match: ~30–60 min of compute per 5-minute match.
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
| 6 Showcase | 1–2 | ~1 h | Video |
| **Total** | **~10–13 sessions** | **~4–7 nights** | |

A cloud GPU (e.g. an A100, roughly 10× faster than the 2060 Super) would turn most of those nights into hours.

---

## 5. Top risks, reordered by what we know now

| # | Risk | Status | Plan |
|---|---|---|---|
| 1 | Sensory signals don't reach steering at realistic drive | **Unresolved** (M3 evidence was weak) | Phase 1 statistics, then fall back to bypass injection |
| 2 | Recurrent runaway on the full graph under sustained input | **Untested** (looming check saturated) | Phase 1 stability test |
| 3 | Training too slow on the 2060 Super | Likely | navcore, shorter episodes, overnight runs, optional cloud |
| 4 | Guessed cell-type identities are wrong (pIP1 as P1, DNg100 as forward) | Open | Phase 1 checks whether they respond at all. Swap candidates if not |
| 5 | Replay files too large | Unmeasured | Measure in Phase 3, with a quantized-heat fallback |
| 6 | Two copies of the repo drift apart | **Happening already** | Pick one canonical folder (§6) |

---

## 6. Decisions needed from you

1. **Canonical folder.** `FruitFly/` (has the venv and data junction) or `FruitFly-repo/` (clean clone)? Working in both will cause drift. Suggestion: keep working in `FruitFly/`, and either delete `FruitFly-repo/` or push both to a **private GitHub repo** as the real backup.
2. **Cloud GPU for Phases 4–5?** Yes/no, plus a rough budget. Without one, plan on overnight runs.
3. **Hiders in the showcase:** 5 (the original plan) or 3 (about 40% less compute and bigger brain panels)?
4. **"Feel" values nobody publishes:** movement speed, vision radius, kill distance. Suggested approach: pick them by eye in Phase 3 so matches look like Among Us.
5. **How honest the UI should be about bypasses.** If Phase 1 forces downstream injection, keep a visible "bypass" label (recommended) or hide it?

---

## 7. Next session, concretely

Start Phase 1:
1. Fix the refractory countdown.
2. Write `flyseek/brain/stability.py` and run it on pruned5 and full.
3. Rewrite `sanity_checks.py` with baselines, seeds, longer windows, drive sweeps, and the reverse-side test.
4. Build navcore and benchmark it under real load.
