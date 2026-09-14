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

### Phase 1 — Make the brain trustworthy · ✅ DONE 2026-09-13 (one open item carried forward)
*Goal: know, with real statistics, whether senses reach motor neurons at realistic drive levels.* **Full write-up: [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md).**

- [x] **Stability test** (5 s sustained drive). No condition grows. Found the uncalibrated ignition problem and fixed it with `connectome_weight_scale: 0.514`, derived from dataset statistics.
- [x] **Match the Shiu reference model**: refractory 2.2 ms (Brian2 semantics), 1.8 ms synaptic delay buffer, exact integration, 68.75 mV stimulus kicks. Covered by unit tests (`tests/test_lif.py`).
- [x] **Rewrite the sanity checks properly:**
  - [x] Undriven baseline and matched conditions in the same batch.
  - [x] 10 replicates per condition with independent input, 95% CIs, Welch t-test, Cohen's d.
  - [x] 1 s measurement windows.
  - [x] Drive sweep 10/25/50/100 Hz. **Pursuit encoder ≤ 25 Hz** (wiring-specific vs 100 shuffles only up to 25 Hz).
  - [x] Self-readout checks removed.
  - [x] Reverse-side test: the asymmetry flips sign for DNa02, DNa03, DNp01.
  - [x] Forward candidates DNg100/DNp09 and MDN get **no** sensory drive, so base forward speed becomes a disclosed decoder constant. pIP1's responses don't fit a P1 identity.
- [x] **Build `navcore`** (strength-thresholded: 22,686 neurons, 985k edges). Reproduces the full-graph effects; 15.6 fly-s/s at 64 flies.
- [x] **Benchmark on active networks**, plus sync removal (~1.7× faster).
- [x] **Shuffled-connectome controls**: 100 navcore shuffles, 3 pruned5, 1 full. Looming→DNp01 is wiring-specific at all rates; pursuit at 10–25 Hz (p = 0.01).
- [x] **dt check**: 0.1 ms and 0.5 ms agree within noise.
- [ ] **Carried forward, blocking Phase 6 only:** odor (ORN) input ignites a self-sustained Kenyon-cell/mushroom-body state on `pruned5`/`full` (not navcore). Also, the dataset labels all KCs as dopamine. Fixes to test are in PHASE1_REPORT §6.3.

**Done when:** a table with statistics shows which pathways work at which drive levels on `full`, `pruned5` and `navcore`, and there's a go/no-go on direct injection vs `bypass_downstream` for each pathway.

**If pursuit doesn't carry a signal:** use `bypass_downstream` (inject at AOTU019/025). If even that fails, inject at DNa03. This lowers the "the connectome did it" factor, but the game still works. The UI already has a flag to disclose it.

### Phase 2 — One fly, one closed loop · ✅ DONE 2026-09-13
*Goal: a single brain drives a body around an empty arena using its own vision.* **Write-up: [docs/PHASE2_REPORT.md](docs/PHASE2_REPORT.md).**

- [x] `flyseek/motors/body_kinematic.py` (+ `flyseek/world/grid.py`): unicycle body with wall sliding.
- [x] `config/motors.yaml` + `flyseek/motors/decoders.py`: EMA-smoothed DN rates → turn/dash/back; forward speed is an engineered constant (disclosed).
- [x] `flyseek/senses/vision.py`: ray casting, side-level LC10a (≤ 25 Hz) and LC4/LPLC2 looming (≤ 100 Hz), photoreceptor brightness. **Not retinotopic yet.**
- [x] DNa01 contralateral looming tested vs 100 shuffles: **not confirmed** (p ≈ 0.05 only at ≥ 50 Hz), so its decoder weight starts at 0 and is left to training.
- [x] `flyseek/agents/fly_agent.py`: batched flies, sense → 40 brain steps → decode → move.
- [x] Arena pursuit: **88%** turn toward the target (blind 0%, 10 shuffles 0–23%).
- [x] **Added:** arena escape. **100%** giant-fiber dash before contact (blind 0%, all 10 shuffles 0%).
- [x] Replay recorder: 177 KB for 40 flies × 2 s of navcore spikes.

**Done when:** a recorded run shows the fly turning toward the target in ≥ 70% of 20 trials, compared with ~50% for the control. **Met: 88% vs 0% blind.**

**Carried into Phase 3/4:**
- Turn gain is gentle (flies don't reach the target in 2 s).
- Escape triggers too early (~1.6 s before contact); the looming gain and dash threshold need calibration for gameplay.
- Retinotopic vision.

### Phase 3 — First watchable match (vertical slice) · ✅ core done 2026-09-14
*Goal: 1 Seeker + 3 Hiders play Hide n Seek on The Skeld, recorded and replayed in the viewer with real spikes.*

- [x] `flyseek/world/rules.py`: hide → seek → Final Hide, kill radius, auto-vent rule (disclosed), pings, danger meter, win check. 4 unit tests.
- [x] Scripted agents (`flyseek/agents/scripted.py`) with geodesic pathfinding (`flyseek/world/pathing.py`).
- [x] Game senses: Seeker sees Hiders (LC10a), Hiders see the Seeker loom (LC4/LPLC2), danger → aversive ORNs, pings → attractive ORNs L/R. Odor only on navcore. Caught flies stop sensing. **Wall touch not modeled**; a disclosed wall-bump turn reflex is used instead.
- [x] Replay format + exporter: ~0.4–1.4 MB per 90 s, 4-fly match. Far below the 500 MB fallback.
- [x] Viewer: Skeld map panel (walls, vents, agents, vision rings, kills, pings, phase clock), playback controls, brain panels driven by real recorded spikes, model disclaimer. Fixed a WebGL viewport-flip bug and a double pixel-ratio bug.
- [x] Matches recorded (short preset, seed 0, navcore):
  - `match01_all_brains`: Hiders win, 2 of 3 caught
  - `match02_brain_seeker_vs_scripted`: Hiders win, 0 caught
  - `match03_scripted_seeker_vs_brain_hiders`: Seeker wins in 14 s

**Done when:** you can load a replay and watch a full match, with brain panels lighting up from real simulated activity and the game ending on a real win condition. **Met.**

**What the first matches showed:** untrained brain flies bounce around Cafeteria ("Pac-Man"). Brains only turn when they see another fly, and the wall reflex makes the rest a random walk. That's the Phase 4 problem.

> 🎉 **This is the point where the project "exists."** Everything after this makes the flies play better.

### Phase 4 — Navigation with the central complex · revised 2026-09-14 · ~3–4 sessions + ~2 nights of GPU
*Goal: brain-driven flies leave Cafeteria and move around The Skeld with purpose, using the fly's own compass → goal → steering circuit (EPG → FC2 → PFL3 → steering DNs).* Biology: EPG neurons carry a heading bump, FC2 neurons carry a goal direction, and PFL3 neurons compare them to produce a left/right steering signal (Mussells Pires et al. 2024; Westeinde et al. 2024).

**What is connectome vs engineered in this design (disclosed):**

| Piece | Source |
|---|---|
| Heading → EPG bump | Engineered **idealized compass**: the body's true heading injected as an EPG activity bump (real flies use landmarks + self-motion) |
| Goal → FC2 bump | Engineered **goal policy** chooses a direction (e.g. toward unexplored space, a hiding spot, or a hider), injected as an FC2 bump. Its weights are what training learns |
| Heading vs goal → steering | **Connectome**: EPG/FC2 → PFL3 → LAL → DNa02/DNa03 and others |
| Turn / dash / speed decoding | Existing motor decoder (forward speed stays an engineered constant) |

**4.0 — Baseline and spawn-preset experiment** (runs in background while 4.1–4.3 are built)
- [x] Add a `spread` spawn preset: agents start in random rooms instead of all in Cafeteria. Default stays Cafeteria (the real Among Us rule).
- [x] Add coverage metrics to every match: rooms visited per agent, fraction of the map within 1 unit of an agent's path, time to first sighting.
- [x] Run the untrained baseline matrix: {default, spread} × {all brains, brain Seeker vs scripted, scripted vs brain Hiders} × 3 seeds = 18 matches (~1 h GPU). This is the "before" for the trained "after".

**4.1 — Map the compass circuit onto real neurons**
- [x] From MaleCNS annotations (`instance` / glomerulus / column labels), give every EPG, FC2 (A/B/C) and PFL3 neuron an angular position (protocerebral-bridge glomerulus or fan-shaped-body column → angle).
- [x] (navcore already contains the full pathway, so no v2 rebuild was needed) Check that the intermediate neurons (Δ7, P-EN, LAL, PFL2) exist, and build **navcore-v2** including the full EPG/FC2 → PFL3 → DN pathway. Rebuild its shuffles.

**4.2 — Test the circuit before using it (Phase 1 style)**
- [x] Inject an EPG bump at heading h and an FC2 bump at goal g, sweeping g − h over 360°. Expect PFL3 left−right and DNa02/DNa03 left−right to follow the sign of sin(g − h), i.e. turn toward the goal.
- [x] (10 shuffles; passed: DNa02 L−R 22.5 Hz, R² 0.35 vs shuffles ≤ 1.9 Hz / R² ≤ 0.06) 10 replicates per offset, vs 100 shuffled navcore-v2 graphs, plus an FC2-silenced control.
- [x] **Go/no-go (GO):** if steering doesn't follow the goal offset in the real wiring and not in shuffles, report it and fall back to injecting the goal signal at PFL3 (disclosed), or to the target-channel adapter.

**4.3 — Closed-loop goal navigation**
- [x] Compass + goal channels in `FlyPopulation`.
- [x] (real 75% vs no-goal 15%, PFL3 silenced 15%, shuffles 5–38%) Arena test: reach a goal that is **out of sight** (goal direction given, no target visible). Measure success vs blind-goal, shuffled wiring and CX-silenced.
- [ ] Skeld test: follow a scripted route of room waypoints via the goal channel alone.

**4.4 — Training infrastructure**
- [x] `flyseek/train/es.py`: CMA-ES over adapter parameters (EPG/FC2 injection gains, goal-policy weights, decoder weights). Population = GPU batch.
- [x] (replaced by the map-based route planner, `flyseek/agents/route_policy.py`) Goal policy features: free distance per direction (rays), visit novelty map, room-exit directions, seen-fly bearing.
- [ ] Rewards and curriculum (W2: Cafeteria → room + corridor → full map), checkpoint/resume, headless CLI, Kaggle notebook, configurable data paths (done).

**4.5 — Training runs and controls**
- [x] W2 exploration training on navcore-v2 (overnight).
- [x] Same budget on shuffled navcore-v2, and with the CX silenced (the fly then relies on the goal policy + decoder alone).
- [x] (`docs/phase4_trained_matrix.json`: brain-fly rooms per match 3.67 → 5.08 Cafeteria spawn, p=0.29; 4.04 → 6.67 spread spawn, p=0.004) Re-run the 4.0 match matrix with trained adapters and report before/after coverage and outcomes.

**Done when:**
- Trained brain flies visit **≥ 5 of the 14 rooms** on average in a 90 s match starting in Cafeteria.
- They beat untrained, shuffled-wiring and CX-silenced flies.
- 4.2 shows the circuit steers toward goals in the real wiring but not in shuffles.

**Held-out results (40 unseen 45 s episodes from Cafeteria; `docs/phase4_eval_real.json`, `docs/phase4_eval_controls.json`):**

| Condition | Rooms (95% CI) | Wall contact (s of 45) |
|---|---|---|
| Real wiring, untrained | 3.40 [2.91, 3.90] | 21.0 |
| **Real wiring, trained** | **5.25 [4.72, 5.78]** | 21.7 |
| Shuffled wiring, untrained | 1.93 [1.48, 2.37] | 41.4 |
| Shuffled wiring, trained (same budget) | 2.38 [1.90, 2.85] | 39.7 |
| PFL3 silenced, untrained | 1.38 [1.14, 1.61] | 32.2 |
| PFL3 silenced, trained (same budget) | 2.40 [1.93, 2.87] | 36.1 |
| Real-trained adapter, PFL3 silenced at test | 1.73 [1.43, 2.02] | 27.8 |

Trained real vs every control: −2.85 to −3.88 rooms for the control, paired t p ≤ 3e-9. Room criterion (≥ 5) is met in half the match length; wall contact is not yet reduced.

**Not fixed by Phase 4:** hunting and hiding skill (Phase 5), escape timing calibration, retinotopic vision.

### Phase 5 — Role training · ~2 sessions + ~2–4 nights of GPU

**Design (revised 2026-09-14, after Phase 4).** Same split as Phase 4: an engineered, disclosed **role policy** decides *where* to go using only game-legal information, the frozen connectome decides *how* to steer (EPG/FC2 → PFL3 → DNs, plus the sensory pathways), and CMA-ES learns the policy weights, channel gains and decoder. Each role starts from its graph's own Phase 4 explorer (`explore_navcore`, or `explore_shuf0` for the shuffled control).

| Role | Information the policy may use | Trainable |
|---|---|---|
| Seeker | own position, hiders currently in sight (≤ 6 units, line of sight), Final Hide pings | LC10a target gain, chase lookahead, last-seen memory, ping following, explore weights, decoder |
| Hider | own position, the seeker when in sight, the danger meter, vent locations (map knowledge) | LC4/LPLC2 loom gain, danger (odor) gain, hiding-spot weights (far from seeker / concealed / near vent / travel cost), camp time, camp speed, flee threshold, decoder |

Camping (a hider standing still) scales the engineered forward speed; this is disclosed, like the scripted hider's pauses. Fitness: Seeker = Σ caught × (1 + time left / round) / hiders; Hider = mean survival fraction + 0.5 × survived. Episodes use the short preset (90 s, 5 s freeze); matches run in parallel (Seeker: 48 matches per generation, 1 brain vs 3 scripted Hiders; Hider: 24 matches, 3 brain Hiders vs 1 scripted Seeker). Measured cost ≈ 8–10 min per generation, 25 generations per run.

- [ ] Batched role environment (`flyseek/train/role_env.py`) and role policies (`flyseek/agents/role_policy.py`), with scripted and random-walk controllers for baselines.
- [ ] **R1:** train `seeker_navcore` and `hider_navcore` against scripted opponents (night 1).
- [ ] Controls with the same budget: `seeker_shuf0`, `hider_shuf0` (night 2).
- [ ] Held-out evaluation (40 matches, unseen seeds): trained vs Phase 4 explorer (no role training) vs random walk vs scripted; ablations LC10a silenced and PFL3 silenced (Seeker), LC4+LPLC2 silenced, DNp01 silenced and PFL3 silenced (Hiders).
- [ ] Win-rate/fitness curves and a comparison table; showcase replays.
- [ ] **R2 (optional):** trained Seeker vs trained Hiders, alternating fine-tunes with a hall of fame. Only if R1 works.

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
