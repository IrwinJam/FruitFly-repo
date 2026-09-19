# Phase 5 report: Seeker and Hider roles

*2026-09-14 to 2026-09-19. Held-out evaluations use match seeds 90000+ and exploration seeds 50000+, which were never used in training. The raw numbers are in `docs/phase5_eval_*.json`.*

## 1. Setup

The split is the same as in Phase 4. An engineered, disclosed **role policy** decides *where* a fly goes, using only information a player of that role has in Among Us. The frozen connectome decides *how* it steers there: world goal direction → FC2 goal bump, idealised EPG compass bump → PFL3 → steering DNs. Sensory pathways also drive the brain: LC10a for seen hiders, LC4/LPLC2 for a looming seeker, and aversive-odor ORNs for the danger meter. CMA-ES tunes the policy weights, channel gains and motor decoder. **No connectome weight is ever changed.**

| Role | What the policy may use | Opponent in training |
|---|---|---|
| Seeker | Hiders in sight (≤ 6 units, line of sight), Final Hide pings | 3 scripted hiders |
| Hider | The seeker when in sight, the danger meter, vent locations | 1 scripted seeker |

Other settings:
- **Game:** short preset (90 s round, 5 s seeker freeze, 30 s Final Hide), Cafeteria spawn, real rules engine.
- **Fitness, seeker:** Σ caught × (1 + time left / round) / hiders, range 0–2.
- **Fitness, hider:** survival fraction + 0.5 × survived.
- **Engineered and disclosed:** forward speed, hider camping (standing still scales forward speed), auto-vent, the route planner.

## 2. First round: role training on the Phase 4 walker (max turn rate 3 rad/s)

| Held-out, 40 matches | Result |
|---|---|
| Seeker, trained vs walker | 0.77 vs 0.66, p = 0.22 (4 matches per candidate); 0.79 vs 0.66, p = 0.20 (8 matches per candidate) |
| Hider, best trained vs walker | survival 31.3 s vs 21.5 s, p = 0.015; the population mean did not improve |
| Scripted seeker | 1.49 (35/40 wins); brain seeker 3/40 wins |

The gains were marginal. Diagnostics pointed at locomotion rather than strategy:

| Finding | Number |
|---|---|
| Wall contact vs rooms visited | Spearman ρ = −0.39, p = 9e-6 |
| Wall time spent in contacts longer than 2 s | 74% (longest 38 s) |
| Body turn rate | 0 half the time, saturated 27% of ticks |
| Steering error, open arena vs The Skeld | 26–32° vs 54–70° |
| Goal direction blocked by a wall | only 15% of near-wall ticks |

## 3. Locomotion fixes (40 held-out 45 s exploration episodes each)

| Change | Rooms | Wall contact | vs baseline |
|---|---|---|---|
| Baseline (max turn rate 3 rad/s) | 5.25 | 21.7 s | — |
| Wall-aware goal (probe own rays, steer to nearest clear direction) | 4.83–5.15 | 17.4–23.0 s | −0.10 to −0.42, n.s. — **rejected** |
| Cut thrust on wall contact (factor 0.5 / 0.2 / 0) | 4.50 / 3.88 / 4.65 | 23.4 / 26.9 / 12.2 s | −0.60 to −1.38 — **rejected** |
| Stronger compass drive (r_max 250 Hz) | — | — | open-arena error 51° vs 32° — **rejected** |
| Longer planner lookahead (3 units) | 5.10 | 25.9 s | −0.15, n.s. |
| **Max turn rate 4.5 / 6 / 8 / 10 / 12 rad/s** | 5.95 / **6.30** / 4.83 / 4.00 / 3.60 | 19.6 / **14.7** / 18.6 / 19.2 / 15.5 s | **6 rad/s: +1.05, p = 0.004** |

- **Why 6 rad/s:** 3 rad/s was well below a real walking fly, which saccades at about 8–17 rad/s. Above 6 rad/s the model fly overshoots its goal direction. The default is now 6 rad/s and it is trainable (`config/motors.yaml`).
- **Wall contact:** reducing it directly (cutting thrust) did not help. The ρ = −0.39 correlation was therefore **not causal**. Wall contact is a symptom of slow turning.

## 4. Second round: everything retrained on fixed locomotion

### 4.1 Exploration (45 s, rooms visited)

| Condition | Rooms (95% CI) | Wall s | vs untrained |
|---|---|---|---|
| Untrained | 5.35 [4.83, 5.87] | 12.8 | — |
| Retrained (`explore_navcore_v2`) | 6.33 [5.68, 6.97] | 14.0 | +0.97, p = 0.006 |
| Phase 4 adapter + new turn rate | 6.45 [5.84, 7.06] | 14.8 | +1.10, p = 0.003 |
| Shuffled wiring, trained (same budget) | 2.40 [1.92, 2.88] | 40.0 | −2.95, p = 2e-12 |
| PFL3 silenced | 1.50 [1.24, 1.76] | 21.1 | −3.85, p = 4e-17 |

Retraining added nothing beyond the turn-rate fix.

### 4.2 Seeker (`seeker_v3`: 8 candidates × 8 matches × 20 generations)

| Condition | Fitness (95% CI) | Wins /40 | vs walker |
|---|---|---|---|
| Walker, no role training | 1.064 [0.90, 1.23] | 15 | — |
| Role-trained, best / mean | 1.014 / 1.040 (0.970 on re-run) | 13 / 14 (9) | −0.02 to −0.09, n.s. |
| LC10a silenced | 0.943 | 11 | −0.12, p = 0.22 |
| **PFL3 silenced** | **0.567** | **2** | **−0.50, p = 7e-6** |
| **Shuffled wiring, trained (same budget)** | **0.016** | **0** | **−1.05, p = 6e-15** |
| Random walk | 0.052 | 0 | −1.01, p = 7e-15 |
| Scripted seeker | 1.486 | 35 | +0.42, p = 2e-6 |

### 4.3 Hiders (`hider_v2`: 12 candidates × 2 matches × 3 hiders × 25 generations)

| Condition | Survival s (of 90) | Seeker wins /40 | Fitness vs walker |
|---|---|---|---|
| Walker, no role training | 29.4 | 36 | — |
| Role-trained, best | 39.9 (42.2 on re-run) | 34 (33) | +0.13 to +0.17, p ≤ 0.006 |
| **Role-trained, population mean** | **45.4** | **30** | **+0.21, p = 0.0002** |
| LC4 + LPLC2 silenced | 42.8 | 31 | no drop |
| DNp01 silenced | 41.4 | 33 | no drop |
| **PFL3 silenced** | **14.9** | **40** | **−0.18, p = 9e-6** |
| **Shuffled wiring, trained (same budget)** | **19.8** | **40** | **−0.12, p = 0.002** |
| Random walk | 18.2 | 40 | −0.14, p = 5e-4 |
| Scripted hider | 42.5 | 35 | +0.15, p = 0.001 |

*Re-runs on the same seeds differ by about ±0.05 fitness because brain noise depends on batch composition.*

## 5. Conclusions

1. **The real wiring is necessary for every role.** With the same training budget, shuffled wiring fails as a walker (2.4 vs 6.3 rooms), as a seeker (0/40 wins) and as a hider (19.8 s survival, worse than an untrained real-wiring hider).
2. **The compass-steering circuit carries the behaviour.** Silencing the 24 PFL3 neurons collapses exploration, seeking and hiding (all p ≤ 1e-5).
3. **The escape pathway is not used.** Silencing LC4/LPLC2 or DNp01 does not hurt hiders; they survive by hiding, not by reflexive escape. LC10a gives the seeker at most a small, non-significant contribution.
4. **Locomotion mattered more than tuning.** One biologically justified constant (max turn rate 3 → 6 rad/s) took brain-seeker wins from 3/40 to 15/40, more than any training.
5. **Role training helps hiders, not seekers.** Trained brain hiders survive 13–16 s longer than untrained ones, on par with or better than the hand-written scripted hiders. Seeker training shows no held-out gain in any of three attempts.

## 6. Negative results worth reporting

- **Photoreceptors barely exist in navcore** (16 L / 9 R neurons). One-sided drive up to 60 Hz produces zero output in DNa02/DNa03/DNg13, so the navcore fly has no visual wall signal.
- **Looming drives only DNp01** (strongly lateralised: 105 vs 26 Hz at 10 Hz input), never the steering DNs.
- **Three plausible locomotion fixes failed** (Section 3).
- **Seeker role training:** no held-out gain in three runs (4 matches/candidate, 8 matches/candidate, and after the locomotion fix).

## 7. Limitations

- The role policies, route planner, forward speed, camping, auto-vent and idealised compass are engineered. The claim is about the **steering and sensory pathways** between those pieces, tested with shuffles and ablations.
- All results are on `navcore` (22,686 neurons), 1 shuffled graph for the role controls, and the short preset. The full graph is reserved for the showcase.
- Scripted opponents only. Self-play (R2) was not run.

## 8. Reproduce

```bash
bash flyseek/train/run_phase55.sh
```

The pipeline is resumable. Per-run logs are in `C:\flyseek-data\results\train\<run>\log.jsonl`, and the pipeline log is `C:\flyseek-data\phase55.log`.
