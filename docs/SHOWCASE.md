# AMongus Fly showcase: connectome flies playing Hide n Seek

*Replays: `showcase_v4_*` (on the [v1.0 release](https://github.com/IrwinJam/FruitFly-repo/releases/tag/v1.0)).
Every number below comes from a file in [`docs/results/`](results/); the table in Section 7 lists them.*

Every fly in these matches is a spiking model of the fruit fly central nervous system, built from the
**MaleCNS v1.0** connectome. The wiring is never trained. Training only tunes a small set of engineered
parameters around it (Section 3, [PARAMETERS.md](PARAMETERS.md)).

## 1. What you are watching

| Panel | Meaning |
|---|---|
| Point cloud | 165,122 neurons in the MaleCNS layout, of which **22,686 are simulated** (the `navcore` subgraph). A glowing point means that neuron just spiked. |
| Circuit bars | The current mean firing rate of a named pathway in that fly: compass and goal (EPG/FC2/PFL3), steering descending neurons (DNa02/DNa03), pursuit (LC10a), looming and escape (LC4/LPLC2/DNp01), and smell (danger meter, pings; always empty, see Section 3). |
| Danger meter | The Among Us hider signal for how close the seeker is. The hider's policy uses it to decide when to flee. |
| Map | The Skeld. Circles show vision range, lines show recent paths, and squares are the 14 vents at their real in-game coordinates. |

## 2. Showcase matches

All flies are brains, playing under full-length rules: 10 s freeze, 300 s round, 120 s Final Hide. The seeds
were fixed before recording and every recorded match is published, whatever its outcome. The seeker uses the
walker adapter, chosen by a rule written down before the run (role training gave no gain, Section 4.4).

| Replay | Winner | Ended | Hiders caught (times, s) |
|---|---|---|---|
| `showcase_v4_3h_s6000` | Seeker | 3:26 | 3/3 (19, 23, 206) |
| `showcase_v4_3h_s6001` | Seeker | 1:13 | 3/3 (47, 65, 73) |
| `showcase_v4_3h_s6002` | Seeker | 3:16 | 3/3 (134, 177, 197) |
| `showcase_v4_3h_s6003` | Seeker | 3:19 | 3/3 (58, 59, 200) |
| `showcase_v4_3h_s6004` | **Hiders** | 5:00 | 2/3 (131, 269) |
| `showcase_v4_3h_s6005` | Seeker | 3:38 | 3/3 (115, 120, 218) |
| `showcase_v4_5h_s6000` | Seeker | 1:23 | 5/5 (40, 41, 80, 81, 84) |
| `showcase_v4_5h_s6001` | Seeker | 3:22 | 5/5 (31, 55, 58, 169, 203) |
| `showcase_v4_5h_s6002` | Seeker | 3:17 | 5/5 (87, 98, 126, 179, 198) |

Eight seeker wins and one hider win, consistent with the 300 further games in Section 4.3.

### 2.1 Quality checks

Every showcase run is measured against thresholds fixed in advance (`amongusfly.experiments.clean_gate`,
`quality_checks.json`) and then watched.

| Check | Threshold | Result |
|---|---|---|
| Time inside geometry | ≈ 0 | **0** |
| Seeker spinning on the spot | ≤ 3% | **1.1%** |
| Seeker lost in one room (worst match) | < 50% | **37%** |
| Seeker within one grid cell of a wall | ≤ 25% | 46.4% (fail) |
| Seeker ground speed | ≥ 75% of commanded | 69.5% (fail) |

Three of five pass. The two failures are real and have one cause:

- Measured against the exact wall geometry, the seeker's body is **in contact 39.4% of the time**, spends
  **20.4%** of the match in contact for more than half a second at a stretch, and makes **16.4 sharp turns
  per minute while touching a wall**, about one bounce every four seconds. It reaches 91% of the speed the
  near-wall slowdown rule allows, so it is not pinned, but it bounces along walls.
- **The steering is threshold-like.** The pathway from the compass to the descending neurons barely turns
  the fly for heading errors below about 25–40° (`steering_curve.json`), so a fly holds a standing offset and
  drifts wide in corridors. An idealised controller with the same planner and body touches walls 7% of the
  time; given a 40° dead zone, 22% (`ideal_steering.json`). The brain-driven walker touches them 36% of the
  time (`walking_quality.json`).
- Changes around the network do not fix it (`steering_fixes.json`): the best of six, exaggerating the goal
  error threefold, lowers time near walls from 43% to 34%, still above the 25% threshold; raising the gain on
  small errors or stronger corridor centring make it worse. A learned wall sense did not help either (Section 5).
- **The dead zone belongs to the frozen model** (`steering_sensitivity/`): it stays at about −25° to 40° for
  every connection scale from 0.41 to 0.57 (widening at 0.36; above 0.57 the steering neurons fall almost
  silent), and appears at every compass and goal input strength from 0.5× to 2×.
- **Driving the compass more gently halves wall contact.** At 0.75× input the fly responds to smaller errors
  (steady heading error 10° vs 16°). Walking for 120 s, wall contact falls from 36% to 21% and rooms rise from
  13.8 to 15.0 (`walking_input_gain.json`). On seeds used nowhere else: 9.1 vs 8.3 rooms in 45 s (p = 0.01,
  `navigation_input_gain.json`), and in 20 full rounds against scripted hiders the seeker touches walls 21% vs
  42% (p = 7×10⁻⁸), is stuck 11 s vs 61 s and wins 20/20 vs 14/20 (p = 5×10⁻⁴, `seeker_input_gain.json`).
  With the gentler input in every fly, the seeker wins 20/40 short all-brain rounds instead of 7 (p = 0.003,
  `allbrain_input_gain.json`) and, over the 300 verification games (`mass_gentle/`,
  [`seeker_paths_300_gentle.png`](seeker_paths_300_gentle.png)), 198/200 against three hiders (99%, CI 96–100%,
  was 91%; p = 3×10⁻⁴) and 87/100 against five (was 76%; p = 0.07), touching walls 25–30% of the time. Hiders
  gain little (61 s vs 58 s against the scripted seeker, n.s., `hiders_input_gain.json`), so the game tilts
  toward the seeker. The showcase games and every other result use the original strength.
- [`seeker_paths.png`](seeker_paths.png) draws the seeker's path in all nine matches: it runs
  down corridors and through rooms, and bounces along walls at close range.
  [`seeker_paths_300.png`](seeker_paths_300.png) draws all 300 verification games, replayed with every
  position saved (seeker wins 181/200 and 77/100 against the published 182 and 76; wall contact 41%).

A second recording of the same nine seeds gives the same picture (`quality_checks_rerun.json`: contact 37.9%,
18.1 bounces per minute).

## 3. What the connectome does, and what is engineered

| Component | Source | Notes |
|---|---|---|
| Neurons and synapses | **Connectome** | MaleCNS v1.0. Synapse counts become weights and the neurotransmitter sets the sign. |
| Neuron model | Engineered | Leaky integrate-and-fire with the Shiu et al. 2024 parameters, plus one global weight scale (0.514) derived from dataset statistics (`brain_calibration_sweep.json`, `brain_stability.json`). |
| Which neurons are simulated | Engineered | `navcore`: 22,686 neurons kept by connection strength between the sensory inputs and the descending neurons. |
| Seeing a hider or a looming seeker | Engineered encoder → **connectome** | Side-level (not retinotopic) firing rates into LC10a and LC4/LPLC2. |
| Danger meter and Final Hide pings | Engineered policy only | Meant to drive the aversive and attractive odour neurons, but the channel selects neurons by cell-body side and the odour receptors have none in the annotations (their cell bodies are in the antenna), so it reaches no neurons. The flies still react to both through the engineered policies. The fix (take the side from the nerve each receptor enters by, `rootSide`) is left to follow-up work. |
| Heading (compass) | Engineered | An idealised EPG bump computed from the true heading. The model has no path integration. |
| Where to go | Engineered policy | A route planner plus role policies. The seeker chases, remembers where it last saw a hider and follows pings. The hider picks a spot, camps and flees. Their weights are trained. |
| **Turning toward the goal** | **Connectome** | EPG/FC2 → PFL3, PFL2 → LAL → DNa02/DNa03. This is the part the controls test. |
| Turn/dash/back readout | Engineered decoder | Weighted left−right rates of the descending neurons, with trained weights. |
| Forward speed and max turn rate | Engineered constants | 2.0 units/s and 3–7 rad/s (trained within that range). A real walking fly saccades at about 8–17 rad/s. |
| Near-wall slowdown, wedge escape, re-plan on contact | Engineered rules | The fly slows to 45% within one body width of geometry. A body pushed inside geometry nudges itself out. After 0.6 s of wall contact the route is re-planned. |
| Auto-vent and hider camping | Engineered rules | A brain cannot press "vent". A camping hider stands still. |
| Training | CMA-ES on the parameters above | **No connectome weight is ever changed.** |

## 4. Does the wiring matter?

All matches below are held-out: their seeds were never used in training. Shuffled wiring is a
degree-preserving shuffle of the same graph, trained with the same budget.

### 4.1 Senses and compass

- **Senses reach the right neurons.** Pursuit input into LC10a drives the steering neurons DNa02/DNa03
  on the side of the target (lateralisation index 27 Hz at 25 Hz input, above all 100 shuffled graphs), and looming input
  drives the escape neuron DNp01 (179 Hz at 25 Hz input; shuffled maximum 1.4 Hz)
  (`brain_sensory_checks_vs_shuffles.json`). Pursuit is wiring-specific only up to 25 Hz input (at 50 Hz some
  shuffled graphs match the real wiring), and its encoder is capped at 25 Hz; the looming drive onto DNp01
  exceeds every shuffle at every rate tested, up to the looming encoder's cap of 100 Hz.
- **The compass circuit steers to a goal the fly cannot see** (`compass_open_loop.json`,
  `invisible_goal_100_shuffles.json`). With heading on EPG and goal on FC2, every fly reaches the goal (40/40);
  with the goal removed, 18%; across 100 shuffled graphs, 0–33% (mean 12%). The real wiring beats all 100,
  an empirical p of 0.01.
- **PFL3 and PFL2 convert the goal into a turn** (`goal_to_turn.json`). With the decoder's goal-behind rule
  off, success falls from 95% to 45% with PFL3 silenced, 30% with PFL2 silenced and 18% with both, as low
  as with no goal signal at all (measured with the rule on).
- **It works in the whole connectome.** Run unchanged on all 165,122 neurons, the compass circuit swings the
  steering neurons by 38 Hz (40 Hz on `navcore`, `compass_open_loop_full.json`), and 83% of flies reach the
  unseen goal, against 18% without the goal and 15% with PFL3 silenced (`invisible_goal_full.json`). The
  walker's trained settings do not transfer: on the same 40 episodes it visits 2.9 rooms instead of 8.1 and
  is stuck 32 of 45 s (`navigation_full.json`).

### 4.2 Navigation (40 × 45 s episodes from Cafeteria, `navigation.json`)

| Condition | Rooms visited | Wall contact |
|---|---|---|
| Untrained | 7.40 | 5.4 s |
| **Trained** | **8.20** (+0.80, p = 0.02) | 7.6 s |
| Shuffled wiring, same training budget | 2.12 (−5.28, p = 2×10⁻¹⁷) | 39.8 s |
| PFL3 silenced | 2.23 (−5.17, p = 9×10⁻¹⁶) | 15.6 s |

### 4.3 The game: every fly a brain

| Setting | Seeker wins | Hider survival |
|---|---|---|
| 90 s rounds, 40 matches, **trained hiders** | **8/40** | 64.4 s |
| 90 s, hiders without role training | 10/40 (p = 0.05) | 59.3 s |
| 90 s, PFL3 silenced in every fly | 30/40 (p = 5×10⁻¹⁰) | 24.9 s |
| 90 s, shuffled wiring in every fly | 1/40 (p = 0.002) | 76.7 s* |
| 300 s rounds, 20 matches, 3 hiders | 19/20 | 133.1 s |
| 300 s, PFL3 silenced | 17/20 (p = 0.003) | 40.1 s |
| 300 s, **5 hiders** (same adapter, no retraining) | 17/20 | 140.0 s |

\*With shuffled wiring the seeker spends most of the round against walls, so the hiders' long survival
reflects a broken seeker, not good hiding. (`allbrain_90s_3hiders.json`, `allbrain_300s_3hiders.json`,
`allbrain_300s_5hiders.json`)

**Verification: 300 more games on seeds never used elsewhere** (`mass_verification.json`):

| Setting | Seeker wins | 95% interval | Hider survival |
|---|---|---|---|
| 300 s, 3 hiders, 200 games | **182/200 (91%)** | 86–94% | 131.9 s |
| 300 s, 5 hiders, 100 games | **76/100 (76%)** | 67–83% | 131.0 s |

The recorded showcase games, the 20-game evaluations above and a second run of the same evaluations
(`replication_*.json`) are all consistent with these rates (binomial p ≥ 0.44). The GPU simulation is not
bit-for-bit repeatable, so the same seeds give different games on a second run (19 vs 18 wins of 20 with
three hiders, 17 vs 16 with five): the second run measures run-to-run variation, not an independent sample.

**Silencing PFL3 flips the game.** With the compass-steering neurons gone from every fly, the seeker goes
from winning 8 of 40 matches to 30 of 40, and hider survival falls from 64 s to 25 s: the hiders can no
longer get anywhere useful, and the seeker stumbles into them.

**Balance depends on round length.** In 90 s rounds the hiders usually survive. Over the full 5 minutes the
seeker finds almost everyone, as in the real game.

### 4.4 Each role against scripted (hand-written) opponents, 90 s, 40 matches

| Seeker (vs 3 scripted hiders, `seeker_vs_scripted.json`) | Wins | vs walker |
|---|---|---|
| Walker without role training (used in the showcase) | 7/40 | - |
| Role-trained seeker | 0–6/40 | no gain (p = 0.26 and 0.04; neither survives Holm correction) |
| LC10a (visual pursuit) silenced | 5/40 | n.s. (p = 0.08) |
| PFL3 silenced | 4/40 | n.s. (p = 0.07) |
| Shuffled wiring, walker and role training | 0/40 | p = 7×10⁻¹² |
| Random walk | 0/40 | p = 7×10⁻¹² |
| **Scripted seeker** | **29/40** | p = 1×10⁻⁸ |

| Hiders (vs scripted seeker, `hiders_vs_scripted.json`) | Survival | vs walker |
|---|---|---|
| Walker without role training | 53.2 s | - |
| Role-trained (used in the showcase) | 57.0 s | n.s. (p = 0.27) |
| DNp01 (escape) silenced in that hider | 58.6 s | n.s. vs that hider (p = 0.71) |
| LC4 + LPLC2 silenced in that hider | 56.0 s | n.s. vs that hider (p = 0.88) |
| **PFL3 silenced in that hider** | **16.2 s** | p = 7×10⁻¹⁴ vs that hider |
| Shuffled wiring, same training | 24.5 s | p = 2×10⁻⁹ |
| Random walk | 23.3 s | p = 9×10⁻¹⁰ |
| Scripted hiders | 51.1 s | n.s. |

**Which circuits the seeker needs: full 300 s games against scripted hiders, silencing the seeker only**
(`seeker_circuits_300s.json`). In 90 s rounds the seeker wins too rarely for a drop to be measurable;
over a full round it usually wins, so the effect of each circuit is visible.

| Seeker | Wins | vs normal |
|---|---|---|
| Normal | 13/20 | - |
| **PFL3 silenced** (24 neurons) | **1/20** | p = 2×10⁻⁶ |
| **PFL2 silenced** (12 neurons) | **0/20** | p = 3×10⁻⁷ |
| PFL3 and PFL2 silenced | 0/20 | p = 2×10⁻¹⁰ |
| LC10a (visual pursuit) silenced | 8/20 | n.s. (p = 0.06) |
| Shuffled wiring, walker and role training | 0/20 | p = 3×10⁻¹⁴ |

The shuffled seeker had role training on top of the walker, one step more than the real one, so the
comparison favours it. The readout's goal-behind turn reads PFL2's rate directly, so the key conditions were
repeated with that rule switched off (`seeker_circuits_300s_rule_off.json`): normal 15/20, PFL2 silenced 0/20
(p = 9×10⁻⁹), PFL3 silenced 2/20 (p = 2×10⁻⁵), Holm-corrected. Both effects come from the wiring.

What the controls show:
- **PFL3 is necessary for both roles, and PFL2 for the seeker**, with or without the readout rule.
  Silencing PFL3 in a hider costs 41 s of survival; silencing it in every fly hands the game to the seeker.
- **Visual pursuit (LC10a) is not significantly needed** (13 → 8 wins of 20, p = 0.06).
- **Silencing the looming and escape neurons does not change hider survival**, but this test cannot show a
  role for them: the escape dash needs DNp01 above 60 Hz and never triggered in the recorded games.
- **Role training does not help either role**, and **a hand-written seeker wins far more often** (29/40 vs
  7/40). The brain plays the game; it does not play it well.

### 4.5 Circuit activity during play

Spikes from two recordings of the nine showcase games (18 in all, `circuit_activity.json`), per neuron:

- In a hider, the escape neuron DNp01 is silent except while the seeker can see it: **45.7 Hz** then,
  0.2 Hz otherwise. That stays below the 60 Hz the dash needs.
- PFL3 and PFL2 fire at 11–13 Hz while the seeker searches or chases and near 0.5–3 Hz while it is frozen at
  the start, following the goal input on FC2.

## 5. A learned wall sense (negative result)

To reduce wall contact without hand-written avoidance, the walker was given an obstacle sense into the
network and retrained with a wall-contact penalty.

- **Channel choice** (`obstacle_channel_selection.json`): each visual cell type was driven on one side; the
  rule, fixed in advance, picks the type that steers hardest without triggering escape. LLPC1 was selected
  (82 Hz left−right in the steering neurons, no DNp01 rise). On shuffled wiring the same input steers by
  2.5 Hz (`obstacle_channel_shuffled.json`).
- **Result** (`obstacle_sense_navigation.json`, `obstacle_sense_walking.json`, `obstacle_sense_seeker.json`):
  the retrained walker visits fewer rooms (6.80 vs 8.20, p = 0.0002), touches walls more (37% vs 31% with the
  sense switched off) and wins fewer seeker games (3/40 vs 7/40, p = 0.002). Switching the sense off or
  silencing LLPC1 recovers most of the loss.

A frozen network with threshold-like steering cannot use this signal well through a trained adapter alone;
learning inside the network is left to follow-up work. The showcase uses the walker without the sense.

## 6. Limitations

- The games run on the `navcore` subgraph. On the full 165,122-neuron graph the compass circuit works, but the
  walker's trained settings do not transfer (Section 4.1).
- Planning and the compass input are engineered and idealised. The claim is about the steering and sensory
  pathways in between, which the shuffles and silencing test.
- Vision is side-level, not retinotopic. There is no path integration, and escape timing is uncalibrated.
- The odour channels (danger meter, pings) reach no neurons (Section 3).
- Flies touch walls about 40% of the time at the standard compass input; 0.75× halves this (Section 2.1).
- One shuffled graph is used for the trained controls (10 for the open-loop compass test, 100 for the
  invisible-goal test and the sensory checks);
  20–40 held-out matches per condition, plus 300 games for the headline win rates.
- The GPU simulation is not bit-for-bit repeatable, so a rerun of the same seeds gives different games.
- The weight scale was checked for runaway activity on the `pruned5` graph, not on `navcore`.
- `pIP1` and the forward-drive candidates are unverified cell-type guesses.

## 7. Result files

All in [`docs/results/`](results/), written by the tools named in [`run_pipeline.sh`](../amongusfly/train/run_pipeline.sh)
and [`run_obstacle_sense.sh`](../amongusfly/train/run_obstacle_sense.sh).

| File | Contents |
|---|---|
| `brain_calibration_sweep`, `brain_stability`, `brain_sensory_checks`, `brain_sensory_checks_vs_shuffles` | Brain model: weight scale, stability, sensory pathways against 100 shuffled graphs |
| `compass_open_loop`, `compass_by_heading`, `invisible_goal`, `invisible_goal_100_shuffles`, `goal_to_turn` | Compass circuit: open-loop steering, closed-loop goal reaching, PFL3/PFL2 silencing |
| `compass_open_loop_full`, `invisible_goal_full`, `navigation_full` | The same model on all 165,122 neurons |
| `navigation`, `walking_quality`, `steering_curve`, `ideal_steering`, `steering_fixes` | Walker: exploration, wall contact, the steering dead zone, fixes tried |
| `steering_sensitivity/`, `walking_input_gain`, `navigation_input_gain`, `seeker_input_gain`, `hiders_input_gain`, `allbrain_input_gain`, `mass_gentle/` | The dead zone against connection scale and input strength; the gentler compass input, including the 300 verification games |
| `seeker_vs_scripted`, `hiders_vs_scripted`, `seeker_circuits_300s`, `seeker_circuits_300s_rule_off` | Each role against scripted opponents, with silencing and shuffled wiring |
| `allbrain_*`, `replication_*`, `mass_verification`, `mass/` | Every fly a brain: held-out evaluations, replication and 300-game verification |
| `quality_checks`, `quality_checks_rerun`, `circuit_activity` | Showcase games: quality checks and circuit activity |
| `obstacle_channel_*`, `obstacle_sense_*` | Wall-sense experiment (Section 5) |

## 8. Credits

- **MaleCNS v1.0 connectome:** HHMI Janelia FlyEM, Google Research, University of Cambridge. CC-BY 4.0. <https://male-cns.janelia.org/>
- **Leaky integrate-and-fire reference model:** Shiu et al. 2024.
- **PFL2/PFL3 steering:** Westeinde et al. 2024.
- **Vent coordinates and links:** SkeldJS generated map data (MIT). <https://github.com/SkeldJS/SkeldJS>
- **Among Us and The Skeld:** © Innersloth. This is a non-commercial fan research project, not affiliated with Innersloth. No game art is used or distributed.
