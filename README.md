# AMongus Fly

This project uses a connectome-constrained model of the fruit fly central nervous system (MaleCNS v1.0) to play Among Us Hide n Seek on The Skeld. One fly is the Seeker, three (or five) are Hiders, and each one gets a live brain-activity panel driven by its own recorded spikes.

**The wiring is never trained.** Training only tunes a small set of engineered parameters around it: sensory gains, a route planner, role policies and the motor readout. The tests below ask whether the real wiring is doing the work.

- **Watch it:** a short showcase video and all nine recorded games are on the [v1.0 release](https://github.com/IrwinJam/FruitFly-repo/releases/tag/v1.0).
- **Showcase, methods and results:** [docs/SHOWCASE.md](docs/SHOWCASE.md)
- **Design and prior work:** [docs/DESIGN.md](docs/DESIGN.md)
- **Result files:** [docs/results/](docs/results/) (listed in [SHOWCASE.md §7](docs/SHOWCASE.md#7-result-files))

## Results

All result files are in `docs/results/`.

| Question | Answer | Evidence |
|---|---|---|
| Do senses reach the steering neurons at realistic drive? | Yes. Pursuit (LC10a) up to 25 Hz input, where its encoder is capped (at 50 Hz some shuffled graphs match the real wiring); looming (DNp01) at every rate tested, up to 100 Hz | `brain_sensory_checks_vs_shuffles.json` (100 shuffled graphs) |
| Can the fly's own compass circuit steer it to an unseen goal? | Yes: 100% reach it vs 18% with the goal removed and at most 33% on any of 100 shuffled graphs (p = 0.01). Silencing PFL3 and PFL2 together drops it to 18% | `invisible_goal_100_shuffles.json`, `goal_to_turn.json` |
| Does the compass circuit work in the whole connectome? | Yes: on all 165,122 neurons it swings the steering neurons by 38 Hz (40 Hz on `navcore`) and 83% of flies reach the unseen goal (18% without the goal, 15% with PFL3 silenced). The walker's trained settings do not transfer: 2.9 rooms vs 8.1 | `compass_open_loop_full.json`, `invisible_goal_full.json`, `navigation_full.json` |
| Does the real wiring matter for exploring The Skeld? | Yes: 8.2 rooms per 45 s vs 2.1 with shuffled wiring and 2.2 with PFL3 silenced, same training budget | `navigation.json` |
| Does it matter for playing the game? | Yes: a shuffled-wiring seeker catches no one (0/40); with PFL3 silenced in every fly the seeker wins 30/40 instead of 8/40 and hider survival falls from 64 s to 25 s | [docs/SHOWCASE.md](docs/SHOWCASE.md) §4 |
| Which circuits does the seeker need? | The steering cells: silencing PFL3 (24 neurons) or PFL2 (12) alone drops the seeker from 13/20 to 1/20 and 0/20 in full games, and still to 2/20 and 0/20 (vs 15/20) with the readout rule that reads PFL2 switched off; silencing visual pursuit (LC10a) does not significantly | `seeker_circuits_300s.json`, `seeker_circuits_300s_rule_off.json` |
| Do the showcase results hold up at scale? | Yes: 300 more games on fresh seeds, seeker wins 91% (3 hiders, CI 86–94%) and 76% (5 hiders, CI 67–83%); all smaller samples consistent | `mass_verification.json` |
| Does role training help? | No, for either role (hiders +3.8 s, p = 0.27; the role-trained seeker is no better than the plain walker). The showcase seeker is therefore the walker, by a rule fixed before the run | [docs/SHOWCASE.md](docs/SHOWCASE.md) §4.3–4.4 |
| Why do flies touch walls so often, and can it be fixed? | Steering ignores heading errors of about −25° to 40°, at every connection scale and input strength that still steers. Driving the compass input at 0.75× halves wall contact: on fresh seeds the seeker touches walls 21% vs 42% and wins 20/20 vs 14/20 against scripted hiders; with it in every fly, the seeker wins 198/200 verification games vs 3 hiders (99%, was 91%) and 87/100 vs 5 (was 76%), while hiders gain little | `steering_sensitivity/`, `*_input_gain.json`, `mass_gentle/` |
| Do the flies beat a hand-written policy? | No: a scripted seeker wins 29/40 where the brain seeker wins 7/40 | `seeker_vs_scripted.json` |

## Watch a match

The quickest way is the showcase video, `amongusfly_showcase.webm` on the [v1.0 release](https://github.com/IrwinJam/FruitFly-repo/releases/tag/v1.0): under a minute of
real, unedited game time, one clip per story beat.

To watch the games themselves, with every fly's brain activity live:
1. Download `amongusfly_showcase_replays.zip` from the [v1.0 release](https://github.com/IrwinJam/FruitFly-repo/releases/tag/v1.0) and unzip it into
   `viewer/public/replays/` (so that `viewer/public/replays/showcase_v4_3h_s6000/meta.json` exists).
2. Start the viewer:
   ```bash
   cd viewer && npm install && npm run dev
   ```
3. Open <http://localhost:5173>. The picker at the bottom right switches between the nine games.

Controls: space to play or pause, click a brain panel to follow that fly, and the marks on the seek bar jump to
catches, vents and pings. URL options for clips: `?replay=NAME&start=115&end=190&speed=2`.

Re-making the video (written into `media/`, not downloaded):
```bash
.venv/Scripts/python -m amongusfly.experiments.plan_reel --prefix showcase_v4
# then open http://localhost:5173/?reel=/replays/reel_showcase_v4.json&rec=1&save=server
```

> Status: an early research prototype. This is a simplified leaky integrate-and-fire (LIF) model with engineered sensory and motor mappings. It is **not** a validated or living fly brain.

## Reproducing from a fresh clone

The connectome data (~1.4 GB including caches) is **not stored in this repo**. It is public and gets rebuilt with the scripts below. On Windows, run these from the repo root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
.venv/Scripts/python -m pip install -r requirements.txt

# data lives outside the repo; paths currently point at C:\amongusfly-data
.venv/Scripts/python -m amongusfly.connectome.download        # ~1.05 GB from Janelia's public bucket
.venv/Scripts/python -m amongusfly.connectome.celltypes       # resolve sensory/motor roles
.venv/Scripts/python -m amongusfly.connectome.build_graph     # full + pruned5 graphs
.venv/Scripts/python -m amongusfly.connectome.layout          # 2D soma layout
.venv/Scripts/python -m amongusfly.connectome.export_viewer_data
.venv/Scripts/python -m amongusfly.world.skeld                # Skeld grid + vents

.venv/Scripts/python -m amongusfly.brain.bench                # GPU benchmark
```

Then train and play (each script is resumable):

```bash
bash amongusfly/train/run_pipeline.sh         # every adapter, evaluation, control, showcase game and check
bash amongusfly/train/run_obstacle_sense.sh   # the wall-sense experiment (SHOWCASE.md §5)
.venv/Scripts/python -m amongusfly.experiments.results_tables   # results tables
```

## Data and attribution

- **MaleCNS v1.0 connectome:** HHMI Janelia FlyEM, Google Research, and the University of Cambridge. Licensed CC-BY 4.0. <https://male-cns.janelia.org/>
- **Among Us and The Skeld:** © Innersloth. This is a non-commercial fan research project, and no game assets are distributed. `skeld_map.json` holds walkable-area geometry only.
- **Vent coordinates and links:** SkeldJS generated map data (MIT). <https://github.com/SkeldJS/SkeldJS>

## License

Code: [MIT](LICENSE). Third-party data keeps its own terms (above).
