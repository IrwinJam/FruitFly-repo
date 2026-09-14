# FlySeek — Connectome Fruit Flies Play Among Us Hide n Seek on The Skeld

*Project plan · drafted 2026-09-13*

Simulated fruit fly brains (built from the real connectome wiring) control players in an Among Us Hide n Seek game on The Skeld. One fly is the **Seeker** and the others are **Hiders**. Each fly gets its own live brain panel in the **IMG_0897 style**: a black background, a grey point cloud of the brain above the nerve cord, and colored glows where neurons are firing.

---

## 0. TL;DR — the architecture in one picture

```
 ┌─────────────────────────── Python backend (GPU) ────────────────────────────┐
 │                                                                              │
 │  skeld_map.json ──► Skeld World (2D nav grid, walls, vents, tasks, timers)   │
 │                         │  per-fly egocentric senses                         │
 │                         ▼                                                    │
 │   Sensory encoders  (ray-cast vision → photoreceptors, danger → odor,        │
 │                      pings → odor L/R, wall bump → mechanosensory)           │
 │                         │ Poisson drive                                      │
 │                         ▼                                                    │
 │   Batched LIF brain  [fly0=Seeker, fly1..N=Hiders]  MaleCNS v1.0 (166,700 n) │
 │   one shared sparse weight matrix · per-fly membrane state (batch dim)       │
 │                         │ descending-neuron (DN) firing rates                │
 │                         ▼                                                    │
 │   Motor decoders  (DNa02 L−R → turn, oDN1/DNg100 → forward, MDN → back,      │
 │                    DNp01 giant fiber → dash) ──► body/kinematics ──► World   │
 │                                                                              │
 │   Recorder: per-tick positions + binned spike indices ──► .npz replay files  │
 └───────────────────────────────┬──────────────────────────────────────────────┘
                                 │ WebSocket (live) or replay file
 ┌───────────────────────────────▼──────────────────────────────────────────────┐
 │  Browser viewer (Three.js/WebGL)                                             │
 │   [Seeker brain]   [ Skeld top-down map, players, vision cones ]  [Hider 1]  │
 │                                                                  [Hider 2]  │
 │                                                                  [Hider …]  │
 └──────────────────────────────────────────────────────────────────────────────┘
```

**The main findings from the research that shape this plan:**

1. **Use MaleCNS v1.0, not FlyWire, as the brain.** It includes the brain *and* the ventral nerve cord (VNC), which is the "brain on top, spinal cord below" look in IMG_0897 (that screenshot says 165,122 neurons, which is MaleCNS scale). It's public (CC‑BY 4.0) and needs no token for bulk download. Almost every viral game project this month (DOOMFLY, Fly64 / Mario, Minecraft, Beat Saber) uses it.
2. **A connectome simulation cannot learn to walk by itself.** It has fixed wiring and no plasticity. Every project that "walks" does one of two things. Most read a few descending neurons and feed them into a pre-built locomotion controller (a CPG or an imitation-learned policy, as Eon and NeuroMechFly do). Some also train a **small adapter** (input gains and a linear DN readout) with RL or evolution while the connectome stays frozen (fly-craftax does this). "Training the flies" here therefore means training adapters, plus optional mushroom-body dopamine plasticity as a stretch goal.
3. **Among Us is 2D and top-down**, so the game doesn't need a physics-simulated leg body. A kinematic "unicycle" body driven by DN rates is enough, and it's about 1000× cheaper. NeuroMechFly/MuJoCo is an optional visual-flair add-on (like IMG_0900) and isn't needed for gameplay.
4. **The biology fits the roles well:**
   * **Seeker:** MaleCNS is a *male* fly. Males have a dedicated **pursuit circuit**: LC10a visual neurons detect a moving target, that signal passes through AOTU019/025 to the steering neurons DNa03 and DNa02, and **P1** arousal neurons turn up the gain. "Seek mode" can literally mean driving P1.
   * **Hider:** An approaching Seeker is a **looming** stimulus. LC4 and LPLC2 feed the **giant fiber (DNp01)**, which triggers escape. **Moonwalker (MDN)** neurons drive backing away.
   * **Final-hide pings** can become a lateralized **attractive odor** for the Seeker. The **danger meter** can become an **aversive odor** for Hiders.
5. **Your hardware** (RTX 2060 Super 8 GB, Ryzen 5 7500F, 15.6 GB RAM) can run a few whole-CNS flies, but **not in real time**. The plan slaves game time to brain time: record the match, then play it back smoothly at 1×. Live mode uses a pruned graph.

---

## 1. What you have in the folder

| Item | Findings |
|---|---|
| `skeld_map.json` | `grid_res` 0.15, **13,563 walkable cells** with `x, y, room, count`. The coordinates are Among Us world units (x −22.8…18.5, y −17.1…6.0), with Cafeteria near the origin. **99.96% of cells form one connected region.** Two tiny islands (5 cells in Reactor, 1 in Storage) should be dropped. Cafeteria tables and the Admin table show up as holes, which is good. There's a rendered preview at `docs/skeld_map_preview.png`. |
| Room labels | They need normalizing: `Upper Engine`/`UpperEngine`, `Lower Engine`/`LowerEngine`, `Nav`/`Navigation`, `O2`/`LifeSupp`, and 38 `Unknown` cells (assign each one the label of its nearest labeled neighbor). |
| `count` field | Ranges 1–1633 (total 98,268). Its meaning isn't documented. It looks like a visit density from recorded player traces. It could be a spawn or hiding-spot prior, or a "popular route" heatmap. **Confirm with whoever made the file.** |
| Missing from the map | There are no walls (derive them as the boundary of the walkable set), no **vents** (14 on The Skeld, linked in chains), and no **task stations**. These need to be added by hand in a small `skeld_extras.json`. |
| `cave-secret.json` | Contains one `token` field. This is almost certainly a **FlyWire CAVE token** (for `CAVEclient`, datastack `flywire_fafb_public`). **The main MaleCNS path doesn't need it.** Only the optional FlyWire features (§3.2) use it. Note that neuPrint uses a *different* token. |
| `Untitled_Message/` | Six reference screenshots (see §2). |

> ⚠️ **Housekeeping:**
> 1. This folder is inside **OneDrive**. The connectome download is 1–3 GB, plus caches. Put `data/` outside OneDrive (for example `C:\flyseek-data`) or exclude it from sync.
> 2. Before any `git init`, add `cave-secret.json` to `.gitignore` and load it through an env var such as `CAVE_TOKEN_PATH`. Never print the token.

---

## 2. Reference screenshots → design requirements

| Image | What it shows | What we take from it |
|---|---|---|
| **IMG_0897** ⭐ | "Neural motor rehearsal" demo with an Among Us soundtrack. On the right is a black panel titled "165,122 neurons · … motor neurons". The brain is seen **from the front** (optic lobes spread wide), with a **VNC below** it. Points are grey-white, with **green and blue additive glows** at active clusters and neurite streaks. | **This is the target brain panel.** Front view of the brain with the VNC underneath, soma point cloud, additive colored glows that fade, and a neuron-count header. One panel per fly. |
| **IMG_0898** | "6 fruit fly brains in a virtual carnival". The game view is in the middle and **6 small brain thumbnails are stacked on the right**, one per fly. | Multi-fly layout: a column of per-fly brain panels next to the world view, with the selected fly highlighted. |
| **IMG_0899** | Fly64 (Mario). Panels: soma cloud ("Measured soma positions + deterministic missing-position layout"), fly retinal input, spike-density raster, superclass rates, decoded stick, latency, real-time factor, and an honesty disclaimer. | A debug/"science" overlay per fly: retina view, decoded motor state, superclass Hz, real-time factor. Include a disclaimer too. |
| **IMG_0900** | DOOM plus a NeuroMechFly body: "138,639 neurons / FlyWire v783 female", leg drive L/R, forward/turn channel Hz. | DN channel bars (forward L/R, turn L/R) per fly. Optional 3D body view as a stretch goal. |
| **IMG_0901/0902** | Minecraft bee "fly‑001" with a floating neural HUD: dark panel with a few large colored blobs (yellow, pink, blue). | Named-circuit glows that are **readable at a glance** (e.g. pink = pursuit circuit, blue = steering). Name labels over players. |

---

## 3. Research summary

### 3.1 Connectome datasets

| Dataset | Neurons | Coverage | Access | Use here |
|---|---|---|---|---|
| **MaleCNS v1.0** (Janelia + Google + Cambridge; released 2026‑06‑08, Cell paper 2026‑09‑03) | ~166,700 | Brain + optic lobes + **VNC** | Public bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (Feather). neuPrint for queries (needs a neuPrint token). HF mirror `svgmediabills/malecns-connectome`. | **Primary brain for all flies** |
| FlyWire FAFB v783 (female) | ~139,255 | Brain only (no VNC) | Zenodo/GitHub tables. CAVE via **your token** | Optional cross-check / "female hider" variant |
| MANC / BANC | 23k VNC / whole female CNS | VNC / brain+nerve cord | neuPrint / CAVE | Reference for leg motor circuits |

**MaleCNS files to download** (from the official download page):

| File | Size | Need? |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | ✅ types, class, superclass, side, soma location |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | ✅ excitatory/inhibitory sign |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | ✅ the graph |
| `body-stats-…feather` | 780 MB | optional |
| `syn-points` / `syn-partners` / `tbar-neurotransmitters` | 12.7 / 6.8 / 2.7 GB | ❌ skip (RAM limit). Use synapse centroids only if soma positions are missing, and pull them per neuron from neuPrint |
| Neuropil ROI meshes | small | ✅ faint brain and VNC outline for the panel |

### 3.2 Brain simulators (prior art)

| Project | Model | Notes |
|---|---|---|
| **Shiu et al. 2024, *Nature*** — `philshiu/Drosophila_brain_model` | LIF with α-synapses, Brian2 | The reference model. Stimulate by ID with Poisson input and read spike rates. Validated on sugar → MN9 feeding and grooming circuits. |
| `eonsystemspbc/fly-brain` | Same model in Brian2, Brian2CUDA, **PyTorch**, NEST GPU, GeNN | Activation/silencing API and a benchmark suite. **We base our PyTorch kernel on this.** GPL‑2. |
| `vshapenko/flypoke` | NumPy exponential-Euler LIF | Parameters: rest −52 mV, threshold −45 mV, τm 20 ms, delay 1.8 ms, 0.275 mV/synapse. About a second of compute per trial on a laptop CPU. MIT. |
| `blendi-remade/fly-brain-minecraft` | MaleCNS, edges ≥ 5 synapses (**176k neurons, 6.29 M edges**), one thread per fly, **max 4 flies** | Neurons "light up at their real soma positions and fade ~300 ms" (the HUD look we want). Only integrates neurons that are active. 25–60 ms of compute per 50 ms tick on 32 cores. MIT. |
| `abgnydn/webgpu-fly` | FlyWire brain + MANC VNC in WebGPU, linked by DN type name | About 4× slower than real time on an M2 Pro. MIT. |
| `ZeroXClem/closed-loop-fly` | MaleCNS in WebGPU. The compound eye is sampled at **1,771 connectome columns**. Steers with DNa02. | A good reference for the eye model. |
| `eonfathom/FastFly` | CUDA/CuPy, aiming for real-time FlyWire | Performance reference. |

### 3.3 Game and embodiment projects: how they wire senses and motors

| Project | Input mapping | Output mapping | Training / learning | Honest status |
|---|---|---|---|---|
| **DOOMFLY** (`nftechie/doomfly`) | 3,335 R1–R6 cells get brightness, 811 R8 cells get color | DNp20 R−L → turn. DNpe017 → move/fire | Aversive dopamine into 2 PPL101 cells on damage. Plasticity on 4,184 KC→MBON11 synapses | "Failed visual, conditioning and survival validation gates" |
| **Fly64** (`ornata/fly`, IMG_0899) | Six 128×128 cameras (~270°) at 10 Hz → eye cells. Brain at 50 Hz | **DNg100 → forward. DNa02/DNg13 R−L → steer. DNp01/DNp10 burst → jump** | None. The stick output is smoothed, deadbanded, and clamped | "Not a validated fly or trained player" |
| **Minecraft** (`blendi-remade`) | Vision, odor tables, sugar/bitter GRNs, Johnston's organ, looming | Priority decoder: escape > landing > walk > feed > groom | None | Works as an interactive demo |
| **Eon embodied fly** (Mar 2026) | Vision (flyvis, "decorative"), GRNs, antennal mechanosensation | **DNa01/DNa02 → steering, oDN1 → forward**, MN9 → feeding. Feeds an **imitation-learned** NeuroMechFly controller | Hand-tuned mapping. Body/brain sync every 15 ms | "Research and demonstration platform" |
| **fly-craftax** | Game state → sensory neurons | Linear DN readout | **PPO trains only the linear readout. The connectome stays frozen.** | Includes baselines |
| **TheMrRaGe/flybrain** | Vision, olfaction | DNa02/DNg13 steering, DNp01/DNp10 jump | Mushroom-body DAN→KC→MBON conditioning, with a documented FINDINGS.md | Careful about validation |

### 3.4 Bodies

| Option | What it is | Verdict |
|---|---|---|
| **Kinematic unicycle** (ours) | Position, heading, speed. `v = gain·forward − back`, `ω = gain·(R−L)` with collisions on the nav grid | ✅ **Use for gameplay and training** |
| **NeuroMechFly v2 / FlyGym** (EPFL, *Nature Methods* 2024) | MuJoCo fly with 87 joints, vision, olfaction. A CPG/hybrid turning controller takes a **2D descending command** (the same interface Eon uses) | 🟡 Stretch goal: a 3D "hero cam" of one fly (IMG_0900 look), driven by the same DN signals |
| **flybody** (DeepMind + Janelia, *Nature* 2025) | MuJoCo fly with an RL walking policy that follows steering commands | 🟡 Alternative to FlyGym. Apache‑2.0 |

### 3.5 Hugging Face

* `svgmediabills/malecns-connectome` (dataset): an unmodified MaleCNS mirror, 31.5 GB in total (grab only the files you need). A fallback if the Google bucket is slow.
* `spaces/rf223x/fly-sim`: NeuroMechFly + MaleCNS running in the browser (WASM). A UI and embodiment reference.
* `ngxson/fly-llm-hf` and `eob/gpt-fly`: the connectome used as an ML substrate (a reservoir or sparsity mask). These show the "frozen connectome + trained adapter" idea we're using.
* Curated index: `github.com/cobanov/awesome-fly` (90+ projects).

### 3.6 Neuroscience we'll use (cell types to look up in the MaleCNS annotations)

| Function | Cell types | Role in our game |
|---|---|---|
| Photoreceptors | R1–R6 (motion/brightness), R7/R8 (color) | Ray-cast view → Poisson drive |
| Small moving object | **LC10a** → AOTU019/025 → **DNa03 → DNa02** | Seeker spots and tracks a hider |
| Arousal / pursuit gain | **P1** | Tonic drive while the Seeker is "hunting" (stronger in Final Hide) |
| Looming threat | **LC4, LPLC2 → DNp01 (giant fiber)** | Hider notices the Seeker approaching → dash |
| Steering | **DNa01** (low gain), **DNa02** (high gain), DNg13 | Turn command from the R−L difference |
| Forward walking | **oDN1** (Eon), **DNg100** (Fly64), DNp09 | Forward speed |
| Backward walking | **MDN** (moonwalker), MooSEZ | Back away from the Seeker or a wall |
| Heading / goal | EPG (compass), FC2 (goal), PFL3 (steer toward goal) | Stretch goal: "remember where the hiding spot is" |
| Odor | ORNs → antennal lobe → LH/MB | Pings (attractive, L/R antenna) and danger (aversive) |
| Learning | PAM/PPL1 dopamine → KC→MBON | Stretch goal: reward/punish plasticity |
| Wall contact | Leg/bristle mechanosensory, Johnston's organ | Collision signal |

> **Step 1 of implementation is always to look up the exact type names in the MaleCNS v1.0 annotations** (names like `DNg100` and `oDN1` vary between datasets). The code must fail loudly if a type isn't found. It should never guess silently.

### 3.7 Among Us Hide n Seek rules to model

* 1 Seeker (Impostor) vs up to 14 Hiders (Crewmates). **Plan: 1 Seeker + 3–5 Hiders** (limited by GPU memory).
* Hiders win if they survive the timer. The Seeker wins by eliminating everyone.
* **Tasks** make the timer run down faster.
* **Vents** have limited uses per Hider and a maximum time inside.
* **Danger meter**: fills as the Seeker gets closer.
* **Final Hide**: default **120 s**. The Seeker gets a speed boost, a Seek map, and **pings every ~6 s** revealing the living Hiders.
* **Flashlight mode** (optional): cone vision instead of radial vision.
* Speed and vision radius are host settings. We expose every value in `config/game.yaml` rather than hard-coding values we couldn't verify.

---

## 4. System design

### 4.1 Repository layout

```
FruitFly/
├─ PROJECT_PLAN.md
├─ skeld_map.json                 (given)
├─ cave-secret.json               (given, gitignored)
├─ config/
│  ├─ game.yaml                   timers, speeds, vision, vents, player count
│  ├─ brain.yaml                  LIF params, dt, edge threshold, backend
│  ├─ senses.yaml                 cell-type lists + gains for each encoder
│  └─ motors.yaml                 DN channel definitions + gains
├─ data/  → symlink to C:\flyseek-data (outside OneDrive)
│  ├─ raw/malecns_v1/*.feather
│  └─ cache/ W_csr.pt, neurons.parquet, soma_xyz.npy, groups.json
├─ flyseek/
│  ├─ connectome/   download.py, build_graph.py, celltypes.py, layout.py
│  ├─ brain/        lif_torch.py (batched), subgraph.py, stim.py, record.py
│  ├─ world/        skeld.py (grid, walls, raycast), rules.py (HnS), extras.json
│  ├─ senses/       vision.py, odor.py, touch.py
│  ├─ motors/       decoders.py, body_kinematic.py, body_nmf.py (stretch)
│  ├─ agents/       fly_agent.py, scripted_seeker.py, scripted_hider.py
│  ├─ train/        es.py (CMA-ES/OpenAI-ES), curricula.py, rewards.py
│  ├─ server/       ws_server.py (live), replay_writer.py
│  └─ cli.py        flyseek download | build | bench | walk | train | play | replay
├─ viewer/          Vite + Three.js
│  ├─ src/brainPanel.ts   (IMG_0897 renderer)
│  ├─ src/mapView.ts      (Skeld top-down)
│  └─ src/layout.ts       (seeker left, hiders right, IMG_0898 column)
└─ tests/
```

**Stack:** Python 3.11, PyTorch (CUDA 12.x for the RTX 2060S), pandas/pyarrow, numpy, scipy, `cmaes` or `evosax`, `websockets`, `neuprint-python` (optional), `caveclient` (optional), and pytest. Viewer: TypeScript, Vite, Three.js.

### 4.2 Brain engine (`brain/lif_torch.py`)

* **Model:** the Shiu/flypoke LIF (rest −52 mV, threshold −45 mV, reset −52 mV, τm 20 ms, synaptic τ ~5 ms, refractory 2 ms, delay ≈ 1.8 ms). Weight = synapse count × sign(neurotransmitter) × w0 (0.275 mV).
* **Batched flies:** one CSR weight matrix `W` (float32, on the GPU) shared by everyone. The state `v, g` has shape `[B, N]`, where B = number of flies. All flies step together with one sparse matmul, so **extra flies cost memory, not a second copy of the graph.**
* **Spike-driven propagation:** spikes are sparse. Use `W.T @ spikes` via index gather over only the columns that spiked (or `torch.sparse.mm` if the density is high).
* **Graph sizes** (set in `brain.yaml`):
  * `full`: all MaleCNS edges (~25.6 M). Used for the showcase replay.
  * `pruned5`: edges with ≥ 5 synapses (~6.3 M, as in the Minecraft project). The default.
  * `navcore`: neurons within k hops of the chosen sensory types *and* upstream of the motor DNs (~10–30k neurons). Used for **training**.
* **Time step:** dt = 0.5 ms by default (Shiu uses 0.1 ms). Validate that 0.5 ms reproduces the Shiu sugar → MN9 result. If it doesn't, use 0.25 ms.
* **Clocking:** 1 game tick = 20 ms of brain time (40 brain steps). Senses update once per tick, DN rates come from an exponential moving average (τ ≈ 100 ms), and the body integrates once per tick.
* **Rough budget (estimate, to be confirmed in Milestone 1):**
  * `pruned5`, B = 6, on the 2060S: roughly 1–3 ms per step → 40–120 ms of compute per 20 ms of brain time, which is **2–6× slower than real time**.
  * `full`: roughly 5–10× slower.
  * `navcore`: likely faster than real time, so it's used for live mode and training.
* **Memory:** `full` CSR ≈ 25.6 M × (4 + 4 bytes) + row pointers ≈ 0.25 GB, plus the state. That fits in 8 GB with room to spare. Host RAM (15.6 GB) is the tighter limit when building the graph, so load the Feather file in columns and chunks.

### 4.3 Soma layout for the brain panel (`connectome/layout.py`)

1. Use each neuron's soma location from the annotations.
2. If it's missing, use the centroid of that neuron's synapses (per neuron from neuPrint, or a skeleton root). If that's also missing, place it deterministically near others of the same type and side (this is the "deterministic missing-position layout" from Fly64).
3. Project to a **frontal view** (lateral axis horizontal, dorsal–ventral vertical), so the brain sits above the VNC as in IMG_0897. Normalize to a fixed canvas.
4. Precompute `soma_xy.npy` and `groups.json` (neuron index → display group: vision, pursuit, looming, steering, forward, backward, olfaction, MB, other).

### 4.4 World (`world/skeld.py`)

* Clean the map: snap cells to integer grid coordinates, drop the islands, normalize room names, fill in `Unknown`.
* **Occupancy grid** at 0.15 units (about 275 × 155 cells). Walls are the non-walkable cells. Precompute a **distance transform** for fast collisions and a ray-marching grid.
* `skeld_extras.json` (hand-authored): 14 vents with their link graph, task stations, and Seeker/Hider spawn points.
* **Agent body:** a circle of radius r at `(x, y, θ)`. Motion is clamped with collision sliding along walls.
* **Line of sight:** a DDA ray-cast on the grid, used for vision and for "danger" (distance plus LOS).
* **Rules engine** (`rules.py`): hide phase → seek → Final Hide (speed boost, pings every `ping_interval`), kill radius, task timer reduction, vent uses and timeouts, win check.

### 4.5 Senses → neurons (`senses/`)

| Channel | Game signal | Encoding | Target cells | Who |
|---|---|---|---|---|
| **Vision** | Ray-cast panorama: 64–128 rays over ~270°, each ray returning wall distance and whether it hit a player (and their role) | Per ray: brightness = f(distance), plus a color tag (seeker = red, hider = other). Ray azimuth maps to ommatidial columns (using the 1,771-column map from closed-loop-fly). Poisson rate = gain × value | R1–R6 (brightness), R7/R8 (color) | All |
| **Target feature** | Angular position and size of the nearest visible other player | Rate-coded injection into LC10a on the matching side / retinotopic subset. *Bypass mode*: if the photoreceptor → LC10a path is too weak in the LIF, inject directly (flag it in the UI) | LC10a | Seeker |
| **Looming** | d(angular size)/dt of the Seeker in view | Rate ∝ expansion (LC4) and size (LPLC2) | LC4, LPLC2 | Hiders |
| **Arousal** | Game phase | Tonic Poisson drive, higher in Final Hide | P1 | Seeker |
| **Pings** | Direction to the pinged hider | Left vs right antenna odor ORN rates ∝ cos of bearing | Attractive ORNs (e.g. vinegar-responsive) | Seeker |
| **Danger meter** | Seeker distance (same curve as the game) | Aversive ORN rate | Aversive ORNs (e.g. CO₂/geosmin) | Hiders |
| **Wall touch** | Collision normal (left/right/front) | Short burst | Leg/bristle mechanosensory | All |
| **Reward** (stretch) | Kill / survive / task complete | 200 ms burst (DOOMFLY style) | PAM (reward), PPL1 (punish) | All |

### 4.6 Neurons → movement (`motors/decoders.py`)

```
rate_X   = EMA of spikes/s over all neurons of type X on a side (τ≈100 ms)
forward  = a_f · (rate_oDN1 + rate_DNg100 …) − a_b · rate_MDN
turn     = a_t · ((rate_DNa02_R − rate_DNa02_L) + 0.3 · (rate_DNa01_R − rate_DNa01_L))
dash     = burst(rate_DNp01) > θ  → 0.5 s speed ×1.8 (hider escape)
v        = clamp(v_max · tanh(forward − bias_f), −v_back, v_max)
ω        = clamp(ω_max · tanh(turn), −ω_max, ω_max)
```

* Smoothing, deadband, and clamping are used (as in Fly64) so movement doesn't jitter.
* Every constant (`a_*`, `bias_*`, thresholds) lives in the **trainable adapter** (§5).
* **Vents / tasks / kill:** automatic when inside the interaction radius. Optional stretch goal: gate the kill on a "strike" channel (a DN burst) for fun.

### 4.7 Viewer (IMG_0897 look, IMG_0898 layout)

* **Layout (16:9):** Seeker brain panel large on the left. Skeld map in the center (players, vision radius/cone, danger ring, ping markers, timer). A column of Hider brain panels on the right (up to 5, like the 6 thumbnails in IMG_0898). Click a panel to swap it into the large slot.
* **Brain panel renderer** (Three.js `Points` + custom shader):
  * Base layer: every soma as a tiny grey-white point (alpha ~0.35), with a faint neuropil outline from the ROI meshes.
  * Activity layer: when a neuron spikes, bump its `heat` (a per-vertex attribute texture) and decay it with τ ≈ 300 ms. Point size and brightness scale with heat.
  * **Additive blending** plus a bloom pass (UnrealBloomPass) for the glow. Color comes from the display group (e.g. vision = green, steering/DN = blue, pursuit/P1 = pink, looming/escape = yellow, olfaction = cyan).
  * Header: "166,700 neurons · 25.6 M connections · N descending / motor neurons", plus the fly name and role.
  * Footer: brain time, real-time factor, DN channel bars (forward, turn L/R, back, dash), and superclass Hz (IMG_0899/0900 style).
  * Disclaimer line: *"Connectome-constrained LIF model with engineered sensory/motor mappings — not a validated fly brain."*
* **Data stream:** each 33 ms frame per fly carries a delta-encoded `uint32` list of the neuron indices that spiked (or a `uint8` heat quantized per neuron at low FPS), plus world state. MessagePack over WebSocket. The same format is written to disk as a replay file.
* **Performance:** 166k points × 6 panels is about 1 M points, which is fine for WebGL. The heat texture is updated as one `Float32Array` per panel.

---

## 5. Training plan ("teach them to walk, then to play")

**Principle:** the connectome weights are **frozen**. We train a small **adapter θ** per role:
* **Encoder gains:** one per sensory channel and side.
* **Tonic biases:** e.g. P1 drive level.
* **Decoder weights:** a linear map from a few dozen DN-type rates to `(forward, turn, dash)`.
* **Smoothing constants.**

That's about 20–200 parameters. The spiking model isn't differentiable in a clean way, so we use **evolution strategies** (CMA-ES or OpenAI-ES) rather than backprop. We evaluate the population *in parallel as the GPU batch dimension*. PPO on the readout (as in fly-craftax) is a backup option.

### Stage W0 — "Is the brain wired right?" (sanity checks, no training)
* Reproduce Shiu: sugar GRN stimulation → MN9 fires, and bitter suppresses it.
* Stimulate DNa02‑L alone → the kinematic body turns left. oDN1/DNg100 → forward. MDN → backward. DNp01 → dash.
* Put LC10a stimulation on one side → measure the DNa02 L/R asymmetry. **This is the key check that the pursuit path carries a signal through the model.** If it doesn't, move the injection point downstream (AOTU019/025) and document it.
* Looming on LC4/LPLC2 → the DNp01 rate rises.

### Stage W1 — "Walk" (open arena, empty 10×10 room)
* Task: drive forward without spinning in place, and turn toward a light spot on the left or right (**phototaxis**, using the vision encoder).
* Reward: distance traveled + time with heading aligned to the target − time stuck on walls.
* Output: a base adapter θ₀ shared by both roles.

### Stage W2 — "Coordinate around the map" (Skeld navigation curriculum)
1. Single room (Cafeteria): wall avoidance, exploring coverage.
2. Room + corridor: reach a beacon in the next room (beacon = attractive odor gradient + a visual marker).
3. Full Skeld: **coverage-exploration reward** (new grid cells visited per minute) + reaching random beacons in other rooms.
* The domain randomization includes spawn point, heading, and Poisson noise seeds.
* Success criterion: reach a random room within 90 s of game time in ≥ 60% of trials, averaging < 10% of time stuck on walls.

### Stage R1 — Role training against scripted opponents
* **Seeker adapter θ_S** (starting from θ₀). The opponent is a scripted hider that runs away along the distance field (with speed noise).
  * Reward: +10 per kill, − time to first kill, + (coverage during the hide phase). Final Hide adds ping-following.
* **Hider adapter θ_H** (starting from θ₀). The opponent is a scripted Seeker with an A* chase once in LOS, patrolling otherwise.
  * Reward: + seconds survived, − danger-meter integral, + task completions. Vents are used automatically when they're adjacent and the danger is high.

### Stage R2 — Self-play league
* Alternate: freeze θ_H → run ES on θ_S for K generations → freeze θ_S → run ES on θ_H. Keep a **hall of fame** of past adapters to avoid cycling.
* Run on `navcore` for speed. Every M generations, **validate the best on `pruned5` / `full`** to confirm the behavior transfers to the full brain.

### Stage S (stretch goals) — make it real "learning" in the brain
* **Mushroom-body plasticity:** dopamine-gated KC→MBON weight changes (PAM = reward on kill/task, PPL1 = punishment on getting caught or danger spikes). DOOMFLY and TheMrRaGe/flybrain are the references. Use **held-out validation gates** (DOOMFLY's own candidate failed its gates, so be skeptical of results).
* **Central complex goal memory:** inject heading into EPG and let FC2 hold a "favorite hiding spot" direction, with PFL3 → steering.
* **NeuroMechFly body:** feed `(forward, turn)` into FlyGym's hybrid turning controller for one "hero" fly and render it in a small 3D inset.

### Compute budget (estimates, to be revised after benchmarking)
| Stage | Graph | Population × episodes | Episode length | Estimated wall time on 2060S |
|---|---|---|---|---|
| W1 | navcore | 16 × 200 gens | 20 s | ~hours |
| W2 | navcore | 16 × 300 gens | 90 s | ~1–2 days |
| R1 (each role) | navcore | 12 × 300 gens | 180 s | ~2–3 days |
| R2 | navcore (+ periodic pruned5 checks) | 12 × open-ended | 300 s | ongoing |
| Showcase match | full, B = 6 | 1 match | 5 min game | ~30–60 min compute → replay |

If this is too slow, options are cloud GPU hours (an A100 is about 10× faster), a smaller `navcore`, or a larger dt for training only.

---

## 6. Milestones & acceptance criteria

| # | Milestone | Deliverable | Done when |
|---|---|---|---|
| **M0** | Setup | Repo skeleton, venv, CUDA PyTorch, data dir outside OneDrive, `.gitignore` for secrets | `python -c "import torch; print(torch.cuda.is_available())"` prints True |
| **M1** | Connectome ingest + benchmark | `flyseek download/build/bench`; CSR caches; cell-type lookup report (`celltypes_report.md`) listing the resolved IDs for every type in §3.6 | All required types resolved (or flagged). Steps/sec measured for full/pruned5/navcore × B = 1, 6 |
| **M2** | Brain panel (static + synthetic spikes) | Viewer showing 1 and then 6 panels in the IMG_0897 style from `soma_xy.npy` | Screenshot side by side with IMG_0897 looks right. 60 FPS with 6 panels |
| **M3** | Sanity circuits (W0) | Notebook plus recorded replays of sugar → MN9, DNa02 turn, LC10a → DNa02 asymmetry, looming → DNp01 | Every check passes, or has a documented workaround |
| **M4** | Skeld world + scripted agents | Clean grid, extras file, rules engine, scripted seeker vs hiders, map view in the viewer | Full scripted match plays to a win condition |
| **M5** | Closed loop, one fly walking (W1) | Fly moves in an arena from its own DN output, with a live brain panel | Phototaxis success ≥ 80% |
| **M6** | Skeld navigation (W2) | θ₀ adapter checkpoint + evaluation report | Criteria in W2 met |
| **M7** | Role training (R1 → R2) | θ_S, θ_H checkpoints, win-rate curves | Trained Seeker beats scripted hiders more often than a random-walk seeker. Trained Hider outlives the random baseline |
| **M8** | Showcase | 1 Seeker + 5 Hiders on the full graph, recorded replay, polished viewer, 60–90 s video capture | You're happy with it 🎉 |
| M9 (stretch) | Plasticity / central complex / NeuroMechFly inset | — | — |

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Sensory signals die out before reaching the DNs in a plain LIF model (the "decorative vision" problem Eon reported) | **High** | W0 checks first. Allow injection downstream (LC10a/LC4 or AOTU) with a visible "bypass" flag in the UI. Tune encoder gains inside the adapter |
| Not real time on the 2060S | High | Game clock slaved to brain time, record → replay, `navcore` for live mode |
| Cell-type names differ in MaleCNS v1.0 | Medium | Resolver with fuzzy matching plus a manual override file. Fail loudly |
| RAM (15.6 GB) during graph build | Medium | Load the Feather file by column and in chunks, keep only (pre, post, weight), store IDs as int32 indices |
| OneDrive syncing gigabytes of data | High | `data/` outside OneDrive |
| "Training" gets over-claimed | — | Disclaimer in the UI. Always compare against random and scripted baselines. The adapter is openly separate from the frozen connectome |
| Token leak | Low | Gitignore, env var, never logged. Not needed for the main path anyway |
| Among Us IP | Low | Personal, non-commercial fan project. Use our own simple sprites and the map geometry only. Don't redistribute game assets |
| Ethics / optics ("disturbing") | — | This is a simplified model and not a living or conscious fly. Say that clearly in videos |

---

## 8. First concrete steps (the first session of coding)

1. Create the repo skeleton from §4.1, a venv, and install PyTorch with CUDA.
2. Download the three required MaleCNS Feather files plus neuropil meshes into `C:\flyseek-data\raw\malecns_v1\`.
3. `build_graph.py`: create the neuron index, sign from the neurotransmitter predictions, and CSR caches for full and pruned5.
4. `celltypes.py`: resolve every type in §3.6 and write `celltypes_report.md`.
5. `lif_torch.py`: batched LIF, then `flyseek bench` on the 2060S.
6. `layout.py` → `soma_xy.npy`, then build the first brain panel in the viewer with synthetic random spikes (the quickest visual win).

---

## 9. Sources

**Datasets & official**
- [MaleCNS connectome](https://male-cns.janelia.org/) · [Download page](https://male-cns.janelia.org/download/) · [Release notes](https://male-cns.janelia.org/release/) · [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome) · [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/)
- [Sexual dimorphism in the complete Drosophila male CNS connectome (Cell)](https://www.sciencedirect.com/science/article/pii/S0092867426009426)
- [FlyWire](https://flywire.ai/) · [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) · [CAVEclient](https://github.com/CAVEconnectome/CAVEclient) · [FlyWire CAVE tutorial](https://github.com/seung-lab/FlyConnectome/blob/main/CAVE%20tutorial.ipynb) · [fafbseg FlyWire setup](https://fafbseg-py.readthedocs.io/en/stable/source/tutorials/flywire_setup.html)
- [HF: svgmediabills/malecns-connectome](https://huggingface.co/datasets/svgmediabills/malecns-connectome) · [HF Space: fly-sim](https://huggingface.co/spaces/rf223x/fly-sim) · [HF: fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) · [HF: gpt-fly](https://huggingface.co/eob/gpt-fly)

**Brain models & simulators**
- [Shiu et al. — LIF model of the whole Drosophila brain (PubMed)](https://pubmed.ncbi.nlm.nih.gov/37205514/) · [code](https://github.com/philshiu/Drosophila_brain_model)
- [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) · [flypoke](https://github.com/vshapenko/flypoke) · [webgpu-fly](https://github.com/abgnydn/webgpu-fly) · [closed-loop-fly](https://github.com/ZeroXClem/closed-loop-fly) · [TheMrRaGe/flybrain](https://github.com/TheMrRaGe/flybrain) · [State of Brain Emulation Report 2025](https://arxiv.org/pdf/2510.15745)

**Embodiment**
- [Eon: How the Eon Team Produced a Virtual Embodied Fly](https://eon.systems/updates/embodied-brain-emulation) · [Carboncopies: "No, a fruit fly has not been uploaded"](https://carboncopies.org/Blog/Posts/FruitFlyNotUploaded/Post/)
- [NeuroMechFly v2 (Nature Methods)](https://www.nature.com/articles/s41592-024-02497-y) · [FlyGym](https://github.com/NeLy-EPFL/flygym/) · [NeuroMechFly tutorials](https://neuromechfly.org/tutorials/)
- [Whole-body simulation of fruit fly locomotion (Nature)](https://www.nature.com/articles/s41586-025-09029-4) · [flybody in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/flybody/README.md)

**Game projects**
- [awesome-fly index](https://github.com/cobanov/awesome-fly) · [DOOMFLY](https://github.com/nftechie/doomfly) · [Fly64 (Mario)](https://github.com/ornata/fly) · [Fly Brain Minecraft](https://github.com/blendi-remade/fly-brain-minecraft) · [fly-craftax](https://github.com/liuzihe02/fly-craftax)
- [Tom's Hardware coverage](https://www.tomshardware.com/software/programming/google-maps-entire-brain-and-central-nervous-system-of-adult-male-fruit-fly-software-engineers-immediately-make-it-run-doom-ai-powered-3d-model-of-over-166-000-neurons-can-also-play-super-mario-64) · [Gizmodo](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616) · [Know Your Meme: Fruit Fly Brain Simulations](https://knowyourmeme.com/memes/fruit-fly-brain-simulations)

**Neuroscience (circuits used)**
- [Fine-grained descending control of steering (DNa01/DNa02)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614758/) · [Neural circuit mechanisms for steering control (eLife)](https://elifesciences.org/articles/102230)
- [MANC descending → motor organization (eLife)](https://elifesciences.org/articles/96084) · [Moonwalker SEZ neurons & backward locomotion](https://pubmed.ncbi.nlm.nih.gov/35139358/)
- [Sexual arousal gates visual processing during courtship (P1/LC10a, Nature)](https://www.nature.com/articles/s41586-021-03714-w) · [Parallel pathways for visual object pursuit (Neuron)](https://www.cell.com/neuron/fulltext/S0896-6273(26)00001-2)
- [LC4/LPLC2 looming → giant fiber](https://www.sciencedirect.com/science/article/pii/S0960982219301381) · [Visual projection neurons link features to behaviors](https://pmc.ncbi.nlm.nih.gov/articles/PMC5293491/)
- [Allocentric goal → egocentric steering (FC2/PFL3, Nature)](https://www.nature.com/articles/s41586-023-07006-3) · [Head direction → goal steering (Nature)](https://www.nature.com/articles/s41586-024-07039-2)

**Among Us**
- [Hide n Seek (Among Us Wiki)](https://among-us.fandom.com/wiki/Hide_n_Seek) · [Innersloth: Hide n Seek announcement](https://www.innersloth.com/new-game-mode-hide-n-seek-is-here-emergency-meeting-35/) · [The Skeld (wiki)](https://among-us.fandom.com/wiki/The_Skeld) · [Vision (wiki)](https://among-us.fandom.com/wiki/Vision)
