# FlySeek — build status

*Last updated: 2026-09-13, after the first implementation session.*

This tracks progress against the milestones in [PROJECT_PLAN.md](PROJECT_PLAN.md#6-milestones--acceptance-criteria).
Everything below is real, run, and verified on this machine — no placeholders or
mocked numbers except where explicitly labeled "synthetic"/"demo".

## Done

### M0 — Setup ✅
- Repo skeleton, git initialized, `.gitignore` (secrets, data, venv, build output).
- `data/` is a directory junction to `C:\flyseek-data` (outside OneDrive) so the
  multi-GB connectome never syncs.
- Python 3.11 venv with PyTorch 2.5.1+cu121 — confirmed CUDA available on the RTX
  2060 Super. All other deps (pandas, pyarrow, scipy, etc.) installed and verified.
- Node v24 / npm 11 confirmed for the viewer.

### M1 — Connectome ingest + benchmark ✅
- Downloaded and verified (exact byte counts) all 3 required MaleCNS v1.0 files
  from the public Janelia/Google bucket (no auth needed) — 1.05 GB total.
  [flyseek/connectome/download.py](flyseek/connectome/download.py)
- **Resolved all 20 sensory/motor roles** against real `type` names in the
  annotations table. [flyseek/connectome/celltypes.py](flyseek/connectome/celltypes.py),
  report at [docs/celltypes_report.md](docs/celltypes_report.md). Notable
  corrections found during resolution (documented in the module docstring):
  - **`oDN1` does not exist in MaleCNS** — it's specific to Eon's *female* FlyWire
    model. Substituted `DNg100`/`DNp09` for forward drive.
  - **P1 resolves to `pIP1`** via synonym cross-reference to the courtship-arousal
    literature, not an exact `type` match — flagged uncertain.
  - R1–R6 photoreceptors are one pooled type; R7/R8 split into pale/yellow/
    dorsal-rim subtypes.
  - No gustatory GRNs at this confidence threshold; danger/ping odor channels use
    literature-identified ORN glomeruli instead (documented in the code).
- **Built the graph** and cross-validated it against independently published
  numbers: [flyseek/connectome/build_graph.py](flyseek/connectome/build_graph.py)
  - `full` (traced-to-traced): **165,122 neurons, 25,563,197 edges** — matches the
    "166,700 neurons / 25,582,938 connections" figures reported by community
    MaleCNS projects (DOOMFLY, ornata/fly) almost exactly.
  - `pruned5` (≥5 synapses): **6,235,682 edges** — matches the "6.29M connections"
    figure reported by the fly-brain-minecraft project.
  - This agreement across three independent sources (our pipeline, DOOMFLY,
    fly-brain-minecraft) is a strong correctness signal for the ingest code.
- **Batched GPU LIF simulator**: [flyseek/brain/lif_torch.py](flyseek/brain/lif_torch.py).
  Smoke-tested (not just benchmarked): forcing strong drive on one neuron produces
  spiking that cascades to 25 distinct downstream neurons via the real synaptic
  graph, confirming propagation is wired correctly.
- **Benchmarked** on the RTX 2060 Super: [docs/bench_results.json](docs/bench_results.json)

  | graph | batch | steps/sec | ms/step | real-time factor |
  |---|---|---|---|---|
  | pruned5 | 1 fly | 928.6 | 1.08 | 0.46× |
  | pruned5 | 6 flies | 302.4 | 3.31 | 0.15× |
  | full | 1 fly | 514.1 | 1.95 | 0.26× |
  | full | 6 flies | 95.8 | 10.44 | 0.048× |

  Confirms the plan's record-then-replay approach (section 4.2) is necessary, not
  optional — even the smallest configuration doesn't reach real time.

### M2 — Brain panel viewer ✅
- [flyseek/connectome/layout.py](flyseek/connectome/layout.py): soma positions
  projected to 2D using axis analysis of the *real* coordinate data (not assumed) —
  `soma_x` is left-right, `soma_z` (not `soma_y`) is the head-to-tail axis that
  separates brain from VNC by far the widest margin. Missing soma positions
  (~15% overall, ~100% of photoreceptors/ORNs) filled via deterministic
  donor-based scatter, seeded from bodyId so it's stable across rebuilds.
  Static preview: [docs/brain_panel_preview.png](docs/brain_panel_preview.png)
  — compare against `Untitled_Message/IMG_0897.jpeg`.
- **Working Three.js viewer** at [viewer/](viewer/): 1 large Seeker panel + 5 Hider
  panels (IMG_0898 layout), additive-blended point-cloud brain rendering with
  per-neuron "heat" glow (spike → bright colored flash → fades over ~300ms),
  headers with live neuron/synapse counts, DN channel bars, and an explicit
  on-screen disclaimer.
  - Fixed a real bug during testing: Three's `ShaderMaterial` always declares a
    built-in `vec3 position` attribute; the exported 2-component soma coordinates
    had to be expanded to 3 components (z=0) to match, or WebGL silently failed to
    link a usable program.
  - Tuned the fragment shader's resting alpha down (0.12 → 0.035) after the first
    render came out blown-out white — with additive blending across ~165k
    overlapping points, even a modest per-point alpha saturates fast.
  - **Measured 144.5 FPS with all 6 panels live** on this machine (target was 60).
  - **Run it yourself:** `cd viewer && npm install && npm run dev`, or ask me to
    open it via the preview tool. Currently shows **synthetic/demo activity**
    (randomized, role-biased spike bursts) — it is NOT yet wired to the real brain
    simulator or game world. That's the honest state; see "Not done" below.

## Not done yet

Everything past M2 in the plan is unbuilt. In particular, **no training has
happened** — the plan's Stage W0–S training schedule is measured in hours to days
of GPU time even on `navcore`-sized subgraphs, which doesn't fit in an interactive
session. Concretely, still open:

- **M3 — Sanity circuits (W0)**: haven't yet verified the pursuit path (LC10a →
  DNa02 asymmetry) or looming path (LC4/LPLC2 → DNp01) actually carry a usable
  signal through the LIF model. This is the single biggest open risk in the plan
  (section 7) — vision/sensory signals dying out before reaching motor neurons is
  a documented failure mode in Eon's own writeup.
- **M4 — Skeld world**: map cleanup (room-name normalization, dropping the 2 tiny
  disconnected islands), vents/tasks/spawns (`skeld_extras.json`), the rules
  engine, and the map view in the viewer are all unbuilt.
- **M5–M7 — closed-loop walking, navigation, and role training**: none of the
  sensory encoders or motor decoders from `config/senses.yaml`/`config/motors.yaml`
  are wired to the actual LIF simulator yet — those configs currently describe
  the intended design, not running code. No adapters have been trained.
- **`flyseek/server/ws_server.py`** (live brain → viewer streaming) doesn't exist;
  the viewer's "live" data is a synthetic stand-in, clearly labeled as such on
  screen.
- **`config/motors.yaml`** and **`config/game.yaml`** (referenced in the plan)
  haven't been written yet.

## Honest assessment

The parts that could be *verified against external ground truth* — the data
download, the edge counts, the FPS, the GPU throughput — all check out and two of
them (edge counts) were independently cross-validated against numbers other
projects have published. The parts that are inherently unverifiable without
running them (whether the sensory pathways actually carry signal, whether trained
adapters actually learn to navigate) are exactly the parts marked "not done" above
— they're the real, hard, multi-day parts of this project, not skipped because
they were easy.

## Next session's concrete first steps

1. Run the M3 sanity checks from the plan (sugar→MN9, DN stimulation → kinematic
   turn, LC10a→DNa02 asymmetry, looming→DNp01) using `LIFBrain` directly. This is
   the highest-priority next step — everything downstream depends on knowing
   whether the direct-injection approach works or whether sensory input needs to
   be injected further downstream (the `bypass_downstream` flag already stubbed
   in `config/senses.yaml`).
2. Write `flyseek/world/skeld.py`: clean the map, build the occupancy grid +
   distance transform, author `skeld_extras.json` (vents/tasks/spawns).
3. Wire one sensory encoder (vision) and one motor decoder (steering) end-to-end
   on a single fly in an empty arena — the first real closed loop.
