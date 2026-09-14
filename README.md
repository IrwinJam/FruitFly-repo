# FlySeek (FruitFly Amogus)

This project uses a connectome-constrained model of the fruit fly central nervous system (MaleCNS v1.0) to play Among Us Hide n Seek on The Skeld. Each fly gets a live brain-activity panel.

- **Design and research:** [PROJECT_PLAN.md](PROJECT_PLAN.md)
- **Roadmap:** [GAMEPLAN.md](GAMEPLAN.md)
- **What's built so far:** [STATUS.md](STATUS.md)

> Status: an early research prototype. This is a simplified leaky integrate-and-fire (LIF) model with engineered sensory and motor mappings. It is **not** a validated or living fly brain.

## Reproducing from a fresh clone

The connectome data (~1.4 GB including caches) is **not stored in this repo**. It is public and gets rebuilt with the scripts below. On Windows, run these from the repo root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
.venv/Scripts/python -m pip install -r requirements.txt

# data lives outside the repo; paths currently point at C:\flyseek-data
.venv/Scripts/python -m flyseek.connectome.download        # ~1.05 GB from Janelia's public bucket
.venv/Scripts/python -m flyseek.connectome.celltypes       # resolve sensory/motor roles
.venv/Scripts/python -m flyseek.connectome.build_graph     # full + pruned5 graphs
.venv/Scripts/python -m flyseek.connectome.layout          # 2D soma layout
.venv/Scripts/python -m flyseek.connectome.export_viewer_data
.venv/Scripts/python -m flyseek.world.skeld                # Skeld grid + vents

.venv/Scripts/python -m flyseek.brain.bench                # GPU benchmark
```

Viewer:

```bash
cd viewer && npm install && npm run dev
```

## Data and attribution

- **MaleCNS v1.0 connectome:** HHMI Janelia FlyEM, Google Research, and the University of Cambridge. Licensed CC-BY 4.0. <https://male-cns.janelia.org/>
- **Among Us and The Skeld:** © Innersloth. This is a non-commercial fan research project, and no game assets are distributed. `skeld_map.json` holds walkable-area geometry only.
- **Design reference screenshots** (IMG_0897–0902, mentioned in the plan) are other creators' content. They are kept locally only and not included in this repo.
