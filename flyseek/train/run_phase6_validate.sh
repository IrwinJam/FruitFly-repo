#!/usr/bin/env bash
# Phase 6.1 / 6.2 validation (GAMEPLAN Phase 6). Held-out seeds 90000+; finished
# evaluations are skipped on re-runs. Log: C:\flyseek-data\phase6.log
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=/c/flyseek-data/phase6.log
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
need() { [ ! -f "docs/$1.json" ]; }
S=init:explore_navcore_v2          # seeker adapter (best held-out seeker)
HT=mean:hider_v2                   # trained hider adapter
HU=init:explore_navcore_v2         # hider without role training
SHUF=brain:navcore_shuf0:init:explore_shuf0_v2:mean:hider_shuf0_v2

stage "6.1B all-brain, 3 hiders, short preset (+ controls D)"
need phase6_allbrain3_short && $PY -m flyseek.train.eval_role --role both --preset short --n 40 \
  allbrain=brain:navcore:$S:$HT \
  untrained_hiders=brain:navcore:$S:$HU \
  pfl3_off=brain:navcore:$S:$HT:PFL3 \
  shuffled=$SHUF \
  --out phase6_allbrain3_short >> "$LOG" 2>&1

stage "6.1C all-brain, 5 hiders, short preset"
need phase6_allbrain5_short && $PY -m flyseek.train.eval_role --role both --preset short --n 40 --hiders 5 \
  allbrain=brain:navcore:$S:$HT \
  untrained_hiders=brain:navcore:$S:$HU \
  --out phase6_allbrain5_short >> "$LOG" 2>&1

stage "6.1B all-brain, 3 hiders, full-length preset"
need phase6_allbrain3_full && $PY -m flyseek.train.eval_role --role both --preset full --n 20 \
  allbrain=brain:navcore:$S:$HT \
  pfl3_off=brain:navcore:$S:$HT:PFL3 \
  --out phase6_allbrain3_full >> "$LOG" 2>&1

stage "6.1C all-brain, 5 hiders, full-length preset"
need phase6_allbrain5_full && $PY -m flyseek.train.eval_role --role both --preset full --n 20 --hiders 5 \
  allbrain=brain:navcore:$S:$HT \
  --out phase6_allbrain5_full >> "$LOG" 2>&1

stage "6.1A full-length preset vs scripted opponents"
need phase6_seeker_full && $PY -m flyseek.train.eval_role --role seeker --preset full --n 20 \
  brain=brain:navcore:$S scripted=scripted --out phase6_seeker_full >> "$LOG" 2>&1
need phase6_hider_full && $PY -m flyseek.train.eval_role --role hider --preset full --n 20 \
  untrained=brain:navcore:$HU trained=brain:navcore:$HT scripted=scripted --out phase6_hider_full >> "$LOG" 2>&1

stage "6.2 full-graph gate: exploration transfer navcore -> full (16 flies, 45 s)"
need phase6_full_graph_gate && $PY -m flyseek.train.eval_explore --n 16 \
  navcore=navcore:best:explore_navcore_v2 \
  full=full:best:explore_navcore_v2 \
  --out phase6_full_graph_gate >> "$LOG" 2>&1
stage "=== PHASE 6 VALIDATION DONE ==="
