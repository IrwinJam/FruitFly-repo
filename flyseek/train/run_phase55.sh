#!/usr/bin/env bash
# Phase 5.5 pipeline (2026-09-18): re-train everything on the corrected locomotion
# (max turn rate 6.0 rad/s, config/motors.yaml) and re-run every control.
# Resumable: training continues from checkpoints, finished evaluations are skipped.
# Log: C:\flyseek-data\phase55.log   Summary: docs/PHASE5_OVERNIGHT.md
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=/c/flyseek-data/phase55.log
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
report() { $PY -m flyseek.train.phase5_report >> "$LOG" 2>&1; }
need() { [ ! -f "docs/$1.json" ]; }

stage "1 explorer re-training on fixed locomotion"
$PY -m flyseek.train.es --run explore_navcore_v2 --graph navcore --policy route --generations 20 >> "$LOG" 2>&1
stage "1b shuffled-wiring explorer control"
$PY -m flyseek.train.es --run explore_shuf0_v2 --graph navcore_shuf0 --policy route --generations 20 >> "$LOG" 2>&1
report

stage "2 held-out exploration evaluation (fixed locomotion)"
need phase5_eval_explore_omega6 && $PY -m flyseek.train.eval_explore \
  untrained=navcore:init \
  trained=navcore:best:explore_navcore_v2 \
  trained_mean=navcore:mean:explore_navcore_v2 \
  old_adapter=navcore:best:explore_navcore \
  shuf_trained=navcore_shuf0:best:explore_shuf0_v2 \
  pfl3_off=navcore:best:explore_navcore_v2:PFL3 \
  --out phase5_eval_explore_omega6 >> "$LOG" 2>&1
report

stage "3 seeker role training (fixed locomotion)"
$PY -m flyseek.train.es --run seeker_v3 --graph navcore --policy seeker --init-from explore_navcore_v2 --pop 8 --seeds-per-candidate 8 --generations 20 --sigma 0.15 >> "$LOG" 2>&1
report
stage "4 hider role training (fixed locomotion)"
$PY -m flyseek.train.es --run hider_v2 --graph navcore --policy hider --init-from explore_navcore_v2 --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1
report

stage "5 held-out seeker evaluation + ablations"
need phase5_eval_seeker_v3 && $PY -m flyseek.train.eval_role --role seeker \
  explorer=brain:navcore:init:explore_navcore_v2 \
  trained_best=brain:navcore:best:seeker_v3 \
  trained_mean=brain:navcore:mean:seeker_v3 \
  random=random scripted=scripted \
  lc10a_off_trained=brain:navcore:best:seeker_v3:LC10a \
  pfl3_off_trained=brain:navcore:best:seeker_v3:PFL3 \
  --out phase5_eval_seeker_v3 >> "$LOG" 2>&1
report
stage "6 held-out hider evaluation + ablations"
need phase5_eval_hider_v2 && $PY -m flyseek.train.eval_role --role hider \
  explorer=brain:navcore:init:explore_navcore_v2 \
  trained_best=brain:navcore:best:hider_v2 \
  trained_mean=brain:navcore:mean:hider_v2 \
  random=random scripted=scripted \
  loom_off_trained=brain:navcore:best:hider_v2:LC4,LPLC2 \
  dnp01_off_trained=brain:navcore:best:hider_v2:DNp01 \
  pfl3_off_trained=brain:navcore:best:hider_v2:PFL3 \
  --out phase5_eval_hider_v2 >> "$LOG" 2>&1
report

stage "7 shuffled-wiring role controls (same budget)"
$PY -m flyseek.train.es --run seeker_shuf0_v3 --graph navcore_shuf0 --policy seeker --init-from explore_shuf0_v2 --pop 8 --seeds-per-candidate 8 --generations 20 --sigma 0.15 >> "$LOG" 2>&1
$PY -m flyseek.train.es --run hider_shuf0_v2 --graph navcore_shuf0 --policy hider --init-from explore_shuf0_v2 --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1
report
stage "8 held-out shuffle comparisons"
need phase5_eval_seeker_shuffle_v3 && $PY -m flyseek.train.eval_role --role seeker \
  explorer=brain:navcore:init:explore_navcore_v2 \
  trained_best=brain:navcore:best:seeker_v3 \
  shuf_trained=brain:navcore_shuf0:best:seeker_shuf0_v3 \
  --out phase5_eval_seeker_shuffle_v3 >> "$LOG" 2>&1
need phase5_eval_hider_shuffle_v2 && $PY -m flyseek.train.eval_role --role hider \
  explorer=brain:navcore:init:explore_navcore_v2 \
  trained_best=brain:navcore:best:hider_v2 \
  shuf_trained=brain:navcore_shuf0:best:hider_shuf0_v2 \
  --out phase5_eval_hider_shuffle_v2 >> "$LOG" 2>&1
report
stage "=== PHASE 5.5 PIPELINE DONE ==="
