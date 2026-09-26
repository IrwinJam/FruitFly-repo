#!/usr/bin/env bash
# Obstacle-sense experiment: choose the visual cell type that steers hardest (checked against shuffled
# wiring), retrain the walker with that sense and a wall-contact penalty, and evaluate it with the sense
# switched off and the cells silenced. Run after run_pipeline.sh (it starts from explore_v5).
set -u
cd "$(dirname "$0")/../.."
PY=${AMONGUSFLY_PY:-.venv/Scripts/python}
LOG=${AMONGUSFLY_DATA:-/c/amongusfly-data}/obstacle_sense.log
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
need() { [ ! -f "docs/results/$1.json" ]; }
R=results

stage "1 which visual cell type steers"
need obstacle_channel_selection && $PY -m amongusfly.experiments.obstacle_channel --out $R/obstacle_channel_selection >> "$LOG" 2>&1
need obstacle_channel_shuffled && $PY -m amongusfly.experiments.obstacle_channel --graph navcore_shuf0 --only LLPC1 LC10a \
  --out $R/obstacle_channel_shuffled >> "$LOG" 2>&1

stage "2 walker with the obstacle sense (contact penalty 0.15 per second)"
$PY -m amongusfly.train.es --run explore_v6 --graph navcore --policy route_obs --init-from explore_v5 \
  --seconds 90 --generations 20 --contact-penalty 0.15 >> "$LOG" 2>&1

stage "3 evaluation with ablations"
need obstacle_sense_navigation && $PY -m amongusfly.train.eval_explore --policy route_obs --n 40 \
  walker_v5=navcore:best:explore_v5 walker_v6=navcore:best:explore_v6 sense_off=navcore:best:explore_v6:NOSENSE \
  llpc1_off=navcore:best:explore_v6:LLPC1 pfl3_off=navcore:best:explore_v6:PFL3 --out $R/obstacle_sense_navigation >> "$LOG" 2>&1
need obstacle_sense_walking && $PY -m amongusfly.experiments.locomotion_quality --run explore_v6 --n 24 --seconds 120 \
  --conditions "sense_on=2.0:repel=1.5" "sense_off=2.0:repel=1.5:obstacle=0" --out $R/obstacle_sense_walking >> "$LOG" 2>&1
need obstacle_sense_seeker && $PY -m amongusfly.train.eval_role --role seeker --preset short --n 40 \
  walker_v5=brain:navcore:init:explore_v5 walker_v6=brain:navcore:init:explore_v6 \
  llpc1_off=brain:navcore:init:explore_v6:LLPC1 scripted=scripted --out $R/obstacle_sense_seeker >> "$LOG" 2>&1
stage "=== OBSTACLE SENSE DONE ==="
