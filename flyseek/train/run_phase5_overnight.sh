#!/usr/bin/env bash
# Phase 5 overnight pipeline (2026-09-14). Every stage is resumable/idempotent: training
# runs continue from their checkpoints, so re-running this script picks up where it stopped.
# Progress: C:\flyseek-data\phase5_overnight.log ; summary: docs/PHASE5_OVERNIGHT.md
#
#  0. wait for the already-running seeker evaluation and hider_navcore training
#  1. finish hider_navcore (no-op if complete)
#  2. held-out hider evaluation (+ circuit ablations)
#  3. seeker branch: if seeker_navcore beat the Phase 4 explorer -> ablations on it;
#     otherwise retrain with less noisy fitness (8 matches per candidate) = seeker_navcore_v2, then evaluate
#  4. hider_shuf0 control (same budget as hider_navcore), then its evaluation
#  5. seeker shuffle control for whichever seeker setup is current (continues into the morning)
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=/c/flyseek-data/phase5_overnight.log
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
report() { $PY -m flyseek.train.phase5_report >> "$LOG" 2>&1; }
wait_pid() { while tasklist //FI "PID eq $1" 2>/dev/null | grep -q " $1 "; do sleep 60; done; }

stage "0 waiting for running seeker eval (pid ${EVAL_PID:-none}) and hider_navcore training (pid ${TRAIN_PID:-none})"
[ -n "${EVAL_PID:-}" ] && wait_pid "$EVAL_PID"
[ -n "${TRAIN_PID:-}" ] && wait_pid "$TRAIN_PID"
report

stage "1 hider_navcore (resume/finish)"
$PY -m flyseek.train.es --run hider_navcore --graph navcore --policy hider --init-from explore_navcore --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1
report

stage "2 held-out hider evaluation"
$PY -m flyseek.train.eval_role --role hider \
  explorer=brain:navcore:init:explore_navcore \
  trained_best=brain:navcore:best:hider_navcore \
  trained_mean=brain:navcore:mean:hider_navcore \
  random=random scripted=scripted \
  loom_off_trained=brain:navcore:mean:hider_navcore:LC4,LPLC2 \
  dnp01_off_trained=brain:navcore:mean:hider_navcore:DNp01 \
  pfl3_off_trained=brain:navcore:mean:hider_navcore:PFL3 \
  --out phase5_eval_hider >> "$LOG" 2>&1
report

if [ ! -f docs/phase5_eval_seeker.json ]; then
  stage "3a seeker evaluation missing, running it"
  $PY -m flyseek.train.eval_role --role seeker explorer=brain:navcore:init:explore_navcore trained_best=brain:navcore:best:seeker_navcore trained_mean=brain:navcore:mean:seeker_navcore random=random scripted=scripted lc10a_off=brain:navcore:init:explore_navcore:LC10a pfl3_off=brain:navcore:init:explore_navcore:PFL3 >> "$LOG" 2>&1
fi
DEC=$($PY -m flyseek.train.phase5_report --decide phase5_eval_seeker 2>/dev/null | tail -1)
SEEKER_RUN=seeker_navcore
SHUF_SEEKER_RUN=seeker_shuf0
SEEKER_ARGS="--pop 12 --seeds-per-candidate 4 --generations 25"
if [ "$DEC" = "ok" ]; then
  stage "3 seeker_navcore beat the explorer -> ablations on the trained seeker"
  $PY -m flyseek.train.eval_role --role seeker \
    explorer=brain:navcore:init:explore_navcore \
    trained_mean=brain:navcore:mean:seeker_navcore \
    lc10a_off_trained=brain:navcore:mean:seeker_navcore:LC10a \
    pfl3_off_trained=brain:navcore:mean:seeker_navcore:PFL3 \
    --out phase5_eval_seeker_trained_ablations >> "$LOG" 2>&1
else
  stage "3 seeker_navcore did NOT beat the explorer (decision=$DEC) -> seeker_navcore_v2 (8 matches per candidate)"
  SEEKER_RUN=seeker_navcore_v2
  SHUF_SEEKER_RUN=seeker_shuf0_v2
  SEEKER_ARGS="--pop 8 --seeds-per-candidate 8 --generations 20 --sigma 0.15"
  $PY -m flyseek.train.es --run seeker_navcore_v2 --graph navcore --policy seeker --init-from explore_navcore $SEEKER_ARGS >> "$LOG" 2>&1
  report
  stage "3b held-out evaluation of seeker_navcore_v2"
  $PY -m flyseek.train.eval_role --role seeker \
    explorer=brain:navcore:init:explore_navcore \
    trained_best=brain:navcore:best:seeker_navcore_v2 \
    trained_mean=brain:navcore:mean:seeker_navcore_v2 \
    lc10a_off_trained=brain:navcore:mean:seeker_navcore_v2:LC10a \
    pfl3_off_trained=brain:navcore:mean:seeker_navcore_v2:PFL3 \
    --out phase5_eval_seeker_v2 >> "$LOG" 2>&1
fi
report

stage "4 hider_shuf0 control"
$PY -m flyseek.train.es --run hider_shuf0 --graph navcore_shuf0 --policy hider --init-from explore_shuf0 --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1
report
stage "4b held-out evaluation, hider shuffle control"
$PY -m flyseek.train.eval_role --role hider \
  explorer=brain:navcore:init:explore_navcore \
  trained_mean=brain:navcore:mean:hider_navcore \
  shuf_explorer=brain:navcore_shuf0:init:explore_shuf0 \
  shuf_trained_mean=brain:navcore_shuf0:mean:hider_shuf0 \
  --out phase5_eval_hider_shuffle >> "$LOG" 2>&1
report

stage "5 seeker shuffle control: $SHUF_SEEKER_RUN ($SEEKER_ARGS)"
$PY -m flyseek.train.es --run "$SHUF_SEEKER_RUN" --graph navcore_shuf0 --policy seeker --init-from explore_shuf0 $SEEKER_ARGS >> "$LOG" 2>&1
report
stage "5b held-out evaluation, seeker shuffle control"
$PY -m flyseek.train.eval_role --role seeker \
  explorer=brain:navcore:init:explore_navcore \
  trained_mean=brain:navcore:mean:$SEEKER_RUN \
  shuf_explorer=brain:navcore_shuf0:init:explore_shuf0 \
  shuf_trained_mean=brain:navcore_shuf0:mean:$SHUF_SEEKER_RUN \
  --out phase5_eval_seeker_shuffle >> "$LOG" 2>&1
report
stage "=== PHASE 5 OVERNIGHT PIPELINE DONE ==="
