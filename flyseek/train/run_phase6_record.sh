#!/usr/bin/env bash
# Phase 6.3: record the pre-registered showcase matches (seeds 6000-6002, GAMEPLAN Phase 6)
# after the validation pipeline has finished, export them to the viewer, and run the
# same seeds through the batched evaluator as a consistency check. Log: C:\flyseek-data\phase6.log
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
LOG=/c/flyseek-data/phase6.log
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
PRESET=${PRESET:-full}
S=init:explore_navcore_v2
HT=mean:hider_v2

until grep -q "PHASE 6 VALIDATION DONE" "$LOG"; do sleep 60; done

for H in 3 5; do
  for SEED in 6000 6001 6002; do
    NAME=showcase${H}_s${SEED}
    if [ -f "/c/flyseek-data/results/replays/$NAME.npz" ]; then continue; fi
    stage "6.3 recording $NAME ($PRESET preset, all brains, navcore)"
    $PY -m flyseek.world.match --hiders $H --brains all --graph navcore --preset $PRESET --seed $SEED \
      --seeker-adapter $S --hider-adapter $HT --name $NAME >> "$LOG" 2>&1
    $PY -m flyseek.world.export_replay $NAME >> "$LOG" 2>&1
  done
  stage "6.3 consistency: same seeds through the batched evaluator ($H hiders)"
  [ -f docs/phase6_showcase${H}_batched.json ] || $PY -m flyseek.train.eval_role --role both --preset $PRESET --n 3 --chunk 3 \
    --hiders $H --seed-base 6000 allbrain=brain:navcore:$S:$HT --out phase6_showcase${H}_batched >> "$LOG" 2>&1
done
stage "=== PHASE 6 RECORDING DONE ==="
