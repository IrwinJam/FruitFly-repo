#!/usr/bin/env bash
# Phase 5 R1 training queue (resumable: re-running continues each run from its checkpoint).
# Real wiring first, then the shuffled-wiring controls with the same budget, each starting
# from its own graph's Phase 4 explorer.
set -u
PY=.venv/Scripts/python
G=${GENERATIONS:-25}
$PY -m flyseek.train.es --run seeker_navcore --graph navcore --policy seeker --init-from explore_navcore --pop 12 --seeds-per-candidate 4 --generations $G
$PY -m flyseek.train.es --run hider_navcore --graph navcore --policy hider --init-from explore_navcore --pop 12 --seeds-per-candidate 2 --generations $G
$PY -m flyseek.train.es --run seeker_shuf0 --graph navcore_shuf0 --policy seeker --init-from explore_shuf0 --pop 12 --seeds-per-candidate 4 --generations $G
$PY -m flyseek.train.es --run hider_shuf0 --graph navcore_shuf0 --policy hider --init-from explore_shuf0 --pop 12 --seeds-per-candidate 2 --generations $G
echo "=== PHASE 5 TRAINING DONE ==="
