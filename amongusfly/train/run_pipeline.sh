#!/usr/bin/env bash
# Every main result in docs/results/: compass tests, training, held-out evaluations and controls, tests on
# the full network, the
# showcase games and their quality checks, circuit activity, seeker circuits, a replication run and the
# 300-game verification. Resumable: finished results and replays are skipped, training resumes from
# its checkpoints. Log: $AMONGUSFLY_DATA/pipeline.log (default C:\amongusfly-data).
set -u
cd "$(dirname "$0")/../.."
PY=${AMONGUSFLY_PY:-.venv/Scripts/python}
LOG=${AMONGUSFLY_DATA:-/c/amongusfly-data}/pipeline.log
REPLAYS=${AMONGUSFLY_DATA:-/c/amongusfly-data}/results/replays
mkdir -p docs/results/mass "$(dirname "$LOG")"
stage() { echo "[stage] $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
need() { [ ! -f "docs/results/$1.json" ]; }
R=results

stage "1 compass tests"
need compass_open_loop && $PY -m amongusfly.brain.cx_test --r-max 150 --out $R/compass_open_loop >> "$LOG" 2>&1
need invisible_goal && $PY -m amongusfly.experiments.arena_goal --out $R/invisible_goal >> "$LOG" 2>&1
need invisible_goal_100_shuffles && $PY -m amongusfly.experiments.arena_goal --shuffles 100 --out $R/invisible_goal_100_shuffles >> "$LOG" 2>&1
need goal_to_turn && $PY -m amongusfly.experiments.arena_goal --probe --out $R/goal_to_turn >> "$LOG" 2>&1
need compass_by_heading && $PY -m amongusfly.experiments.cx_agent_check --headings 16 --out $R/compass_by_heading >> "$LOG" 2>&1

stage "2 walker training (real and shuffled wiring)"
$PY -m amongusfly.train.es --run explore_v5 --graph navcore --policy route --init-from explore_v4 --seconds 90 --generations 20 >> "$LOG" 2>&1
$PY -m amongusfly.train.es --run explore_shuf_v5 --graph navcore_shuf0 --policy route --init-from explore_shuf_v4 --seconds 90 --generations 20 >> "$LOG" 2>&1

stage "3 navigation and walking quality"
need navigation && $PY -m amongusfly.train.eval_explore --n 40 untrained=navcore:init trained=navcore:best:explore_v5 \
  shuf_trained=navcore_shuf0:best:explore_shuf_v5 pfl3_off=navcore:best:explore_v5:PFL3 --out $R/navigation >> "$LOG" 2>&1
need walking_quality && $PY -m amongusfly.experiments.locomotion_quality --run explore_v5 --n 24 --seconds 120 \
  --conditions "walker=2.0:repel=1.5" --out $R/walking_quality >> "$LOG" 2>&1
need steering_curve && $PY -m amongusfly.experiments.steer_curve --run explore_v5 --out $R/steering_curve >> "$LOG" 2>&1
need ideal_steering && $PY -m amongusfly.experiments.ideal_steer --run explore_v5 --out $R/ideal_steering >> "$LOG" 2>&1
# does the steering dead zone move with the connection scale or the compass input strength?
for W in 0.36 0.41 0.46 0.514 0.57 0.62 0.67; do
  need steering_sensitivity/scale_$W && $PY -m amongusfly.experiments.steer_curve --run explore_v5 --weight-scale $W \
    --out $R/steering_sensitivity/scale_$W >> "$LOG" 2>&1
done
for G in 0.5 0.75 1.5 2.0; do
  need steering_sensitivity/gain_$G && $PY -m amongusfly.experiments.steer_curve --run explore_v5 --input-gain $G \
    --out $R/steering_sensitivity/gain_$G >> "$LOG" 2>&1
done
# the same model on the full network (all 165,122 neurons), run in small batches to fit in GPU memory
need compass_open_loop_full && $PY -m amongusfly.brain.cx_test --graph full --shuffles 0 --r-max 150 \
  --out $R/compass_open_loop_full >> "$LOG" 2>&1
need invisible_goal_full && $PY -m amongusfly.experiments.arena_goal --graph full --shuffles 0 --out $R/invisible_goal_full >> "$LOG" 2>&1
need navigation_full && $PY -m amongusfly.train.eval_explore --n 40 --chunk 20 \
  subnetwork=navcore:best:explore_v5 full=full:best:explore_v5 --out $R/navigation_full >> "$LOG" 2>&1

stage "4 role training (real and shuffled wiring)"
$PY -m amongusfly.train.es --run seeker_v6 --graph navcore --policy seeker --init-from explore_v5 \
  --pop 8 --seeds-per-candidate 8 --generations 20 --sigma 0.15 >> "$LOG" 2>&1
$PY -m amongusfly.train.es --run hider_v5 --graph navcore --policy hider --init-from explore_v5 \
  --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1
$PY -m amongusfly.train.es --run seeker_shuf_v6 --graph navcore_shuf0 --policy seeker --init-from explore_shuf_v5 \
  --pop 8 --seeds-per-candidate 8 --generations 20 --sigma 0.15 >> "$LOG" 2>&1
$PY -m amongusfly.train.es --run hider_shuf_v5 --graph navcore_shuf0 --policy hider --init-from explore_shuf_v5 \
  --pop 12 --seeds-per-candidate 2 --generations 25 >> "$LOG" 2>&1

stage "5 each role against scripted opponents, with silenced circuits and shuffled wiring"
need seeker_vs_scripted && $PY -m amongusfly.train.eval_role --role seeker --preset short --n 40 \
  explorer=brain:navcore:init:explore_v5 trained_best=brain:navcore:best:seeker_v6 trained_mean=brain:navcore:mean:seeker_v6 \
  random=random scripted=scripted lc10a_off=brain:navcore:init:explore_v5:LC10a pfl3_off=brain:navcore:init:explore_v5:PFL3 \
  shuf_trained=brain:navcore_shuf0:best:seeker_shuf_v6 --out $R/seeker_vs_scripted >> "$LOG" 2>&1
need hiders_vs_scripted && $PY -m amongusfly.train.eval_role --role hider --preset short --n 40 \
  explorer=brain:navcore:init:explore_v5 trained_mean=brain:navcore:mean:hider_v5 trained_best=brain:navcore:best:hider_v5 \
  random=random scripted=scripted loom_off=brain:navcore:mean:hider_v5:LC4,LPLC2 dnp01_off=brain:navcore:mean:hider_v5:DNp01 \
  pfl3_off=brain:navcore:mean:hider_v5:PFL3 shuf_trained=brain:navcore_shuf0:mean:hider_shuf_v5 --out $R/hiders_vs_scripted >> "$LOG" 2>&1
need seeker_circuits_300s && $PY -m amongusfly.train.eval_role --role seeker --preset full --n 20 \
  walker=brain:navcore:init:explore_v5 pfl3_off=brain:navcore:init:explore_v5:PFL3 \
  pfl2_off=brain:navcore:init:explore_v5:PFL2 pfl3_pfl2_off=brain:navcore:init:explore_v5:PFL3,PFL2 \
  lc10a_off=brain:navcore:init:explore_v5:LC10a shuffled=brain:navcore_shuf0:best:seeker_shuf_v6 \
  --out $R/seeker_circuits_300s >> "$LOG" 2>&1
# the readout's goal-behind turn reads PFL2, so repeat the key conditions with that rule off
need seeker_circuits_300s_rule_off && $PY -m amongusfly.train.eval_role --role seeker --preset full --n 20 \
  walker=brain:navcore:init:explore_v5 pfl2_off=brain:navcore:init:explore_v5:PFL2 pfl3_off=brain:navcore:init:explore_v5:PFL3 \
  --decoder-set goal_behind.enabled=false --out $R/seeker_circuits_300s_rule_off >> "$LOG" 2>&1

# the role-trained seeker is used only if it beats the walker on held-out matches (p < 0.05)
SEEK=$($PY - <<'PY'
import json
r = json.load(open("docs/results/seeker_vs_scripted.json"))["results"]
v = r["trained_mean"].get("vs_explorer", {})
print("mean:seeker_v6" if v.get("fitness_diff_mean", 0) > 0 and v.get("paired_t_p", 1) < 0.05 else "init:explore_v5")
PY
)
HT=mean:hider_v5

stage "6 every fly a brain (seeker adapter: $SEEK)"
allbrain() {  # $1 = result name, then eval_role arguments
  local name=$1; shift
  need "$name" && $PY -m amongusfly.train.eval_role --role both "$@" --out $R/$name >> "$LOG" 2>&1
}
allbrain allbrain_90s_3hiders --preset short --n 40 allbrain=brain:navcore:$SEEK:$HT \
  untrained_hiders=brain:navcore:$SEEK:init:explore_v5 pfl3_off=brain:navcore:$SEEK:$HT:PFL3 \
  shuffled=brain:navcore_shuf0:init:explore_shuf_v5:mean:hider_shuf_v5
allbrain allbrain_300s_3hiders --preset full --n 20 allbrain=brain:navcore:$SEEK:$HT pfl3_off=brain:navcore:$SEEK:$HT:PFL3
allbrain allbrain_300s_5hiders --preset full --n 20 --hiders 5 allbrain=brain:navcore:$SEEK:$HT
allbrain replication_90s_3hiders --preset short --n 40 allbrain=brain:navcore:$SEEK:$HT
allbrain replication_300s_3hiders --preset full --n 20 allbrain=brain:navcore:$SEEK:$HT
allbrain replication_300s_5hiders --preset full --n 20 --hiders 5 allbrain=brain:navcore:$SEEK:$HT

stage "7 showcase games (and a second recording of the same seeds) and their quality checks"
record() {  # $1 = replay set name
  for H in 3 5; do
    for SEED in 6000 6001 6002 6003 6004 6005; do
      [ "$H" = 5 ] && [ "$SEED" -gt 6002 ] && continue
      NAME=$1_${H}h_s${SEED}
      for TRY in 1 2; do  # retry once if a match process dies
        [ -f "$REPLAYS/$NAME.npz" ] && break
        $PY -m amongusfly.world.match --hiders $H --brains all --graph navcore --preset full --seed $SEED \
          --seeker-adapter $SEEK --hider-adapter $HT --name $NAME >> "$LOG" 2>&1
      done
      [ -d "viewer/public/replays/$NAME" ] || $PY -m amongusfly.world.export_replay $NAME >> "$LOG" 2>&1
    done
  done
}
record showcase_v4
record showcase_v5
need quality_checks && $PY -m amongusfly.experiments.clean_gate --pattern "showcase_v4_*" --out $R/quality_checks >> "$LOG" 2>&1
need quality_checks_rerun && $PY -m amongusfly.experiments.clean_gate --pattern "showcase_v5_*" --out $R/quality_checks_rerun >> "$LOG" 2>&1
need circuit_activity && $PY -m amongusfly.experiments.circuit_activity --pattern "showcase_v[45]_*" --out $R/circuit_activity >> "$LOG" 2>&1
$PY -m amongusfly.experiments.path_figure --pattern "showcase_v4_*" --out seeker_paths >> "$LOG" 2>&1

stage "8 verification: 300 more games on seeds never used elsewhere"
for H in 3 5; do
  CHUNKS=$([ "$H" = 3 ] && echo "0 1 2 3" || echo "0 1")
  BASE=$([ "$H" = 3 ] && echo 100000 || echo 200000)
  for C in $CHUNKS; do
    need "mass/${H}hiders_chunk$C" && $PY -m amongusfly.train.eval_role --role both --preset full --n 50 --hiders $H \
      --seed-base $((BASE + C * 50)) allbrain=brain:navcore:$SEEK:$HT --save-paths mass --out $R/mass/${H}hiders_chunk$C >> "$LOG" 2>&1
  done
done
$PY -m amongusfly.experiments.mass_summary >> "$LOG" 2>&1
$PY -m amongusfly.experiments.path_figure --mass mass --out seeker_paths_300 >> "$LOG" 2>&1

stage "9 a gentler compass input (0.75x), chosen from the steering sweep and confirmed on unused seeds"
need walking_input_gain && $PY -m amongusfly.experiments.locomotion_quality --run explore_v5 --n 24 --seconds 120 \
  --conditions "input_1.0=2.0:repel=1.5" "input_0.9=2.0:repel=1.5:input=0.9" "input_0.75=2.0:repel=1.5:input=0.75" \
  "input_0.6=2.0:repel=1.5:input=0.6" --out $R/walking_input_gain >> "$LOG" 2>&1
need navigation_input_gain && $PY -m amongusfly.train.eval_explore --n 40 --seed-base 60000 walker=navcore:best:explore_v5 \
  input_0.75=navcore:best:explore_v5:input=0.75 --out $R/navigation_input_gain >> "$LOG" 2>&1
need seeker_input_gain && $PY -m amongusfly.train.eval_role --role seeker --preset full --n 20 --seed-base 400000 \
  walker=brain:navcore:$SEEK input_0.75=brain:navcore:$SEEK:input=0.75 --save-paths input_gain \
  --out $R/seeker_input_gain >> "$LOG" 2>&1
need hiders_input_gain && $PY -m amongusfly.train.eval_role --role hider --preset short --n 40 \
  trained_mean=brain:navcore:$HT input_0.75=brain:navcore:$HT:input=0.75 --out $R/hiders_input_gain >> "$LOG" 2>&1
need allbrain_input_gain && $PY -m amongusfly.train.eval_role --role both --preset short --n 40 \
  allbrain=brain:navcore:$SEEK:$HT input_0.75=brain:navcore:$SEEK:$HT:input=0.75 --out $R/allbrain_input_gain >> "$LOG" 2>&1
mkdir -p docs/results/mass_gentle
for H in 3 5; do
  CHUNKS=$([ "$H" = 3 ] && echo "0 1 2 3" || echo "0 1")
  BASE=$([ "$H" = 3 ] && echo 100000 || echo 200000)
  for C in $CHUNKS; do
    need "mass_gentle/${H}hiders_chunk$C" && $PY -m amongusfly.train.eval_role --role both --preset full --n 50 --hiders $H \
      --seed-base $((BASE + C * 50)) allbrain=brain:navcore:$SEEK:$HT:input=0.75 --save-paths mass_gentle \
      --out $R/mass_gentle/${H}hiders_chunk$C >> "$LOG" 2>&1
  done
done
$PY -m amongusfly.experiments.path_figure --mass mass_gentle --out seeker_paths_300_gentle >> "$LOG" 2>&1

stage "=== PIPELINE DONE (seeker adapter $SEEK) ==="
