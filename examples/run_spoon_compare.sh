#!/usr/bin/env bash
# pick_spoon_bowl: 4-optimizer comparison + tick0 zero-init + timing
# oakink_act: ctrl_dt=0.05, sim_dt=0.005, ctrl_steps=10

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider
cd /home/roy/.openclaw/workspace/spider

BASE_FULL="outputs/spoon_full"
BASE_TICK0="outputs/spoon_tick0"
TIMING_LOG="outputs/spoon_timing.txt"

mkdir -p "$BASE_FULL" "$BASE_TICK0"
> "$TIMING_LOG"

TASK="pick_spoon_bowl"
# max_sim_steps=2000 = 200 ctrl ticks × 10 sim steps  (~10s of motion)
FULL_COMMON="+override=oakink_act task=$TASK data_id=0 save_video=false viewer= \
    improvement_threshold=0 max_sim_steps=2000"
# tick0: 1 ctrl tick = 10 sim steps, 32 iters from zero-init
TICK0_COMMON="+override=oakink_act task=$TASK data_id=0 save_video=false viewer= \
    init_ctrl_mode=zero max_sim_steps=10 improvement_threshold=0 max_num_iterations=32"

run_timed() {
    local tag="$1"; local dir="$2"; shift 2
    echo "--- $tag"
    local t0=$SECONDS
    python examples/run_mjwp.py "$@" output_dir="$dir" 2>&1 \
        | grep -E "Final|Saved|Error|Traceback" || true
    local elapsed=$(( SECONDS - t0 ))
    echo "    time=${elapsed}s  → $dir"
    echo "$tag  ${elapsed}s" >> "$TIMING_LOG"
}

echo "===== PART 1: Full task (100 ticks, 4 modes) ====="
run_timed "spoon_full_dial"     "$BASE_FULL/dial"     $FULL_COMMON optimizer_mode=dial
run_timed "spoon_full_mppi"     "$BASE_FULL/mppi"     $FULL_COMMON optimizer_mode=mppi
run_timed "spoon_full_cma_rank" "$BASE_FULL/cma_rank" $FULL_COMMON optimizer_mode=cma_rank cma_sigma0=0.1
run_timed "spoon_full_cma_dial" "$BASE_FULL/cma_dial" $FULL_COMMON optimizer_mode=cma_dial cma_sigma0=0.1

echo ""
echo "===== PART 2: Tick 0 zero-init (32 iters, 4 modes) ====="
run_timed "spoon_tick0_dial"     "$BASE_TICK0/dial"     $TICK0_COMMON optimizer_mode=dial
run_timed "spoon_tick0_mppi"     "$BASE_TICK0/mppi"     $TICK0_COMMON optimizer_mode=mppi
run_timed "spoon_tick0_cma_rank" "$BASE_TICK0/cma_rank" $TICK0_COMMON optimizer_mode=cma_rank cma_sigma0=0.1
run_timed "spoon_tick0_cma_dial" "$BASE_TICK0/cma_dial" $TICK0_COMMON optimizer_mode=cma_dial cma_sigma0=0.1

echo ""
echo "===== Timing summary ====="
cat "$TIMING_LOG"
echo ""
echo "All done."
