#!/usr/bin/env bash
# Multi-task: full run (contact guidance) + tick0 zero-init comparison
# Tasks: p36-tea, p44-dog, p52-instrument | Optimizers: dial, mppi, cma

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider
cd /home/roy/.openclaw/workspace/spider

TASKS=(p36-tea p44-dog p52-instrument)
OPTS=(dial mppi cma_rank)

# ctrl_dt=0.2, sim_dt=0.005 → ctrl_steps=40
TICK0_SIM_STEPS=40

run() {
    local tag="$1"; shift
    local dir="$1"; shift
    echo "--- $tag"
    python examples/run_mjwp.py "$@" output_dir="$dir" 2>&1 \
        | grep -E "INFO.*Saved|Final object|Error|Traceback" || true
    echo "    → $dir"
}

# ── Part 1: Full task (contact guidance, all ticks) ────────────────────────
echo "===== PART 1: Full task comparison ====="
for TASK in "${TASKS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        EXTRA=""; [ "$OPT" = "cma_rank" ] && EXTRA="cma_sigma0=0.1"
        run "${TASK}_${OPT}" "outputs/multitask_compare/${TASK}_${OPT}" \
            +override=gigahand_act task=$TASK data_id=0 \
            optimizer_mode=$OPT save_video=true viewer= $EXTRA
    done
done

# ── Part 2: Tick 0 zero-init (contact guidance params, u0=zero) ───────────
echo ""
echo "===== PART 2: Tick 0 zero-init ====="
for TASK in "${TASKS[@]}"; do
    for OPT in "${OPTS[@]}"; do
        EXTRA=""; [ "$OPT" = "cma_rank" ] && EXTRA="cma_sigma0=0.1"
        run "${TASK}_${OPT}_tick0z" "outputs/multitask_tick0z/${TASK}_${OPT}" \
            +override=gigahand_act task=$TASK data_id=0 \
            optimizer_mode=$OPT save_video=false viewer= \
            init_ctrl_mode=zero max_sim_steps=$TICK0_SIM_STEPS \
            improvement_threshold=0 max_num_iterations=32 $EXTRA
    done
done

echo ""
echo "All done."
