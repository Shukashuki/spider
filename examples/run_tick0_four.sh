#!/usr/bin/env bash
# Tick-0 zero-init comparison: dial / mppi / cma / cma_dial
# Uses gigahand_act (contact guidance params), init_ctrl_mode=zero

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider
cd /home/roy/.openclaw/workspace/spider

BASE="outputs/tick0_four"
# ctrl_dt=0.2, sim_dt=0.005 → ctrl_steps=40
COMMON="+override=gigahand_act data_id=0 save_video=false viewer= \
    init_ctrl_mode=zero max_sim_steps=40 \
    improvement_threshold=0 max_num_iterations=32"

run() {
    local tag="$1"; shift
    echo "--- $tag"
    python examples/run_mjwp.py $COMMON output_dir="$BASE/$tag" "$@" 2>&1 \
        | grep -E "Final|Error|Traceback" || true
    echo "    → $BASE/$tag"
}

for TASK in p36-tea p44-dog p52-instrument; do
    run "${TASK}_dial"     task=$TASK optimizer_mode=dial
    run "${TASK}_mppi"     task=$TASK optimizer_mode=mppi
    run "${TASK}_cma_rank" task=$TASK optimizer_mode=cma_rank cma_sigma0=0.1
    run "${TASK}_cma_dial" task=$TASK optimizer_mode=cma_dial cma_sigma0=0.1
done

echo "All done → $BASE/"
