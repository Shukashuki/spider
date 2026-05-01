#!/usr/bin/env bash
# Multi-task comparison: dial / mppi / cma + contact guidance
# Tasks: p36-tea, p44-dog, p52-instrument (data_id=0 each)

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider

cd /home/roy/.openclaw/workspace/spider

BASE="outputs/multitask_compare"
COMMON="+override=gigahand_act save_video=true viewer="

run() {
    local tag="$1"; shift
    local dir="$BASE/$tag"
    echo "=== $tag ==="
    python examples/run_mjwp.py $COMMON output_dir="$dir" "$@" 2>&1 \
        | grep -E "INFO.*Saved|Final object|Error|Traceback" || true
    echo "  → $dir"
}

for TASK in p36-tea p44-dog p52-instrument; do
    for OPT in dial mppi cma; do
        EXTRA=""
        if [ "$OPT" = "cma" ]; then EXTRA="cma_sigma0=0.1"; fi
        run "${TASK}_${OPT}" task=$TASK data_id=0 optimizer_mode=$OPT $EXTRA
    done
done

echo ""
echo "All done. Results in $BASE/"
