#!/usr/bin/env bash
# Full-task comparison: dial / mppi / cma on gigahand p36-tea.
# Uses conda spider environment. Saves video for each run.

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider

cd /home/roy/.openclaw/workspace/spider

BASE="outputs/full_compare_cg"
COMMON="+override=gigahand_act save_video=true viewer="

run() {
    local tag="$1"; shift
    local dir="$BASE/$tag"
    echo "=== $tag ==="
    python examples/run_mjwp.py $COMMON output_dir="$dir" "$@" 2>&1 \
        | grep -E "INFO|plan time|Total|Saved|Error|Traceback" || true
    echo "  → $dir"
}

run dial optimizer_mode=dial
run mppi optimizer_mode=mppi
run cma_rank optimizer_mode=cma_rank cma_sigma0=0.1

echo ""
echo "All done. Results in $BASE/"
