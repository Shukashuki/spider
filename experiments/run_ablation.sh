#!/usr/bin/env bash
# ===========================================================================
# SPIDER Ablation Experiments — run_ablation.sh
#
# Runs 3 ablation experiments sequentially (CMA-MPPI excluded for now).
# Each experiment outputs to: outputs/ablation/<task>/<exp_name>/
#
# Usage:
#   cd /path/to/spider
#   bash experiments/run_ablation.sh [TASK] [DATA_ID] [DATASET_DIR]
#
# Defaults to gigahand xhand bimanual p36-tea, data_id=0
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Configuration (override via CLI args or env vars)
# ---------------------------------------------------------------------------
TASK="${1:-p36-tea}"
DATA_ID="${2:-0}"
DATASET_DIR="${3:-example_datasets}"
DATASET_NAME="${DATASET_NAME:-gigahand}"
ROBOT_TYPE="${ROBOT_TYPE:-xhand}"
EMBODIMENT="${EMBODIMENT:-bimanual}"

OUTPUT_BASE="outputs/ablation/${TASK}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Common overrides shared by all experiments
COMMON_OVERRIDES=(
    "dataset_dir=${DATASET_DIR}"
    "dataset_name=${DATASET_NAME}"
    "robot_type=${ROBOT_TYPE}"
    "embodiment_type=${EMBODIMENT}"
    "task=${TASK}"
    "data_id=${DATA_ID}"
    "show_viewer=false"
    "save_video=false"
    "save_info=true"
    "save_config=true"
    "viewer="
)

log() { echo -e "\n\033[1;36m>>> [$1] $2\033[0m"; }
err() { echo -e "\033[1;31m!!! $1\033[0m" >&2; }

run_experiment() {
    local name="$1"
    shift
    local overrides=("$@")
    local out_dir="${OUTPUT_BASE}/${name}"

    log "$name" "output_dir=${out_dir}"
    mkdir -p "$out_dir"

    # Save the command for reproducibility
    echo "uv run examples/run_mjwp.py ${COMMON_OVERRIDES[*]} output_dir=${out_dir} ${overrides[*]}" \
        > "${out_dir}/command.txt"

    local t0
    t0=$(date +%s)

    if uv run examples/run_mjwp.py \
        "${COMMON_OVERRIDES[@]}" \
        "output_dir=${out_dir}" \
        "${overrides[@]}" \
        2>&1 | tee "${out_dir}/run.log"; then

        local t1
        t1=$(date +%s)
        log "$name" "DONE in $(( t1 - t0 ))s"

        # Quick validation
        if [ -x "$(command -v python3)" ]; then
            python3 experiments/validate_run.py --exp_dir "$out_dir" || true
        fi
    else
        local t1
        t1=$(date +%s)
        err "Experiment '$name' FAILED after $(( t1 - t0 ))s — check ${out_dir}/run.log"
    fi
}

# ===========================================================================
# Experiment 1: Baseline (no reference, no contact guidance)
# ===========================================================================
run_experiment "baseline" \
    "init_ctrl_mode=zero" \
    "contact_guidance=false" \
    "contact_rew_scale=0.0"

# ===========================================================================
# Experiment 2: Original SPIDER (reference + contact guidance)
# ===========================================================================
run_experiment "spider_original" \
    "init_ctrl_mode=reference" \
    "contact_guidance=true" \
    "contact_rew_scale=0.0"

# ===========================================================================
# Experiment 3: Contact-in-cost (no reference, contact distance in MPPI cost)
# ===========================================================================
run_experiment "contact_in_cost" \
    "init_ctrl_mode=zero" \
    "contact_guidance=false" \
    "contact_rew_scale=1.0"

# ===========================================================================
# Experiment 4: CMA-MPPI (placeholder — needs CMA optimizer)
# ===========================================================================
run_experiment "cma_mppi" \
    "init_ctrl_mode=zero" \
    "optimizer_type=cma" \
    "contact_guidance=false" \
    "contact_rew_scale=1.0"

echo ""
log "ALL" "Ablation experiments complete."
log "ALL" "Results in: ${OUTPUT_BASE}/"
echo ""
echo "Next steps:"
echo "  python experiments/compare_results.py \\"
echo "    --exp_dirs ${OUTPUT_BASE}/baseline ${OUTPUT_BASE}/spider_original ${OUTPUT_BASE}/contact_in_cost \\"
echo "    --labels baseline spider_original contact_in_cost"
