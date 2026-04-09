#!/usr/bin/env bash
# ===========================================================================
# SPIDER Multi-Task Ablation — run_multi_task.sh
#
# Runs 3 methods × N tasks across multiple datasets.
# Methods:
#   1. spider_original — MPPI + reference + contact guidance
#   2. baseline        — MPPI + zero init, no guidance
#   3. cma_with_ref    — CMA-ES + reference + contact guidance
#
# Usage:
#   cd /path/to/spider
#   bash experiments/run_multi_task.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DATASET_DIR="${DATASET_DIR:-example_datasets}"
OUTPUT_BASE="outputs/multi_task"

# ---------------------------------------------------------------------------
# Task definitions: dataset_name | robot_type | embodiment | task
# ---------------------------------------------------------------------------
TASK_DEFS=(
    # gigahand / xhand / bimanual
    "gigahand|xhand|bimanual|p36-tea"
    "gigahand|xhand|bimanual|p44-dog"
    "gigahand|xhand|bimanual|p52-instrument"
    # oakink / xhand / bimanual
    "oakink|xhand|bimanual|uncap_alcohol_burner"
    "oakink|xhand|bimanual|stir_beaker"
    "oakink|xhand|bimanual|unplug"
    "oakink|xhand|bimanual|pour_tube"
    "oakink|xhand|bimanual|lift_board"
    "oakink|xhand|bimanual|wipe_board"
    "oakink|xhand|bimanual|pick_spoon_bowl"
)

# ---------------------------------------------------------------------------
# Methods: name -> extra overrides
# ---------------------------------------------------------------------------
declare -A METHOD_OVERRIDES
METHOD_OVERRIDES[spider_original]="init_ctrl_mode=reference contact_guidance=true contact_rew_scale=0.0 optimizer_type=mppi"
METHOD_OVERRIDES[baseline]="init_ctrl_mode=zero contact_guidance=false contact_rew_scale=0.0 optimizer_type=mppi"
METHOD_OVERRIDES[cma_with_ref]="init_ctrl_mode=reference contact_guidance=true contact_rew_scale=0.0 optimizer_type=cma"

METHODS=("spider_original" "baseline" "cma_with_ref")

log() { echo -e "\n\033[1;36m>>> $1\033[0m"; }
err() { echo -e "\033[1;31m!!! $1\033[0m" >&2; }

total_runs=$(( ${#TASK_DEFS[@]} * ${#METHODS[@]} ))
run_count=0
fail_count=0

for task_def in "${TASK_DEFS[@]}"; do
    IFS='|' read -r ds_name robot_type embodiment task <<< "$task_def"
    task_label="${ds_name}_${task}"

    for method in "${METHODS[@]}"; do
        run_count=$((run_count + 1))
        out_dir="${OUTPUT_BASE}/${task_label}/${method}"

        # Skip if already completed
        if ls "${out_dir}"/trajectory_mjwp*.npz &>/dev/null; then
            log "[${run_count}/${total_runs}] SKIP ${task_label}/${method} (npz exists)"
            continue
        fi

        log "[${run_count}/${total_runs}] ${task_label} / ${method}"
        mkdir -p "$out_dir"

        read -ra method_args <<< "${METHOD_OVERRIDES[$method]}"

        COMMON=(
            "dataset_dir=${DATASET_DIR}"
            "dataset_name=${ds_name}"
            "robot_type=${robot_type}"
            "embodiment_type=${embodiment}"
            "task=${task}"
            "data_id=0"
            "show_viewer=false"
            "save_video=false"
            "save_info=true"
            "save_config=true"
            "viewer="
            "output_dir=${out_dir}"
        )

        cmd="uv run examples/run_mjwp.py ${COMMON[*]} ${method_args[*]}"
        echo "$cmd" > "${out_dir}/command.txt"

        t0=$(date +%s)
        if $cmd 2>&1 | tee "${out_dir}/run.log"; then
            t1=$(date +%s)
            log "[${run_count}/${total_runs}] ${task_label}/${method} DONE in $(( t1 - t0 ))s"
        else
            t1=$(date +%s)
            fail_count=$((fail_count + 1))
            err "${task_label}/${method} FAILED after $(( t1 - t0 ))s — check ${out_dir}/run.log"
        fi
    done
done

echo ""
log "All ${total_runs} runs complete (${fail_count} failures). Results in: ${OUTPUT_BASE}/"
echo ""
echo "Next step:"
echo "  uv run python experiments/compare_multi_task.py --base_dir ${OUTPUT_BASE}"
