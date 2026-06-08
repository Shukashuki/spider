#!/usr/bin/env bash
# Run tick-0 convergence experiments on gigahand p36-tea.
# Compares dial / mppi / cma across horizons and sigma0 values.
# max_sim_steps=40 → exactly 1 control tick (ctrl_dt=0.4, sim_dt=0.01).

set -e
UV="/home/roy/.local/bin/uv"
BASE="outputs/tick0_zero"
COMMON="+override=gigahand data_id=0 max_sim_steps=40 save_video=false viewer= max_num_iterations=32 improvement_threshold=0 init_ctrl_mode=zero"

run() {
    local tag="$1"; shift
    local dir="$BASE/$tag"
    echo "=== $tag ==="
    $UV run examples/run_mjwp.py $COMMON output_dir="$dir" "$@" 2>&1 \
        | grep -E "INFO|plan time|Total|Error|Traceback" || true
    echo "  → $dir"
}

# ── DIAL ──────────────────────────────────────────────────────────────────────
run dial_h08 optimizer_mode=dial  horizon=0.8
run dial_h16 optimizer_mode=dial  horizon=1.6

# ── Pure MPPI ─────────────────────────────────────────────────────────────────
run mppi_h08 optimizer_mode=mppi  horizon=0.8
run mppi_h16 optimizer_mode=mppi  horizon=1.6

# ── MPPI-CMA (full covariance) ────────────────────────────────────────────────
for H in 0.8 1.6; do
    htag=$(echo "$H" | tr '.' '_')
    for S in 0.05 0.1 0.2; do
        stag=$(echo "$S" | sed 's/0\./s/')
        run "cma_rank_h${htag}_${stag}" optimizer_mode=cma_rank horizon=$H cma_sigma0=$S
    done
done

echo ""
echo "All done. Results in $BASE/"
