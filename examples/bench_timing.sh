#!/usr/bin/env bash
# Timing benchmark: p36-tea, gigahand_act, 4 optimizer modes
# early stopping enabled (improvement_threshold=0.01, same as gigahand.yaml default)
# Runs full task, records wall time per tick.

set -e
source /home/roy/miniconda3/etc/profile.d/conda.sh
conda activate spider
cd /home/roy/.openclaw/workspace/spider

BASE="outputs/timing_bench"
TIMING_LOG="$BASE/timing.txt"
mkdir -p "$BASE"
> "$TIMING_LOG"

TASK="p36-tea"
COMMON="+override=gigahand_act task=$TASK data_id=0 save_video=false viewer= \
    improvement_threshold=0.01 max_num_iterations=32"

run_timed() {
    local tag="$1"; local dir="$BASE/$tag"; shift
    echo "--- $tag"
    local t0=$SECONDS
    python examples/run_mjwp.py $COMMON output_dir="$dir" "$@" 2>&1 \
        | grep -E "opt_steps|Final|Saved" | tail -5 || true
    local elapsed=$(( SECONDS - t0 ))
    # count ticks from npz
    local ticks=$(python3 -c "
import numpy as np, glob
f = glob.glob('$dir/trajectory_mjwp*.npz')
if f: print(np.load(f[0])['opt_steps'].shape[0])
else: print(0)
" 2>/dev/null)
    local per_tick=0
    [ "$ticks" -gt 0 ] && per_tick=$(echo "scale=2; $elapsed / $ticks" | bc)
    echo "    total=${elapsed}s  ticks=$ticks  per_tick=${per_tick}s"
    echo "$tag  total=${elapsed}s  ticks=$ticks  per_tick=${per_tick}s" >> "$TIMING_LOG"
}

echo "===== Timing benchmark: p36-tea, gigahand_act, early_stop=0.01 ====="
run_timed "dial"     optimizer_mode=dial
run_timed "mppi"     optimizer_mode=mppi
run_timed "cma_rank" optimizer_mode=cma_rank cma_sigma0=0.1
run_timed "cma_dial" optimizer_mode=cma_dial cma_sigma0=0.1

echo ""
echo "===== Summary ====="
cat "$TIMING_LOG"
echo ""

# also compute mean opt_steps (actual iters used with early stopping)
python3 -c "
import numpy as np, glob, os
base = 'outputs/timing_bench'
print(f'  {\"mode\":<12} {\"mean_iters\":>12} {\"min_iters\":>10} {\"max_iters\":>10}')
print('  ' + '-'*47)
for tag in ['dial','mppi','cma_rank','cma_dial']:
    f = glob.glob(f'{base}/{tag}/trajectory_mjwp*.npz')
    if not f: continue
    d = np.load(f[0])
    steps = d['opt_steps'].flatten()
    print(f'  {tag:<12} {steps.mean():>12.1f} {steps.min():>10d} {steps.max():>10d}')
" 2>/dev/null

echo "Done."
