# Workflow: DexMachina (Genesis)

The DexMachina workflow integrates SPIDER with [Genesis](https://genesis.github.io/) simulator and [DexMachina](https://github.com/MandiZhao/dexmachina) framework for dexterous manipulation with downstream RL training.

## Prerequisites

### Install DexMachina Environment

Follow the [official DexMachina installation guide](https://mandizhao.github.io/dexmachina-docs/0_install.html):

### Install SPIDER (Minimal)

Install SPIDER without MuJoCo Warp (only need optimization components):

```bash
cd /path/to/spider
conda activate dexmachina

# Install SPIDER without dependencies (DexMachina has its own)
pip install --ignore-requires-python --no-deps -e .
# Install other dependencies
pip install loguru tyro hydra-core rerun-sdk==0.26.2
```

## Running DexMachina Workflow

### Basic Example

```bash
conda activate dexmachina
cd /path/to/spider

# Run with default config (inspire hand, box-30-230 task)
python examples/run_dexmachina.py
```

This will:
1. Initialize Genesis environment with DexMachina task
2. Load reference motion from task
3. Optimize trajectory with Sampling
4. Save optimized trajectory and video

### Evaluate Trajectories

After running `run_dexmachina.py`, evaluate object tracking metrics (position, rotation, articulation distances) with:

```bash
conda activate dexmachina
cd /path/to/spider

# Evaluate all tasks under the default data directory
python spider/postprocess/evaluate_dexmachina.py

# Evaluate specific tasks
python spider/postprocess/evaluate_dexmachina.py --tasks box-30-230 laptop-30-230

# Use a different dataset/robot configuration
python spider/postprocess/evaluate_dexmachina.py --dataset_name arctic --robot_type inspire_hand --embodiment_type bimanual
```

The script uses the same dataset path resolution as `config.py` (`dataset_dir/processed/{dataset_name}/{robot_type}/{embodiment_type}/{task}/{data_id}/`). You can also pass `--data_dir` to override with an explicit path.

### Compare SPIDER vs RL (Multi-Seed)

To compare SPIDER (MPC) against RL rollouts across multiple seeds, use `--compare` mode. This loads both `trajectory_dexmachina.npz` (SPIDER) and `rollout_rl.npz` (RL) from each `{task}/{seed}/` directory and prints mean +/- std tables.

```bash
# Compare across seeds 0-4 (default), print Markdown tables
python spider/postprocess/evaluate_dexmachina.py --compare

# Compare with LaTeX tables and per-seed detail
python spider/postprocess/evaluate_dexmachina.py --compare --latex --detail

# Compare specific tasks / seeds
python spider/postprocess/evaluate_dexmachina.py --compare --tasks box-30-230 laptop-30-230 --seeds 0 1 2
```

### Setting Up on a New Machine

Pre-generated inspire_hand data (7 clips) is stored in this repo so you don't need to re-run the retargeting pipeline.

```
example_datasets/raw/dexmachina/
  contact_retarget/inspire_hand/s01/   # contact mapping (.npy) + videos (.mp4)
  retargeted/inspire_hand/s01/         # retargeted kinematics (.pt)
  retargeter_results/inspire_hand/s01/ # retargeter results (.npy)
```

Clips: `box_use_01`, `ketchup_use_01`, `ketchup_use_02`, `laptop_use_01`, `mixer_use_01`, `notebook_use_01`, `waffleiron_use_01`

After installing DexMachina, copy the data into its assets directory:

```bash
DEXMACHINA_ASSETS=$(python -c "from dexmachina.asset_utils import get_asset_path; print(get_asset_path(''))")
SPIDER_DATA=/path/to/spider/example_datasets/raw/dexmachina

cp -r $SPIDER_DATA/retargeted/inspire_hand/        $DEXMACHINA_ASSETS/retargeted/inspire_hand/
cp -r $SPIDER_DATA/contact_retarget/inspire_hand/   $DEXMACHINA_ASSETS/contact_retarget/inspire_hand/
cp -r $SPIDER_DATA/retargeter_results/inspire_hand/  $DEXMACHINA_ASSETS/retargeter_results/inspire_hand/
```

If the data is missing at runtime, the error message will print the exact copy commands for your paths.

### Common Issues

1. **Missing DexMachina data**: Copy pre-generated data from `example_datasets/raw/dexmachina/` into the DexMachina assets directory (see section above).
2. If failed load trajectory from dexmachina due to torch weights loading error, consider `data = torch.load(data_fname, weights_only=False)` to load the entire data.
3. numpy version issue: try to install `numpy==1.26.4` to avoid compatibility issues.
