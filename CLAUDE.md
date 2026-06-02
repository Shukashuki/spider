# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPIDER (Scalable Physics-Informed DExterous Retargeting) is a physics-based retargeting framework that converts human motion (from video or mocap) into robot actions for dexterous hands and humanoid robots. Developed by Meta/FAIR. Python 3.12+, licensed CC BY-NC 4.0.

## Common Commands

### Environment Setup
```bash
uv sync                    # Install dependencies (preferred)
pip install --no-deps -e . # Editable install without pulling deps
```

### Linting & Formatting
```bash
ruff check .          # Lint
ruff check --fix .    # Auto-fix
ruff format .         # Format
```

### Running Tests
```bash
uv run spider/simulators/mjwp_test.py        # MuJoCo Warp simulator
uv run spider/simulators/dexmachina_test.py  # DexMachina simulator
uv run spider/simulators/hdmi_test.py        # HDMI simulator
```

### Running Retargeting (MJWP workflow)
```bash
# Single task
uv run examples/run_mjwp.py +override=gigahand task=p36-tea data_id=0 robot_type=xhand embodiment_type=bimanual

# With remote viewer
uv run examples/run_mjwp.py viewer="rerun"
```

### Full Pipeline (preprocess → retarget → postprocess)
```bash
uv run spider/process_datasets/gigahand.py --task=TASK --embodiment-type=TYPE --data-id=ID
uv run spider/preprocess/decompose_fast.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE
uv run spider/preprocess/detect_contact.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE
uv run spider/preprocess/generate_xml.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE --robot-type=ROBOT
uv run spider/preprocess/ik.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE --robot-type=ROBOT --open-hand
uv run examples/run_mjwp.py +override=NAME task=TASK data_id=ID robot_type=ROBOT embodiment_type=TYPE
```

## Code Style

- **Ruff** for linting and formatting: 88-char line length, Google-style docstrings, Python 3.12+ target
- See `[tool.ruff]` in `pyproject.toml` for full lint rule configuration

## Architecture

### Configuration System
`spider/config.py` defines a single `@dataclass Config` (~150 fields) that drives the entire pipeline. Configs are loaded via Hydra/OmegaConf from YAML files in `examples/config/`. Dataset-specific overrides live in `examples/config/override/`. The `+override=DATASET` Hydra syntax selects a dataset config overlay.

### Data Flow
1. **Raw data** (`.pkl`) → `spider/process_datasets/*.py` converts dataset-specific formats to standard NPZ
2. **Preprocessing** (`spider/preprocess/`) → mesh decomposition, contact detection, scene XML generation, inverse kinematics
3. **Physics optimization** (`examples/run_mjwp.py`) → sampling-based MPC retargeting using simulator
4. **Postprocessing** (`spider/postprocess/`) → success metrics, robot deployment prep

### Simulators (`spider/simulators/`)
- **MJWP** (`mjwp.py`, ~1400 lines) — primary backend, GPU-accelerated MuJoCo + Warp with batched environments
- **MJWP-EQ** (`mjwp_eq.py`) — variant with equality constraints
- **DexMachina** (`dexmachina.py`) — Genesis simulator for RL training
- **HDMI** (`hdmi.py`) — humanoid RL workflow
- **Isaac** (`isaac.py`) — NVIDIA Isaac integration

### Optimizer (`spider/optimizers/sampling.py`)
Sampling-based MPC using cross-entropy method. Key functions: `make_rollout_fn`, `make_optimize_fn`, `make_optimize_once_fn`. Supports `torch.compile` for speedup, noise scheduling with temperature annealing, and contact guidance.

### Viewers (`spider/viewers/`)
Multiple visualization backends (MuJoCo, Rerun, Viser) with a common interface. Selected via `viewer=` config parameter. Supports simultaneous multi-viewer (e.g., `viewer="mujoco-rerun"`).

### Dataset Processors (`spider/process_datasets/`)
Each supported dataset (GigaHands, OakInk, Hot3D, GMR, FAIR-MON, FAIR-FRE) has a dedicated processor that converts raw data to standard format. New datasets require a new processor here outputting NPZ files.

### Adding New Robots
Requires: MJCF assets in `spider/assets/`, embodiment mappings in `spider/config.py`, and reward weight tuning.

### Key Utility Modules
- `spider/io.py` — data loading/saving, path resolution
- `spider/math.py` — quaternion/rotation math utilities
- `spider/interp.py` — trajectory interpolation
- `spider/mujoco_utils.py` — MuJoCo model helpers

---

## Wuji Hand (Shukashuki fork addition)

### Quick-start commands
```bash
# Generate scene XML
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakink --task uncap_alcohol_burner --data-id 0 \
  --embodiment-type bimanual --robot-type wuji_hand

# IK
uv run spider/preprocess/ik.py \
  --dataset-name oakink --task uncap_alcohol_burner --data-id 0 \
  --embodiment-type bimanual --robot-type wuji_hand --open-hand

# Physics optimization — must use oakink_wuji, NOT oakink
uv run examples/run_mjwp.py \
  +override=oakink_wuji task=uncap_alcohol_burner data_id=0 viewer=""
```

### Design decisions

**6DOF forearm chain in right.xml / left.xml**
SPIDER IK uses equality constraints to pull the `right_palm` site to the MANO wrist
position. A fixed-base palm cannot be moved → both hands collapse at the origin.
Fix: prepend 3 prismatic (slide) + 3 revolute joints before palm_link, matching
the XHand forearm chain pattern. kp=1000 for slides, kp=200 for revolutes.

**left.xml body naming**
`bimanual.xml` merges both hands via `<include>`. The original MJCF uses identical
body/joint names for both hands, causing conflicts in bimanual. Fix: prefix all
left-hand body, joint, geom, site, and actuator names with `l_`. Left-hand STL
files also use the `l_` prefix in the shared `assets/` directory.

**Noise scale (oakink_wuji.yaml)**
`oakink.yaml` sets `first_ctrl_noise_scale=2.0`, which exceeds finger joint limits
(±1.57 rad) and applies enormous forces via the kp=1000 wrist slides (~50% NaN).
Always use `+override=oakink_wuji` (noise reduced to 0.2 / 0.4).

**njmax**
52-DOF bimanual model + object contacts exceed the default njmax=350.
`oakink_wuji.yaml` sets `njmax_per_env=512, nconmax_per_env=200`.

### Known limitations
- Left hand FK orientation (mirrored kinematics) not yet visually verified
- Physics optimization NaN rate to be confirmed with oakink_wuji.yaml (expect <10%)
- No bimanual URDF (single-hand URDFs exist; SPIDER does not require them)
