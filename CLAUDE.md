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

---

## Original SPIDER vs Custom Additions

`--act-scene`, `--free-rot-z`, and `contact_guidance` are **original SPIDER features**
present in `generate_xml.py`, `ik.py`, and `config.py`. The official workflow page
(jc-bao.github.io/spider) does not document them but they are fully supported.

| Feature | Origin | Notes |
|---|---|---|
| `generate_xml.py --act-scene` | **Original SPIDER** | generates scene_act.xml |
| `generate_xml.py --free-rot-z` | **Original SPIDER** | rot_z joint has no actuator |
| `ik.py --act-scene` | **Original SPIDER** | generates trajectory_kinematic_act.npz |
| `contact_guidance: true` | **Original SPIDER** | uses scene_act.xml in optimizer |
| knuckle tracking sites in right.xml | **Custom (this fork)** | `track_hand_right_{finger}_knuckle` |
| knuckle mocap bodies in generate_xml.py | **Custom (this fork)** | contact_pos shape: 10→15 |
| `cap_dist_rew` / `cap_dir_rew` | **Custom (this fork)** | fingertip-object geometric reward |
| `knuckle_dist_rew` / `knuckle_dir_rew` | **Custom (this fork)** | DIP joint geometric reward |

---

## Knuckle Tracking & Geometric Rewards (right embodiment)

### What changed in right.xml

Each `fingerX_link3` body (DIP joint level — the phalange just before the fingertip)
now has a `track_hand_right_{finger}_knuckle` site at `pos="0 0 0"` (DIP joint origin):

```
finger1_link3 → track_hand_right_thumb_knuckle
finger2_link3 → track_hand_right_index_knuckle
finger3_link3 → track_hand_right_middle_knuckle
finger4_link3 → track_hand_right_ring_knuckle
finger5_link3 → track_hand_right_pinky_knuckle
```

`generate_xml.py` adds a corresponding `ref_hand_right_{finger}_knuckle` mocap body
for each finger (alongside the existing `ref_hand_right_{finger}_tip` bodies).

### IK output: contact_pos shape changed

IK automatically tracks **all** `track_*` sites. With the new knuckle sites,
`trajectory_kinematic.npz["contact_pos"]` is now **(T, 15, 3)** instead of (T, 10, 3).

Mocap layout (right embodiment, per finger in order thumb→index→middle→ring→pinky):

```
index  body name                        what it stores
─────  ───────────────────────────────  ──────────────────────────────────────
 0     ref_object_right_thumb_tip       contact site pos on object (thumb)
 1     ref_hand_right_thumb_tip         robot thumb fingertip world pos
 2     ref_hand_right_thumb_knuckle     robot thumb DIP joint world pos  ← NEW
 3     ref_object_right_index_tip       contact site pos on object (index)
 4     ref_hand_right_index_tip         robot index fingertip world pos
 5     ref_hand_right_index_knuckle     robot index DIP joint world pos  ← NEW
 ...   (same pattern for middle/ring/pinky, indices 6-14)
```

**The IK optimization is NOT affected** — knuckle sites are passively recorded,
not used as IK targets. Convergence and `qpos` output are identical to before.

**Compatibility:** Old `contact_pos (T, 10, 3)` files are incompatible with
`cap_dist_rew_scale > 0` or `knuckle_dist_rew_scale > 0` (index out of range).
Standard rewards (`cap_dist_rew_scale = 0`, default) are unaffected.

### generate_xml.py: two separate calls for two scene files

```bash
# scene.xml — standard physics (free joint on object, nq=33)
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand

# scene_act.xml — contact guidance (6 single-DOF joints on object, nq=32)
# --act-scene: saves scene_act.xml instead of scene.xml (does NOT overwrite scene.xml)
# --free-rot-z: cap rot_z joint has no actuator (spins freely under hand contact)
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand \
  --act-scene --free-rot-z
```

IK must be run **twice** for the full pipeline — once per scene:

```bash
# standard IK → trajectory_kinematic.npz  (used by Mode A and Mode C)
uv run spider/preprocess/ik.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand --open-hand --no-show-viewer

# act-scene IK → trajectory_kinematic_act.npz  (used by Mode B)
uv run spider/preprocess/ik.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand --open-hand --no-show-viewer \
  --act-scene
```

**Act-scene IK local optimum difference:** `ik.py --act-scene` uses scene_act.xml
(object has 6 actuated single-DOF joints). The different constraint landscape causes
the optimizer to converge to a different local optimum — robot finger joints can differ
by up to **123°** vs the standard IK, while fingertip world positions differ by only
**~2 cm**. Both solutions satisfy the IK objective. This is expected behaviour.

### New reward types (config.py fields + mjwp.py)

All four rewards compare **current simulation** vs **IK reference** at each frame:

| Config field | Formula | Source |
|---|---|---|
| `cap_dist_rew_scale` | `Σᵢ \|dist(tip_i, obj)_sim - dist(tip_i, obj)_ref\|` | contact_pos_ref[1,4,7,10,13] + qpos_ref[-7:-4] |
| `cap_dir_rew_scale` | `Σᵢ ‖unit(tip_i-obj)_sim - unit(tip_i-obj)_ref‖` | same |
| `knuckle_dist_rew_scale` | same formula but with DIP joint positions | contact_pos_ref[2,5,8,11,14] + qpos_ref[-7:-4] |
| `knuckle_dir_rew_scale` | same formula but with DIP joint positions | same |

Reference YAML: `examples/config/override/oakinkv2_wuji_cap_contact.yaml`

`build_hand_contact_site_ids` in `config.py` now filters for `_tip` sites only
(added `"_tip" in name` guard), so knuckle sites are not accidentally selected
for the fingertip reward even though they appear earlier in the site list.

### Contact Guidance (scene_act.xml)

When `contact_guidance: true`, the optimizer uses `scene_act.xml` where:
- Object has 6 single-DOF joints (pos_x/y/z slides + rot_x/y/z hinges)
- Up to 6 position actuators guide the object toward the reference trajectory
- Actuator gains decay by `guidance_decay_ratio` each iteration → 0 at last iter
- With `--free-rot-z`: rot_z hinge has **no actuator**, cap spins freely under hand torque
- `object_rot_threshold: 10.0` (large) prevents early termination from z-rotation

`nq_obj = 6` for act scene (vs 7 for free joint in standard scene).
`ik.py` computes `nq_obj = mj_model_ik.nq - mj_model_ik.nu` dynamically.

Reference YAML: `examples/config/override/oakinkv2_wuji_act.yaml`
