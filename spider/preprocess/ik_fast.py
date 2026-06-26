"""Run IK for the given hand type and mode.

This is a simplified version of IK based on mink: a single zero-qpos seed is
settled in two phases (wrist targets, then + fingertip targets) instead of
SPIDER's `ik.py` random-restart search over `max_num_initial_guess` seeds.

Optional features (off by default, so existing non-wuji callers are unaffected):
- `--open-hand`: smoothly open un-contacted fingers before the MANO contact
  frame, mirroring `ik.py`'s behavior.
- `--act-scene`: load `scene_act.xml` (object = 6 single-axis joints) instead
  of `scene.xml` (object = free joint). Since mink can't write a 7D pos+quat
  target directly into 6 single-axis joint slots, the object is tracked via
  an extra `mink.FrameTask` on its site instead of a direct qpos write — this
  works for any underlying joint parameterization because mink solves through
  whatever Jacobian the site has, without needing SPIDER's mocap+equality-
  constraint apparatus.
- contact / contact_pos recording: forward-kinematics readout of all
  `track_*` sites into their paired `ref_*` mocap bodies, independent of how
  qpos was solved. Needed for the cap/knuckle geometric rewards.

Input: mujoco scene xml file + hand keypoints + object trajectories.
Output: npz file which contains qpos for key frames in xml.

Strategy:
1. Frame 0: set wrist (+ object, if --act-scene) targets first, integrate
   many steps, then add finger tip targets and integrate more steps.
2. Subsequent frames: solve IK with all targets, integrating sim_dt steps
   for each ref_dt interval.
3. Object qpos is set directly when the object has a free joint (no
   --act-scene); tracked via a FrameTask when it doesn't.

Author: Chaoyi Pan
Date: 2026-03-07
"""

import os
from pathlib import Path

import loguru
import mink
import mujoco
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer

# MANO 21-joint skeleton bones (parent -> child) for reference visualization
_MANO_SKELETON_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
_MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def _build_ref_skeleton(server, mano_assets_root, qpos_ref, ref_idx, embodiment_type):
    """Render the reference MANO skeleton in viser using fingertip + wrist targets.

    Returns a callable update_fn(t) -> None that updates joint/bone positions
    for frame t. We don't actually need MANO FK here because qpos_ref already
    has wrist + 5 fingertips per frame; we draw a simple wrist+fingertip
    skeleton plus straight bones (wrist->each fingertip).
    """
    if "right" in embodiment_type or embodiment_type == "bimanual":
        wrist_idx = ref_idx["right_palm"]
        tip_idxs = [ref_idx[f"right_{n}"] for n in
                    ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]]
    elif "left" in embodiment_type:
        wrist_idx = ref_idx["left_palm"]
        tip_idxs = [ref_idx[f"left_{n}"] for n in
                    ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]]
    else:
        return None

    wrist_h = server.scene.add_icosphere(
        "ref_hand/wrist", radius=0.012, color=(80, 130, 220),
        position=tuple(qpos_ref[0, wrist_idx, :3]),
    )
    tip_handles = [
        server.scene.add_icosphere(
            f"ref_hand/tip_{i}", radius=0.008, color=(220, 80, 80),
            position=tuple(qpos_ref[0, tidx, :3]),
        )
        for i, tidx in enumerate(tip_idxs)
    ]
    bone_handles = []
    for tidx in tip_idxs:
        h = server.scene.add_spline_catmull_rom(
            f"ref_hand/bone_w_to_{tidx}",
            positions=np.stack([
                qpos_ref[0, wrist_idx, :3], qpos_ref[0, tidx, :3]
            ]),
            color=(255, 200, 50), line_width=2.5,
        )
        bone_handles.append((h, tidx))

    def update(t):
        wrist_pos = qpos_ref[t, wrist_idx, :3]
        wrist_h.position = tuple(wrist_pos)
        for k, tidx in enumerate(tip_idxs):
            tip_handles[k].position = tuple(qpos_ref[t, tidx, :3])
        for h, tidx in bone_handles:
            h.positions = np.stack([wrist_pos, qpos_ref[t, tidx, :3]])

    return update


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = False,
    save_video: bool = False,
    save_viser: bool = False,
    mano_assets_root: str | None = None,
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float = 0.02,
    wrist_pos_cost: float = 0.3,
    wrist_ori_cost: float = 3.0,
    finger_pos_cost: float = 10.0,
    posture_cost: float = 1e-2,
    wrist_init_steps: int = 200,
    finger_init_steps: int = 300,
    average_frame_size: int = 3,
    z_offset: float = 0.0,
    act_scene: bool = False,
    open_hand: bool = False,
    contact_detection_step_threshold: int = 3,
    aggregate_contact: bool = True,
    object_pos_cost: float = 5.0,
    object_ori_cost: float = 5.0,
    seed: int = 42,
    force: bool = False,
):
    np.random.seed(seed)
    # Resolve directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    suffix = "_act" if act_scene else ""
    out_path = f"{processed_dir_robot}/trajectory_kinematic{suffix}.npz"
    if not force and os.path.exists(out_path):
        loguru.logger.info(f"Skipping ik_fast.py (output exists: {out_path})")
        return
    if act_scene:
        model_path = f"{processed_dir_robot}/../scene_act.xml"
    else:
        model_path = f"{processed_dir_robot}/../scene.xml"

    # Load reference keypoints
    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]
    try:
        contact_left = loaded_data["contact_left"][start_idx:end_idx]
        contact_right = loaded_data["contact_right"][start_idx:end_idx]
    except KeyError:
        loguru.logger.warning("No contact data found, using all one")
        contact_left = np.ones((qpos_finger_right.shape[0], 5))
        contact_right = np.ones((qpos_finger_left.shape[0], 5))
    contact_ref = np.concatenate([contact_right, contact_left], axis=1)
    if aggregate_contact:
        contact_aggregated = np.any(contact_ref, axis=-1)
        for i in range(contact_ref.shape[1]):
            contact_ref[:, i] = contact_aggregated
    # first contact frame per finger (used by --open-hand): first index where
    # contact holds for `contact_detection_step_threshold` consecutive steps
    first_contact_frame_left = np.zeros(5) + qpos_finger_right.shape[0]
    first_contact_frame_right = np.zeros(5) + qpos_finger_left.shape[0]
    for j in range(5):
        for i in range(contact_detection_step_threshold, len(contact_left)):
            if contact_left[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_left[j] = i
                break
        for i in range(contact_detection_step_threshold, len(contact_right)):
            if contact_right[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_right[j] = i
                break

    # Build reference array: (H, num_sites, 7) where 7 = [x, y, z, qw, qx, qy, qz]
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    num_frames = qpos_finger_right.shape[0]

    # Reference index mapping:
    # 0: right_palm, 1-5: right fingers, 6: left_palm, 7-11: left fingers,
    # 12: right_object, 13: left_object
    ref_idx = {}
    ref_idx["right_palm"] = 0
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"right_{name}"] = i + 1
    ref_idx["left_palm"] = 6
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"left_{name}"] = i + 7
    ref_idx["right_object"] = 12
    ref_idx["left_object"] = 13

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)

    # -- Viser setup (optional) --
    viser_state = {"enabled": False}
    if show_viewer or save_viser:
        try:
            from spider.viewers.viser_viewer import (
                _STATE as _VISER_STATE,
                build_and_log_scene,
                init_viser,
                log_frame,
            )

            init_viser(app_name=f"spider-ik-{task}")
            spec, viser_model, viewer_body_entity_and_ids = build_and_log_scene(
                Path(model_path)
            )
            del spec, viser_model
            viser_state.update(
                {
                    "enabled": True,
                    "log_frame": log_frame,
                    "body_ids": viewer_body_entity_and_ids,
                    "state": _VISER_STATE,
                }
            )
            loguru.logger.info(
                "Viser scene initialized (will record IK rollout for inspection)."
            )
        except Exception as e:
            loguru.logger.warning(f"Failed to init viser: {e}")

    # Optional reference MANO skeleton (only if MANO is loaded correctly)
    ref_skeleton = None
    if viser_state["enabled"] and mano_assets_root is not None:
        ref_skeleton = _build_ref_skeleton(
            viser_state["state"].server,
            mano_assets_root,
            qpos_ref,
            ref_idx,
            embodiment_type,
        )

    # Object DOF count
    if act_scene:
        nq_obj = model.nq - model.nu
    elif embodiment_type == "bimanual":
        nq_obj = 14
    elif embodiment_type in ["right", "left"]:
        nq_obj = 7
    else:
        nq_obj = 0

    # Create mink configuration
    configuration = mink.Configuration(model)
    data = configuration.data

    # Build site name lists
    finger_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    wrist_sites = []
    finger_sites = []
    object_sites = []
    if embodiment_type in ["right", "bimanual"]:
        wrist_sites.append("right_palm")
        finger_sites.extend([f"right_{f}" for f in finger_names])
        object_sites.append("right_object")
    if embodiment_type in ["left", "bimanual"]:
        wrist_sites.append("left_palm")
        finger_sites.extend([f"left_{f}" for f in finger_names])
        object_sites.append("left_object")

    # Create IK tasks
    # Cost priority: finger_pos > object > wrist_pos > wrist_ori
    wrist_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost,
            lm_damping=1.0,
        )
        for s in wrist_sites
    ]

    finger_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=finger_pos_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for s in finger_sites
    ]

    # Object tracking: only needed as an explicit task in --act-scene, where
    # the object has 6 single-axis joints (pos_x/y/z slides + rot_x/y/z
    # hinges) instead of a free joint, so a direct 7D pos+quat qpos write
    # (`set_object_qpos` below) isn't possible. A FrameTask sidesteps the
    # parameterization mismatch entirely since mink solves through whatever
    # Jacobian the site has.
    object_tasks = (
        [
            mink.FrameTask(
                frame_name=s,
                frame_type="site",
                position_cost=object_pos_cost,
                orientation_cost=object_ori_cost,
                lm_damping=1.0,
            )
            for s in object_sites
        ]
        if act_scene
        else []
    )

    posture_task = mink.PostureTask(model, cost=posture_cost)
    posture_task.set_target(configuration.q.copy())

    tasks_wrist = [posture_task, *wrist_tasks, *object_tasks]
    tasks_all = [posture_task, *wrist_tasks, *finger_tasks, *object_tasks]

    solver = "daqp"
    n_substeps = max(1, int(round(ref_dt / sim_dt)))

    # -- Helpers --
    def set_object_qpos(t):
        """Direct qpos write — only valid when the object has a free joint."""
        if embodiment_type == "bimanual":
            data.qpos[-14:-7] = qpos_ref[t, ref_idx["right_object"]]
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]
        elif embodiment_type == "right":
            data.qpos[-7:] = qpos_ref[t, ref_idx["right_object"]]
        elif embodiment_type == "left":
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]

    def set_wrist_targets(t):
        for wrist_task, site_name in zip(wrist_tasks, wrist_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            wrist_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    def set_finger_targets(t):
        for finger_task, site_name in zip(finger_tasks, finger_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            finger_task.set_target(
                mink.SE3(wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, *pos]))
            )

    def set_object_targets(t):
        for object_task, site_name in zip(object_tasks, object_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            object_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    # -- track_*/ref_* mocap recording setup (contact_pos) --
    # Pure forward-kinematics readout, independent of how qpos was solved:
    # every `track_*` site has a paired `ref_*` mocap body (added by
    # generate_xml.py / right.xml's knuckle sites); each frame we copy the
    # track site's world position into its ref mocap body, then snapshot all
    # ref mocap positions as that frame's contact_pos.
    track_site_ids = []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None and name.startswith("track"):
            track_site_ids.append(sid)
    ref_mocap_ids = []
    for sid in track_site_ids:
        track_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        ref_name = track_name.replace("track", "ref")
        mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
        ref_mocap_ids.append(model.body_mocapid[mocap_body_id])

    contact_map = {
        "right_thumb": 0,
        "right_index": 1,
        "right_middle": 2,
        "right_ring": 3,
        "right_pinky": 4,
        "left_thumb": 5,
        "left_index": 6,
        "left_middle": 7,
        "left_ring": 8,
        "left_pinky": 9,
    }

    side_map = {"right": first_contact_frame_right, "left": first_contact_frame_left}
    finger_idx_map = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}

    # Separate MjData used only for recording (open-hand override, contact_pos,
    # video/viewer rendering). mink's `configuration`/`data` must always carry
    # the *true* solved qpos forward as the warm start for the next frame, so
    # the open-hand override must never touch it.
    mj_data_record = mujoco.MjData(model)

    def record_frame(q_solved, t):
        q_record = q_solved.copy()
        if open_hand:
            for side in ["right", "left"]:
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    joint_ids = [
                        jid
                        for jid in range(model.njnt)
                        if (
                            (jn := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
                            is not None
                            and side in jn
                            and finger in jn
                        )
                    ]
                    if not joint_ids:
                        continue
                    contact_frame = side_map[side][finger_idx_map[finger]]
                    ratio = np.clip(t / max(contact_frame, 1), 0.0, 1.0)
                    ratio = 1.0 - np.cos(ratio * np.pi * 0.5)
                    for joint_idx in joint_ids:
                        q_record[joint_idx] = (
                            ratio * q_solved[joint_idx] + (1 - ratio) * 0.0
                        )
        mj_data_record.qpos[:] = q_record
        mujoco.mj_forward(model, mj_data_record)
        for track_site_id, mocap_id in zip(track_site_ids, ref_mocap_ids, strict=True):
            mj_data_record.mocap_pos[mocap_id] = mj_data_record.site_xpos[
                track_site_id
            ].copy()
        contact_pos_t = mj_data_record.mocap_pos.copy()
        contact_t = np.zeros(len(track_site_ids))
        for i, track_site_id in enumerate(track_site_ids):
            track_site_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
            )
            for k, v in contact_map.items():
                if k in track_site_name and "object" in track_site_name:
                    contact_t[i] = contact_ref[t, v]
                    break
        return q_record, contact_pos_t, contact_t

    # -- Phase 1+2: settle frame 0 from a zero-qpos seed --
    # wrist (+ object, if --act-scene) targets first, then add fingertips.
    # Structured as a callable (returning (q, cost)) so a future multi-seed
    # restart could call it repeatedly and keep the best result.
    def run_phase12(q_start):
        configuration.data.qpos[:] = q_start
        if act_scene:
            set_object_targets(0)
        else:
            set_object_qpos(0)
        configuration.update()
        set_wrist_targets(0)
        posture_task.set_target(q_start)
        for _ in range(wrist_init_steps):
            vel = mink.solve_ik(configuration, tasks_wrist, sim_dt, solver, damping=1e-5)
            configuration.integrate_inplace(vel, sim_dt)
            if not act_scene:
                set_object_qpos(0)

        configuration.update()
        set_finger_targets(0)
        for _ in range(finger_init_steps):
            vel = mink.solve_ik(configuration, tasks_all, sim_dt, solver, damping=1e-5)
            configuration.integrate_inplace(vel, sim_dt)
            if not act_scene:
                set_object_qpos(0)

        mujoco.mj_forward(model, configuration.data)
        cost = sum(
            w**2
            * float(
                np.sum(
                    (configuration.data.site(s).xpos - qpos_ref[0, ref_idx[s], :3]) ** 2
                )
            )
            for w, names in [
                (wrist_pos_cost, wrist_sites),
                (finger_pos_cost, finger_sites),
                *([(object_pos_cost, object_sites)] if act_scene else []),
            ]
            for s in names
        )
        return configuration.q.copy(), cost

    loguru.logger.info("Phase 1+2: Initializing wrist and finger positions...")
    best_q, init_cost = run_phase12(np.zeros(model.nq))
    loguru.logger.info(f"Frame-0 init cost: {init_cost:.6f}")
    configuration.data.qpos[:] = best_q
    configuration.update()
    if act_scene:
        set_object_targets(0)
    else:
        set_object_qpos(0)

    # -- Main IK loop --
    loguru.logger.info(f"Running IK for {num_frames} frames...")
    qpos_list = []
    contact_pos_list = []
    contact_list = []
    images = []

    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    run_viewer = get_viewer(show_viewer, model, mj_data_record)
    rate_limiter = RateLimiter(1 / ref_dt)

    with run_viewer() as gui:
        for t in range(num_frames):
            set_wrist_targets(t)
            set_finger_targets(t)
            if act_scene:
                set_object_targets(t)
            else:
                set_object_qpos(t)
            configuration.update()

            for _ in range(n_substeps):
                vel = mink.solve_ik(
                    configuration, tasks_all, sim_dt, solver, damping=1e-5
                )
                configuration.integrate_inplace(vel, sim_dt)
                if not act_scene:
                    set_object_qpos(t)

            q_record, contact_pos_t, contact_t = record_frame(
                configuration.q.copy(), t
            )
            qpos_list.append(q_record)
            contact_pos_list.append(contact_pos_t)
            contact_list.append(contact_t)

            if save_video:
                renderer.update_scene(data=mj_data_record, camera="front")
                images.append(renderer.render())

            if viser_state["enabled"]:
                viser_state["log_frame"](
                    mj_data_record, sim_time=t * ref_dt,
                    viewer_body_entity_and_ids=viser_state["body_ids"],
                    show_ui=True, playback_fps=1.0 / ref_dt,
                )
                if ref_skeleton is not None:
                    ref_skeleton(t)

            if show_viewer:
                gui.sync()
                rate_limiter.sleep()

            if t % 100 == 0:
                loguru.logger.info(f"  Frame {t}/{num_frames}")

    qpos_list = np.array(qpos_list)
    contact_pos_list = np.array(contact_pos_list)
    contact_list = np.array(contact_list)

    # -- Post-processing: moving average filter --
    def moving_average_filter(signal_data, window_size=5):
        return np.convolve(
            signal_data, np.ones(window_size) / window_size, mode="valid"
        )

    filtered = np.zeros(
        (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
    )
    for i in range(qpos_list.shape[1]):
        filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
    qpos_list = filtered

    # -- Compute qvel via finite differences --
    n_filtered = qpos_list.shape[0]
    qvel_list = np.zeros((n_filtered - 1, model.nv))
    for i in range(1, n_filtered):
        mujoco.mj_differentiatePos(
            model, qvel_list[i - 1], ref_dt, qpos_list[i - 1], qpos_list[i]
        )
    qpos_list = qpos_list[1:]
    assert qpos_list.shape[0] == qvel_list.shape[0]
    # contact/contact_pos were never smoothed (recorded once per raw frame);
    # align them to qpos_list's final length by taking the matching tail.
    contact_pos_list = contact_pos_list[-qpos_list.shape[0] :]
    contact_list = contact_list[-qpos_list.shape[0] :]
    assert qpos_list.shape[0] == contact_pos_list.shape[0] == contact_list.shape[0]

    # Zero object free-joint velocities: IK finite-diff produces large
    # angular-velocity artifacts (~3 rad/s) that destabilise the simulator.
    for _obj_jname in ["right_object_joint", "left_object_joint"]:
        _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _obj_jname)
        if _jid >= 0:
            _dofadr = model.jnt_dofadr[_jid]
            qvel_list[:, _dofadr : _dofadr + 6] = 0.0

    # -- Forward rollout for validation --
    model.opt.impratio = 1.0  # soften contacts for rollout stability
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mj_data_rollout = mujoco.MjData(model)
    n_rollout_substeps = max(1, int(round(ref_dt / sim_dt)))
    model.opt.timestep = ref_dt / n_rollout_substeps
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mj_data_rollout.ctrl[:] = qpos_list[0][: model.nq - nq_obj]
    for _ in range(n_rollout_substeps):
        mujoco.mj_step(model, mj_data_rollout)
    # Reset after warm-up to undo floor-object penetration impulse, so the
    # recorded qpos_rollout[0] below matches the state the loop continues from.
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mujoco.mj_forward(model, mj_data_rollout)
    n_final = qpos_list.shape[0]
    qpos_rollout = np.zeros((n_final, model.nq))
    qpos_rollout[0] = qpos_list[0]
    for i in range(1, n_final):
        mj_data_rollout.ctrl[:] = qpos_list[i][: model.nq - nq_obj]
        noise = np.random.randn(model.nu) * 0.2
        noise[:6] *= 0.0
        noise[22:28] *= 0.0
        mj_data_rollout.ctrl[:] += noise
        for _ in range(n_rollout_substeps):
            mujoco.mj_step(model, mj_data_rollout)
        qpos_rollout[i] = mj_data_rollout.qpos.copy()

    # -- Save outputs --
    file_dir = processed_dir_robot
    if save_video:
        import imageio

        video_path = f"{file_dir}/visualization_ik{'_act' if act_scene else ''}.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / ref_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    out_npz = f"{file_dir}/trajectory_kinematic{suffix}.npz"
    np.savez(
        out_npz,
        qpos=qpos_list,
        qvel=qvel_list,
        contact=contact_list,
        contact_pos=contact_pos_list,
        frequency=1 / ref_dt,
    )
    loguru.logger.info(f"Saved {out_npz}")

    out_npz = f"{file_dir}/trajectory_ikrollout{suffix}.npz"
    np.savez(out_npz, qpos=qpos_rollout)
    loguru.logger.info(f"Saved {out_npz}")

    # -- Save viser export for offline inspection --
    if save_viser and viser_state["enabled"]:
        viser_path = f"{file_dir}/visualization_ik.viser"
        try:
            server = viser_state["state"].server
            data_bytes = server.get_scene_serializer().serialize()
            Path(viser_path).write_bytes(data_bytes)
            loguru.logger.info(f"Saved viser scene to {viser_path}")
        except Exception as e:
            loguru.logger.warning(f"Failed to save viser scene: {e}")

    if show_viewer and viser_state["enabled"]:
        loguru.logger.info(
            "Viser viewer running. Use timeline slider in browser. Ctrl-C to exit."
        )
        import time

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    tyro.cli(main)
