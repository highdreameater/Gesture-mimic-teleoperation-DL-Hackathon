"""
inference.py  –  G1 Teleoperation Inference Node
==================================================
Receives raw MediaPipe landmarks from vision_node.py over ZMQ, runs:
  • IL brain  (BehavioralCloningMLP) → arm joint targets
  • RL policy (TorchScript)          → leg joint targets
Applies PD control torques to both in a MuJoCo simulation loop.

Usage
─────
  python src/inference.py
  python src/inference.py --config config/inference_config.yaml
  python src/inference.py --config config/inference_config.yaml --stale-threshold 2.0

Prerequisites
─────────────
  1. vision_node.py must be running and publishing on the configured ZMQ address.
  2. config/inference_config.yaml must have valid relative paths to:
       - xml_path        (G1 MuJoCo scene XML)
       - policy_path     (RL walking policy .pt)
       - il_model_path   (IL arm brain .pth)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import yaml
import zmq


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL DEFINITION  (must match train.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

class BehavioralCloningMLP(nn.Module):
    """99-float landmark vector → 8 arm joint angles."""
    def __init__(self, input_dim: int = 99, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  JOINT NAME LISTS
# ──────────────────────────────────────────────────────────────────────────────

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

_IL_IDX = [0, 1, 2, 3, None, 4, 5, 6, 7, None]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_joint_ids(model, names: list[str]):
    """Return (qpos_ids, qvel_ids) for a list of joint names."""
    qpos_ids, qvel_ids = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
    return qpos_ids, qvel_ids


def get_actuator_ids(model, names: list[str]) -> list[int]:
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in names
    ]


def get_gravity_orientation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qx * qx + qy * qy),
    ])


def pd_control(
    target_q:  np.ndarray,
    q:         np.ndarray,
    kp:        np.ndarray,
    target_dq: np.ndarray,
    dq:        np.ndarray,
    kd:        np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def print_dashboard(step: int, arm_targets: np.ndarray, leg_targets: np.ndarray) -> None:
    """Overwrite terminal with a live telemetry dashboard."""
    print("\033[2J\033[H", end="")
    print("=" * 55)
    print(f"  G1 SIMULATION TELEMETRY  |  Step: {step}")
    print("=" * 55)
    print("─── ARM TARGETS (Imitation Learning) ───")
    for name, angle in zip(ARM_JOINT_NAMES, arm_targets):
        print(f"  {name:<32} {angle:>+7.3f} rad")
    print("\n─── LEG TARGETS (RL Walking Policy) ────")
    for name, angle in zip(LEG_JOINT_NAMES, leg_targets):
        print(f"  {name:<32} {angle:>+7.3f} rad")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN INFERENCE LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: Path, stale_threshold: float) -> None:

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).parent.parent

    xml_path       = repo_root / cfg["xml_path"]
    policy_path    = repo_root / cfg["policy_path"]
    il_model_path  = repo_root / cfg["il_model_path"]

    simulation_dt      = cfg["simulation_dt"]
    control_decimation = cfg["control_decimation"]
    kps                = np.array(cfg["kps"],            dtype=np.float32)
    kds                = np.array(cfg["kds"],            dtype=np.float32)
    default_angles     = np.array(cfg["default_angles"], dtype=np.float32)
    assert len(default_angles) == num_actions, (
        f"Config mismatch: default_angles has {len(default_angles)} entries "
        f"but num_actions={num_actions}. Check inference_config.yaml."
    )
    ang_vel_scale      = cfg["ang_vel_scale"]
    dof_pos_scale      = cfg["dof_pos_scale"]
    dof_vel_scale      = cfg["dof_vel_scale"]
    action_scale       = cfg["action_scale"]
    cmd_scale          = np.array(cfg["cmd_scale"],      dtype=np.float32)
    num_actions        = cfg["num_actions"]
    num_obs            = cfg["num_obs"]
    ema_alpha          = cfg.get("ema_alpha",   0.3)
    kp_arm             = cfg.get("kp_arm",    200.0)
    kd_arm             = cfg.get("kd_arm",     10.0)
    cmd                = np.array(cfg.get("cmd_velocity", [0.2, 0.0, 0.0]), dtype=np.float32)
    zmq_addr           = cfg.get("zmq_address", "tcp://127.0.0.1:5555")

    # ── MuJoCo ────────────────────────────────────────────────────────────────
    print("Initializing MuJoCo …")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    arm_qpos_ids, arm_qvel_ids = get_joint_ids(m, ARM_JOINT_NAMES)
    arm_act_ids                = get_actuator_ids(m, ARM_JOINT_NAMES)
    leg_qpos_ids, leg_qvel_ids = get_joint_ids(m, LEG_JOINT_NAMES)
    leg_act_ids                = get_actuator_ids(m, LEG_JOINT_NAMES)

    # ── AI brains ─────────────────────────────────────────────────────────────
    print("Loading AI brains …")
    policy = torch.jit.load(str(policy_path))
    policy.eval()

    il_brain = BehavioralCloningMLP(output_dim=8)
    il_brain.load_state_dict(torch.load(str(il_model_path), map_location="cpu"))
    il_brain.eval()

    # ── ZMQ subscriber ────────────────────────────────────────────────────────
    print(f"Connecting to vision node at {zmq_addr} …")
    context = zmq.Context()
    socket  = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "VISION ")
    socket.connect(zmq_addr)
    time.sleep(1.5)

    # ── State ─────────────────────────────────────────────────────────────────
    action            = np.zeros(num_actions,          dtype=np.float32)
    target_dof_pos    = default_angles.copy()
    obs               = np.zeros(num_obs,              dtype=np.float32)
    arm_target        = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
    il_smoothed       = np.zeros(8,                    dtype=np.float64)
    raw_landmarks     = None
    last_vision_time  = time.time()
    counter           = 0

    print("G1 teleoperation live — press Q in the viewer to quit.\n")

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:

        while viewer.is_running():
            step_start = time.time()

            # ── 1. RECEIVE VISION DATA (non-blocking drain) ──────────────────
            received_new = False
            try:
                while True:
                    msg           = socket.recv_string(flags=zmq.NOBLOCK)
                    payload       = json.loads(msg.replace("VISION ", "", 1))
                    raw_landmarks = payload["landmarks"]
                    last_vision_time = time.time()
                    received_new  = True
            except zmq.Again:
                pass

            # ── 2. STALE DATA GUARD ──────────────────────────────────────────
            vision_age    = time.time() - last_vision_time
            vision_active = vision_age < stale_threshold

            if not vision_active:
                # Vision node is silent — decay arms smoothly to neutral
                il_smoothed = il_smoothed * (1.0 - ema_alpha)

            elif received_new and raw_landmarks is not None:
                # Fresh data — run IL inference
                with torch.no_grad():
                    tensor    = torch.tensor(raw_landmarks, dtype=torch.float32).unsqueeze(0)
                    il_action = il_brain(tensor).squeeze().numpy()

                il_smoothed = il_action * ema_alpha + il_smoothed * (1.0 - ema_alpha)

            # ── 3. UNPACK ARM TARGETS ────────────────────────────────────────
            # IL_IDX maps ARM_JOINT_NAMES index → il_smoothed index (or None = locked to 0)
            for arm_i, il_i in enumerate(_IL_IDX):
                arm_target[arm_i] = il_smoothed[il_i] if il_i is not None else 0.0

            # ── 4. ARM PD TORQUES ────────────────────────────────────────────
            arm_tau = np.array([
                (arm_target[i] - d.qpos[arm_qpos_ids[i]]) * kp_arm
                + (-d.qvel[arm_qvel_ids[i]]) * kd_arm
                for i in range(len(arm_qpos_ids))
            ], dtype=np.float64)

            # ── 5. RL INFERENCE (LEGS) ───────────────────────────────────────
            if counter % control_decimation == 0:
                qj   = np.array([d.qpos[i] for i in leg_qpos_ids], dtype=np.float32)
                dqj  = np.array([d.qvel[i] for i in leg_qvel_ids], dtype=np.float32)
                quat  = d.qpos[3:7]
                omega = d.qvel[3:6]

                obs[:3]                          = omega * ang_vel_scale
                obs[3:6]                         = get_gravity_orientation(quat)
                obs[6:9]                         = cmd * cmd_scale
                obs[9:9+num_actions]             = (qj - default_angles) * dof_pos_scale
                obs[9+num_actions:9+2*num_actions]   = dqj * dof_vel_scale
                obs[9+2*num_actions:9+3*num_actions] = action

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action     = policy(obs_tensor).squeeze().numpy()

                target_dof_pos = action * action_scale + default_angles

            # ── 6. LEG PD TORQUES ────────────────────────────────────────────
            # Use leg_qpos_ids / leg_qvel_ids (not hardcoded slice) to avoid
            # misalignment if the XML joint ordering differs from qpos layout.
            qj_current  = np.array([d.qpos[i] for i in leg_qpos_ids], dtype=np.float32)
            dqj_current = np.array([d.qvel[i] for i in leg_qvel_ids], dtype=np.float32)

            leg_tau = pd_control(
                target_dof_pos,
                qj_current,
                kps,
                np.zeros_like(kds),
                dqj_current,
                kds,
            )

            # ── 7. APPLY & STEP ──────────────────────────────────────────────
            d.ctrl[arm_act_ids] = arm_tau
            d.ctrl[leg_act_ids] = leg_tau

            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

            if counter % 100 == 0:
                print_dashboard(counter, arm_target, target_dof_pos)

            # Real-time pacing
            elapsed = time.time() - step_start
            if m.opt.timestep - elapsed > 0:
                time.sleep(m.opt.timestep - elapsed)

    socket.close()
    context.term()
    print("Inference node stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="G1 teleoperation inference node",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="config/inference_config.yaml",
        type=Path,
        help="Path to inference_config.yaml  (default: config/inference_config.yaml)",
    )
    p.add_argument(
        "--stale-threshold",
        default=1.0,
        type=float,
        help="Seconds before vision data is considered stale, arms decay to neutral  (default: 1.0)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        config_path     = args.config,
        stale_threshold = args.stale_threshold,
    )