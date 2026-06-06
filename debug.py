"""
debug.py  –  Pipeline Diagnostic Script
=========================================
Runs all 5 stages of the pipeline in sequence and reports pass/fail for each.
Does NOT require the webcam or MuJoCo viewer to be open.

Usage
─────
  python debug.py
  python debug.py --config config/inference_config.yaml
  python debug.py --dataset data/gesture.h5

Stages
──────
  1. mp2mujoco  – converter imports, produces correct shape on fake landmarks
  2. dataset    – HDF5 file opens, shapes are correct
  3. il_brain   – weights load, output shape is (1, 8)
  4. zmq        – vision_node is reachable, receives a valid 99-float message
                  (requires vision_node.py to already be running in another terminal)
  5. assets     – XML and RL policy load without error, joint/actuator counts printed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.mp2mujoco import DOF


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"

def header(n: int, title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  Stage {n}: {title}")
    print(f"{'─'*55}")

def ok(msg: str) -> None:
    print(f"{PASS}  {msg}")

def fail(msg: str, exc: Exception | None = None) -> None:
    print(f"{FAIL}  {msg}")
    if exc:
        print(f"         {type(exc).__name__}: {exc}")

def skip(msg: str) -> None:
    print(f"{SKIP}  {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — mp2mujoco converter
# ──────────────────────────────────────────────────────────────────────────────

def stage_mp2mujoco() -> bool:
    header(1, "mp2mujoco converter")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from mp2mujoco import MediaPipeToG1, JOINT_NAMES, DOF, G1_JOINTS

        class FakeLM:
            x = y = z = 0.0
            visibility = 1.0

        converter = MediaPipeToG1()
        lms       = [FakeLM() for _ in range(33)]
        frame     = converter.convert(lms)
        
        assert frame.angles.shape == (DOF,),  f"Expected ({DOF},), got {frame.angles.shape}"
        assert DOF == 22,                     f"Expected DOF=22, got {DOF}"
        assert frame.confidence == 1.0,      f"Expected confidence=1.0, got {frame.confidence}"

        # Check all angles are within joint limits
        violations = []
        for i, (jspec, angle) in enumerate(zip(G1_JOINTS, frame.angles)):
            if not (jspec.lo <= angle <= jspec.hi):
                violations.append(f"{jspec.name}: {angle:.4f} not in [{jspec.lo}, {jspec.hi}]")
        if violations:
            fail(f"Joint limit violations: {violations}")
            return False

        ok(f"DOF={DOF}, angles shape={frame.angles.shape}, all within limits")
        for name, angle in zip(JOINT_NAMES, frame.angles):
            print(f"         {name:<30} {angle:+.4f} rad")
        return True

    except Exception as e:
        fail("mp2mujoco import or conversion failed", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — HDF5 dataset
# ──────────────────────────────────────────────────────────────────────────────

def stage_dataset(dataset_path: Path) -> bool:
    header(2, f"Dataset  ({dataset_path})")

    if not dataset_path.exists():
        skip(f"{dataset_path} not found — run capture.py first, then re-run debug.py")
        return True   # not a hard failure; rest of pipeline can still be tested

    try:
        import h5py
        import numpy as np

        with h5py.File(dataset_path, "r") as f:
            keys = list(f.keys())
            lm_shape  = f["landmarks"].shape    # (N, 33, 3)
            ang_shape = f["angles"].shape        # (N, 23)
            conf_mean = float(f["confidence"][:].mean())

        assert "landmarks"  in keys, "Missing key: landmarks"
        assert "angles"     in keys, "Missing key: angles"
        assert "confidence" in keys, "Missing key: confidence"
        assert lm_shape[1]  == 33,   f"Expected 33 landmarks, got {lm_shape[1]}"
        assert lm_shape[2]  == 3,    f"Expected 3 coords, got {lm_shape[2]}"
        assert ang_shape[1] == 23,   f"Expected 23 angles, got {ang_shape[1]}"

        ok(f"landmarks={lm_shape}  angles={ang_shape}  mean_confidence={conf_mean:.3f}")
        return True

    except Exception as e:
        fail("Dataset load failed", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — IL brain weights
# ──────────────────────────────────────────────────────────────────────────────

def stage_il_brain(model_path: Path) -> bool:
    header(3, f"IL brain  ({model_path})")

    if not model_path.exists():
        skip(f"{model_path} not found — run train.py first, then re-run debug.py")
        return True

    try:
        import torch
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from train import BehavioralCloningMLP

        model = BehavioralCloningMLP()
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()

        dummy = torch.zeros(1, 99)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (1, 8), f"Expected output (1, 8), got {out.shape}"
        ok(f"Weights loaded, output shape={tuple(out.shape)}, values={out.numpy().round(4)}")
        return True

    except Exception as e:
        fail("IL brain load or forward pass failed", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — ZMQ vision node
# ──────────────────────────────────────────────────────────────────────────────

def stage_zmq(zmq_addr: str, timeout_ms: int = 3000) -> bool:
    header(4, f"ZMQ vision node  ({zmq_addr})")
    print(f"         Waiting up to {timeout_ms/1000:.1f}s for a message …")
    print(f"         (vision_node.py must be running in another terminal)")

    try:
        import json
        import zmq

        context = zmq.Context()
        socket  = context.socket(zmq.SUB)
        socket.connect(zmq_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "VISION ")
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        try:
            msg     = socket.recv_string()
            payload = json.loads(msg.replace("VISION ", "", 1))
            lms     = payload["landmarks"]
            ts      = payload["timestamp"]

            assert len(lms) == 99, f"Expected 99 floats, got {len(lms)}"
            assert all(isinstance(v, float) for v in lms[:3]), "Landmarks are not floats"

            age = time.time() - ts
            ok(f"Received message: len={len(lms)}, timestamp age={age:.3f}s")
            ok(f"First 6 values: {[round(v,4) for v in lms[:6]]}")
            return True

        except zmq.Again:
            skip("No message received within timeout — is vision_node.py running?")
            return True   # not a hard failure

        finally:
            socket.close()
            context.term()

    except Exception as e:
        fail("ZMQ connection failed", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — MuJoCo XML + RL policy
# ──────────────────────────────────────────────────────────────────────────────

def stage_assets(config_path: Path) -> bool:
    header(5, f"MuJoCo XML + RL policy  (config: {config_path})")

    if not config_path.exists():
        fail(f"Config not found: {config_path}")
        return False

    try:
        import yaml
        import mujoco
        import torch

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        root       = Path(__file__).parent
        xml_path   = root / cfg["xml_path"]
        policy_path= root / cfg["policy_path"]

        # XML
        if not xml_path.exists():
            fail(f"XML not found: {xml_path}")
            return False

        m = mujoco.MjModel.from_xml_path(str(xml_path))
        ok(f"XML loaded: njnt={m.njnt}  nu={m.nu}  nq={m.nq}  nv={m.nv}")

        # Joint names
        joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        print(f"         Joints: {joint_names}")

        # RL policy
        if not policy_path.exists():
            fail(f"RL policy not found: {policy_path}")
            return False

        policy = torch.jit.load(str(policy_path))
        policy.eval()

        num_obs     = cfg["num_obs"]
        num_actions = cfg["num_actions"]
        dummy_obs   = torch.zeros(1, num_obs)

        with torch.no_grad():
            out = policy(dummy_obs)

        assert out.shape[1] == num_actions, \
            f"Expected policy output ({num_actions},), got {out.shape}"

        ok(f"RL policy loaded: input={num_obs}  output={tuple(out.shape)}")
        return True

    except Exception as e:
        fail("Asset load failed", e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline diagnostic — runs all 5 stages")
    p.add_argument("--config",  default="config/inference_config.yaml", type=Path,
                   help="Path to inference_config.yaml")
    p.add_argument("--dataset", default="data/gesture.h5",              type=Path,
                   help="Path to HDF5 dataset")
    p.add_argument("--model",   default="models/G1_bc_brain.pth",       type=Path,
                   help="Path to trained IL brain weights")
    p.add_argument("--zmq",     default="tcp://127.0.0.1:5555",         type=str,
                   help="ZMQ address to listen on for vision node check")
    p.add_argument("--zmq-timeout", default=3000, type=int,
                   help="ZMQ receive timeout in ms  (default: 3000)")
    args = p.parse_args()

    print("\n" + "=" * 55)
    print("  G1 PIPELINE DIAGNOSTIC")
    print("=" * 55)

    results = {
        "mp2mujoco" : stage_mp2mujoco(),
        "dataset"   : stage_dataset(args.dataset),
        "il_brain"  : stage_il_brain(args.model),
        "zmq"       : stage_zmq(args.zmq, args.zmq_timeout),
        "assets"    : stage_assets(args.config),
    }

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name:<12} {status}")
        if not passed:
            all_passed = False

    print(f"{'='*55}\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()