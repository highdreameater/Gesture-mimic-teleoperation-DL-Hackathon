# Pose-Controlled G1 — Real-Time Humanoid Teleoperation via Body Gestures

**Gesture & Pose-Controlled Autonomous Interface for the Unitree G1 Humanoid**

A real-time teleoperation system that maps human body poses — captured via webcam and MediaPipe — to joint-level control of the Unitree G1 humanoid robot in a MuJoCo simulation. Arm motion is driven by a single behavioural cloning (imitation learning) brain trained on recorded pose data. Leg locomotion is handled by a pre-trained RL walking policy.

---

## Hackathon

**Hack60 — Advanced AI & Robotics Hackathon**
Organised by **HCLTech** at **IIT Mandi**
16th April – 19th April, 2026
Problem Statement PS-8 (Track: Robotics) — *Gesture & Pose-Controlled Autonomous Interface*
Course: **CS-671 Deep Learning and Applications**

https://github.com/highdreameater/Gesture-mimic-teleoperation-DL-Hackathon/blob/main/RL_WALK.gif
## Demo
![Demo](https://github.com/highdreameater/Gesture-mimic-teleoperation-DL-Hackathon/blob/main/RL_WALK.gif)
<p align="center">
  <img src="media/demo.gif](https://github.com/highdreameater/Gesture-mimic-teleoperation-DL-Hackathon/blob/main/RL_WALK.gif
)" width="800">
</p>
## Supported Gestures

The system recognises 5 distinct body poses, each mapped to a specific robot arm configuration:

| # | Gesture | How to perform it | Robot response |
|---|---|---|---|
| 1 | **T-Pose** | Stand upright, both arms extended horizontally at shoulder height | Arms held out straight to sides — robot mirrors the neutral reference pose |
| 2 | **Rotating Arms** | Extend both arms out and trace large circles with both hands simultaneously | Robot arms sweep in a continuous outward rotation — used as a wave / attention signal |
| 3 | **Adaab** | Stand straight, raise right hand to forehead level with palm inward, slight bow of the head | Robot raises right arm in a respectful greeting salute, left arm remains at rest |
| 4 | **Boxing Defence** | Bring both fists up to face level, elbows bent and tucked inward, chin down | Both robot arms come up in a guard position — elbows forward, hands at head height |
| 5 | **Arms Sideways — Alternating** | Raise left arm out to the side while right stays down, then switch; alternate slowly | Robot mirrors the alternating lateral raises — left arm up while right is down, and vice versa |

Each gesture is recorded as a separate labelled dataset using `capture.py` and the single IL brain (`G1_bc_brain.pth`) is trained across all of them together. At inference time, the brain continuously regresses the arm joint angles that best match whatever pose the person in front of the camera is holding.

---

## Repository Structure

```
gesture-controlled-g1/
│
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── inference_config.yaml
│
├── src/
│   ├── mp2mujoco.py       # MediaPipe → G1 joint angle converter (22 DOF)
│   ├── capture.py         # Pose dataset recorder (live webcam or batch video)
│   ├── train.py           # Behavioural cloning MLP trainer
│   ├── vision_node.py     # MediaPipe vision publisher (ZMQ)
│   └── inference.py       # MuJoCo simulation + IL arm + RL leg control
│
├── models/
│   └── g1_description/    # ← see Prerequisites below
│   └── rl_policy.pt       # ← see Prerequisites below
│
├── data/
│   └── .gitkeep
│
├── debug.py               # Pipeline diagnostic script (run from project root)
└── scripts/
    ├── run_capture.sh
    ├── run_vision.sh
    └── run_inference.sh
```

---

## Prerequisites — Unitree Assets

`models/rl_policy.pt` and `models/g1_description/` are Unitree's assets distributed under the **BSD-3-Clause** licence and are not included in this repository. Clone the official Unitree RL gym and copy the relevant files into `models/` yourself:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym
# Copy the G1 MuJoCo scene XML (g1_description/) and the pre-trained
# walking policy (rl_policy.pt) into this repo's models/ directory.
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-handle>/gesture-controlled-g1
cd gesture-controlled-g1

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> MediaPipe requires Python 3.8–3.11. If you encounter an `mp.solutions` AttributeError, downgrade:
> `pip install mediapipe==0.10.9`

---

## Pipeline Overview

```
[Webcam]
   ↓
vision_node.py   →  MediaPipe pose estimation  →  ZMQ PUB (99 floats)
   ↓
inference.py     →  IL brain — G1_bc_brain.pth (arms)   →  MuJoCo arm torques
                 →  RL policy — rl_policy.pt   (legs)   →  MuJoCo leg torques
                 →  Simulation viewer
```

A **single IL model** (`G1_bc_brain.pth`) handles all 5 gestures. It is a small MLP (99 → 256 → 128 → 64 → 8) trained end-to-end on MediaPipe world landmark coordinates and regresses 8 arm joint angles (4 per arm; wrists are locked to neutral).

---

## Usage

> **All commands must be run from the project root directory.**

### Step 1 — Record gesture data

Record each gesture separately using `capture.py`. Press **SPACE** to start/stop recording, **S** to save, **Q** to quit.

```bash
bash scripts/run_capture.sh t_pose
bash scripts/run_capture.sh hands_up
bash scripts/run_capture.sh right_point
bash scripts/run_capture.sh cross_arms
bash scripts/run_capture.sh left_salute

# Or manually with more options:
python src/capture.py live --out data/gesture.h5 --label t_pose --complexity 1

# Batch processing from an existing video file:
python src/capture.py batch --src recording.mp4 --out data/t_pose.h5 --label t_pose
```

---

### Step 2 — Train the IL brain

```bash
python src/train.py

# With custom options:
python src/train.py --dataset data/gesture.h5 --output models/G1_bc_brain.pth --epochs 200
```

This trains the single `G1_bc_brain.pth` model across all recorded gesture data.

---

### Step 3 — Run the simulation

Open **two terminals**, both from the project root.

**Terminal 1 — Vision node:**
```bash
bash scripts/run_vision.sh
# or manually:
python src/vision_node.py --camera 0 --zmq-addr tcp://127.0.0.1:5555
```

**Terminal 2 — Inference + simulation:**
```bash
bash scripts/run_inference.sh
# or manually:
python src/inference.py --config config/inference_config.yaml
```

The MuJoCo viewer will open. Stand in front of your webcam and perform any of the 5 gestures — the G1 robot's arms will respond in real time while the RL policy keeps it walking.

---

## Diagnostics — `debug.py`

`debug.py` is a pipeline diagnostic script located in the **project root**. It runs all 5 stages of the pipeline in sequence and reports PASS / FAIL / SKIP for each — without requiring the MuJoCo viewer or a live simulation.

```bash
# Run from project root
python debug.py

# With custom paths:
python debug.py \
  --config config/inference_config.yaml \
  --dataset data/gesture.h5 \
  --model models/G1_bc_brain.pth \
  --zmq tcp://127.0.0.1:5555
```

### What each stage checks

| Stage | What it tests | Requires |
|---|---|---|
| 1 — mp2mujoco | Converter imports, produces correct (22,) shape, all joints within limits | Nothing |
| 2 — Dataset | HDF5 file opens, landmark/angle shapes are correct | `capture.py` run first |
| 3 — IL brain | Weights load, forward pass produces (1, 8) output | `train.py` run first |
| 4 — ZMQ | vision_node is reachable, receives a valid 99-float message | `vision_node.py` running in another terminal |
| 5 — Assets | MuJoCo XML and RL TorchScript policy load, joint/actuator counts printed | Unitree assets in `models/` |

Stages 2 and 3 show **SKIP** (not FAIL) if their files don't exist yet — this is expected before capture and training. For Stage 4, start `vision_node.py` in a separate terminal first; the ZMQ check has a 3-second timeout.

---

## Key Design Decisions

**Single IL model** — One `BehavioralCloningMLP` handles all 5 gestures. Rather than training a separate classifier or one model per gesture, the brain continuously regresses arm joint angles from raw landmark coordinates. The robot naturally interpolates between poses as the person transitions between gestures.

**22 DOF (not 23)** — The waist joints present in the G1 URDF spec are absent from the MuJoCo XML used here. The joint list was reduced accordingly: 6 left leg + 6 right leg + 5 left arm + 5 right arm = 22.

**Hybrid control** — Arms are driven by imitation learning (MediaPipe pose → IL brain → joint targets). Legs are driven by a pre-trained RL locomotion policy. This separation means arm teleoperation does not interfere with walking stability.

**ZMQ PUB/SUB** — `vision_node.py` and `inference.py` are decoupled processes communicating over ZMQ. Vision latency is independent of simulation step time. If the camera feed is lost, a stale-data guard decays the arms smoothly back to neutral.

**Wrist locking** — Both wrist roll joints are locked to 0.0 in inference. The IL brain outputs 8 values (4 per arm, wrists excluded) to reduce training complexity and avoid noisy wrist predictions from MediaPipe hand landmarks.

---

## Troubleshooting

**`mp.solutions` AttributeError on import**
Use explicit submodule imports or downgrade MediaPipe: `pip install mediapipe==0.10.9`

**Stage 4 ZMQ times out**
Make sure `vision_node.py` is running in a separate terminal before executing `debug.py`.

**MuJoCo XML not found**
Check that `xml_path` in `config/inference_config.yaml` points to the correct location of the G1 scene XML copied from `unitree_rl_gym`.

**Robot falls immediately**
Verify `config/inference_config.yaml` values for `default_angles`, `kps`, `kds`, and `num_actions` match the RL policy's expected observation space (input=47, output=12).

---

## Licence

Code in this repository: MIT.
Unitree G1 assets (`models/rl_policy.pt`, `models/g1_description/`): BSD-3-Clause — see Unitree's official repository for terms.
