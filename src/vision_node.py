"""
vision_node.py  –  MediaPipe Vision Node
==========================================
Captures pose from webcam via MediaPipe and broadcasts 99-float world-landmark
vectors over ZMQ PUB/SUB so inference.py can perform IL arm control.

Key detail: pose_WORLD_landmarks (metric, hip-centred) are used, NOT
pose_landmarks (image-space, normalised). The IL brain was trained on
world coordinates, so this must stay consistent.

Usage
─────
  python src/vision_node.py
  python src/vision_node.py --camera 1 --zmq-addr tcp://127.0.0.1:5556
  python src/vision_node.py --complexity 2
"""

import argparse
import json
import time

import cv2
import mediapipe as mp
import zmq

# ── MediaPipe drawing utils ───────────────────────────────────────────────
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose  = mp.solutions.pose

# ──────────────────────────────────────────────────────────────────────────────
# POSE SETUP
# ──────────────────────────────────────────────────────────────────────────────

def _make_pose(complexity: int):
    return mp_pose.Pose(
        static_image_mode        = False,
        model_complexity         = complexity,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(camera: int, zmq_addr: str, complexity: int) -> None:
    # ── ZMQ Publisher ────────────────────────────────────────────────────────
    context = zmq.Context()
    socket  = context.socket(zmq.PUB)
    socket.bind(zmq_addr)

    print("Vision Node starting …")
    print(f"  ZMQ publisher bound on {zmq_addr}")


    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}.")

    print(f"  Camera {camera} active — broadcasting world landmarks …\n")

    prev_time   = time.time()
    frames_sent = 0

    try:
        with _make_pose(complexity) as pose:
            while True:
                ok, img = cap.read()
                if not ok:
                    print("Empty camera frame — skipping.")
                    continue

                rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                pose_detected = False

                if results.pose_world_landmarks:
                    lms = results.pose_world_landmarks.landmark

                    # 33 landmarks × 3 coords = 99 floats (world-space, metric, hip-centred)
                    flat = []
                    for lm in lms:
                        flat.extend([lm.x, lm.y, lm.z])

                    payload = json.dumps({"landmarks": flat, "timestamp": time.time()})
                    socket.send_string(f"VISION {payload}")
                    frames_sent  += 1
                    pose_detected = True

                    # Draw skeleton for visual feedback
                    mp_draw.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_style.get_default_pose_landmarks_style(),
                    )

                # ── HUD ──────────────────────────────────────────────────────
                now       = time.time()
                fps       = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                status_txt = "BROADCASTING" if pose_detected else "NO POSE"
                status_col = (0, 255, 0)     if pose_detected else (0, 60, 255)

                cv2.putText(img, f"FPS: {fps:.0f}",      (16, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
                cv2.putText(img, status_txt,              (16, 76),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col,    2)
                cv2.putText(img, f"Sent: {frames_sent}",  (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(200,200,200), 1)

                cv2.imshow("Vision Node", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        socket.close()
        context.term()
        print(f"Vision node stopped. Total frames sent: {frames_sent}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediaPipe vision node — streams landmarks over ZMQ",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--camera",     default=0,                      type=int,
                   help="Camera device index                     (default: 0)")
    p.add_argument("--zmq-addr",   default="tcp://127.0.0.1:5555", type=str,
                   help="ZMQ bind address                        (default: tcp://127.0.0.1:5555)")
    p.add_argument("--complexity", default=1,                      type=int, choices=[0, 1, 2],
                   help="MediaPipe model complexity 0=fast 2=best (default: 1)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        camera     = args.camera,
        zmq_addr   = args.zmq_addr,
        complexity = args.complexity,
    )