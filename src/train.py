"""
train.py  –  Behavioral Cloning Trainer
=========================================
Trains a small MLP (BehavioralCloningMLP) on recorded G1 pose data.
Input:  99-float raw MediaPipe world landmarks  (33 landmarks × 3 coords)
Output: 8 arm joint angles                      (4 left + 4 right, wrists locked)

Joint index reference (from mp2mujoco.py G1_JOINTS, 0-indexed):
  15 – left_shoulder_pitch
  16 – left_shoulder_roll
  17 – left_shoulder_yaw
  18 – left_elbow
  19 – left_wrist_roll       ← intentionally skipped (wrist locked in inference)
  20 – right_shoulder_pitch
  21 – right_shoulder_roll
  22 – right_shoulder_yaw
  23 – right_elbow
  24 – right_wrist_roll      ← intentionally skipped (wrist locked in inference)

Usage
─────
  python src/train.py
  python src/train.py --dataset data/gesture.h5 --output models/G1_bc_brain.pth
  python src/train.py --epochs 200 --batch-size 128 --lr 0.0005
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL
# ──────────────────────────────────────────────────────────────────────────────

class BehavioralCloningMLP(nn.Module):
    """
    Maps 99-float landmark vector → 8 arm joint angles.
    Architecture: 99 → 256 → 128 → 64 → 8
    """
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
# 2.  DATASET
# ──────────────────────────────────────────────────────────────────────────────

class G1ImitationDataset(Dataset):
    """
    Loads an HDF5 dataset produced by capture.py.

    X : (N, 99)  – raw MediaPipe world landmarks flattened
    Y : (N,  8)  – arm joint angles [L-shoulder×3, L-elbow, R-shoulder×3, R-elbow]

    Index mapping:
      Y[:, 0:4] = all_angles[:, 15:19]  (left  shoulder pitch/roll/yaw + elbow)
      Y[:, 4:8] = all_angles[:, 20:24]  (right shoulder pitch/roll/yaw + elbow)
      Index 19 (left_wrist_roll)  and index 24 (right_wrist_roll) are intentionally
      excluded — both wrists are locked to 0.0 in inference.py.
    """

    def __init__(self, filepath: str | Path):
        filepath = Path(filepath)
        print(f"[Dataset] Loading {filepath} …")

        with h5py.File(filepath, "r") as f:
            raw_landmarks = f["landmarks"][:]       # (N, 33, 3)
            all_angles    = f["angles"][:].astype(np.float32)  # (N, 23)

        # Flatten landmarks to (N, 99)
        self.X = raw_landmarks.reshape(raw_landmarks.shape[0], -1).astype(np.float32)

        # Extract 4 left-arm + 4 right-arm joints (wrists excluded deliberately)
        left_arm  = all_angles[:, 12:16]   # left_shoulder_pitch/roll/yaw + elbow
        right_arm = all_angles[:, 17:21]   # right_shoulder_pitch/roll/yaw + elbow
        self.Y = np.concatenate([left_arm, right_arm], axis=1)  # (N, 8)

        print(f"[Dataset] {self.X.shape[0]} frames | "
              f"input={self.X.shape[1]}  output={self.Y.shape[1]}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(
    dataset_path: Path,
    output_path:  Path,
    epochs:       int,
    batch_size:   int,
    lr:           float,
) -> None:

    dataset    = G1ImitationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = BehavioralCloningMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[Train] epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"[Train] Output → {output_path}\n")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            predictions  = model(inputs)
            loss         = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            avg = running_loss / len(dataloader)
            print(f"  Epoch [{epoch:>4}/{epochs}]  loss: {avg:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n[Train] Done. Model saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train behavioral cloning MLP on G1 pose data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--dataset",    default="data/gesture.h5",        type=Path,
                   help="Path to HDF5 dataset from capture.py  (default: data/gesture.h5)")
    p.add_argument("--output",     default="models/G1_bc_brain.pth", type=Path,
                   help="Where to save trained weights          (default: models/G1_bc_brain.pth)")
    p.add_argument("--epochs",     default=100,  type=int,
                   help="Number of training epochs              (default: 100)")
    p.add_argument("--batch-size", default=64,   type=int,
                   help="Mini-batch size                        (default: 64)")
    p.add_argument("--lr",         default=1e-3, type=float,
                   help="Adam learning rate                     (default: 0.001)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        dataset_path = args.dataset,
        output_path  = args.output,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
    )