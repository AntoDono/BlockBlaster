"""Train the Block Blast queue piece classifier on synthetic data.

Two phases:
  1. Pre-generate the full synthetic dataset in RAM, in parallel across N
     CPU workers (this is the slow part — it's all `cv2` + `numpy` on CPU).
  2. Train the small CNN on the pre-generated tensors with large batches
     so the GPU stays saturated.

Tweak the constants in the CONFIG section to change dataset size, worker
count, batch size, etc.  No CLI args by design — just run:

    uv run train_piece_cnn.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from blockblaster.piece_cnn import (
    DEFAULT_WEIGHT_PATH,
    NUM_CLASSES,
    PieceCNN,
    pregenerate_dataset,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_TRAIN_SAMPLES = 100_000     # synth examples for training
NUM_VAL_SAMPLES   = 10_000      # held-out synth examples for validation
NUM_WORKERS       = max(1, (os.cpu_count() or 4) - 1)  # CPU procs for synth
BATCH_SIZE        = 1024        # large — model is tiny, GPU is bored
NUM_EPOCHS        = 12
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
TARGET_VAL_ACC    = 0.995       # stop early if reached


def _to_tensor(images_bgr: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) BGR uint8  →  (N, 3, H, W) float32 RGB in [0, 1]."""
    rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))


def _eval(net: PieceCNN, x: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = net(x[i : i + batch])
            correct += (logits.argmax(dim=1) == y[i : i + batch]).sum().item()
    net.train()
    return correct / len(x)


def main() -> None:
    out_path = DEFAULT_WEIGHT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  output={out_path.resolve()}")
    print(f"[train] {NUM_CLASSES} classes (32 pieces + empty)")

    # ── Phase 1: pre-generate dataset ────────────────────────────────────
    print(f"\n[phase 1] pre-generating "
          f"{NUM_TRAIN_SAMPLES:,} train + {NUM_VAL_SAMPLES:,} val samples "
          f"on {NUM_WORKERS} workers")
    t0 = time.time()
    train_imgs, train_lbls = pregenerate_dataset(
        NUM_TRAIN_SAMPLES, n_workers=NUM_WORKERS, seed=0, progress=True,
    )
    val_imgs, val_lbls = pregenerate_dataset(
        NUM_VAL_SAMPLES, n_workers=NUM_WORKERS, seed=10**7, progress=True,
    )
    print(f"[phase 1] dataset ready in {time.time() - t0:.1f}s "
          f"(RAM: {(train_imgs.nbytes + val_imgs.nbytes) / 1e6:.0f} MB)")

    # Move full tensors to device (small enough to fit on any GPU)
    train_x = _to_tensor(train_imgs).to(device)
    train_y = torch.from_numpy(train_lbls).to(device)
    val_x   = _to_tensor(val_imgs).to(device)
    val_y   = torch.from_numpy(val_lbls).to(device)
    del train_imgs, val_imgs   # free CPU RAM

    # ── Phase 2: train ───────────────────────────────────────────────────
    print(f"\n[phase 2] training "
          f"epochs={NUM_EPOCHS}  batch={BATCH_SIZE}  "
          f"steps/epoch={len(train_x) // BATCH_SIZE}")

    net = PieceCNN().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[phase 2] model params: {n_params:,}")
    opt = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    n_train = len(train_x)
    best_acc = 0.0
    train_t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # Shuffle each epoch
        perm = torch.randperm(n_train, device=device)

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0
        ep_t0 = time.time()
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            x = train_x[idx]
            y = train_y[idx]

            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss   += loss.item()
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            n_batches    += 1

        train_acc = epoch_correct / n_train
        val_acc   = _eval(net, val_x, val_y, batch=BATCH_SIZE)
        ep_time   = time.time() - ep_t0

        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), out_path)
            flag = "  ↳ saved"

        print(f"  epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"loss={epoch_loss / n_batches:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"({ep_time:.1f}s){flag}")

        if val_acc >= TARGET_VAL_ACC:
            print(f"[phase 2] hit target val_acc {TARGET_VAL_ACC}, stopping early")
            break

    print(f"\n[done] best val_acc={best_acc:.4f}  "
          f"total train time {time.time() - train_t0:.1f}s  "
          f"weights → {out_path}")


if __name__ == "__main__":
    main()
