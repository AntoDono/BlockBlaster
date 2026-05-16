"""Train the Block Blast queue piece classifier on synthetic data.

Run once (no arguments needed):

    uv run train_piece_cnn.py

Generates training samples on the fly via :mod:`blockblaster.assist.piece_synth`,
trains the small CNN in :mod:`blockblaster.assist.piece_cnn`, evaluates on a
held-out synthetic validation batch, and saves the weights to ``piece_cnn.pt``
in the project root.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from blockblaster.piece_cnn import (
    DEFAULT_WEIGHT_PATH,
    NUM_CLASSES,
    PieceCNN,
    generate_batch,
    preprocess_batch,
)

# Hyperparameters — tiny network, synthetic data: training is fast.
NUM_STEPS         = 1500
BATCH_SIZE        = 128
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
LOG_EVERY         = 50
VAL_EVERY         = 250
VAL_BATCH         = 1024
TARGET_VAL_ACC    = 0.99


def _eval(net: PieceCNN, device: torch.device, rng: random.Random) -> float:
    net.eval()
    with torch.no_grad():
        imgs, labels = generate_batch(VAL_BATCH, rng)
        x = preprocess_batch(imgs).to(device)
        y = torch.from_numpy(labels).to(device)
        preds = net(x).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    net.train()
    return acc


def main() -> None:
    out_path = DEFAULT_WEIGHT_PATH
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  output={out_path.resolve()}")
    print(f"[train] {NUM_CLASSES} classes (32 pieces + empty)")

    train_rng = random.Random(0)
    val_rng   = random.Random(99)

    net = PieceCNN().to(device)
    opt = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] model params: {n_params:,}")

    start = time.time()
    best_acc = 0.0
    for step in range(1, NUM_STEPS + 1):
        imgs, labels = generate_batch(BATCH_SIZE, train_rng)
        x = preprocess_batch(imgs).to(device)
        y = torch.from_numpy(labels).to(device)

        logits = net(x)
        loss   = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                train_acc = (logits.argmax(dim=1) == y).float().mean().item()
            elapsed = time.time() - start
            print(f"  step {step:5d}/{NUM_STEPS}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  ({elapsed:.1f}s)")

        if step % VAL_EVERY == 0:
            acc = _eval(net, device, val_rng)
            print(f"  ── val_acc={acc:.4f} ─────────────────────────")
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), out_path)
                print(f"     ↳ saved checkpoint ({acc:.4f}) → {out_path}")
            if acc >= TARGET_VAL_ACC:
                print(f"  reached target accuracy {TARGET_VAL_ACC}, stopping early")
                break

    # Final eval/save in case the loop exited without a val tick on the last step
    final_acc = _eval(net, device, val_rng)
    print(f"[train] final val_acc={final_acc:.4f}  (best={max(best_acc, final_acc):.4f})")
    if final_acc >= best_acc:
        torch.save(net.state_dict(), out_path)
        print(f"[train] final weights saved → {out_path}")


if __name__ == "__main__":
    main()
