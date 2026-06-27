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
from tqdm import tqdm

from blockblaster.piece_cnn import (
    DEFAULT_DATA_DIR,
    DEFAULT_WEIGHT_PATH,
    NUM_CLASSES,
    PieceCNN,
    load_real_dataset,
    pregenerate_dataset,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_TRAIN_SAMPLES = 2_500     # synth examples for training
NUM_VAL_SAMPLES   = 2_000      # held-out synth examples for validation
NUM_WORKERS       = max(1, (os.cpu_count() or 4) - 1)  # CPU procs for synth
BATCH_SIZE        = 2048        # large — model is tiny, GPU is bored
NUM_EPOCHS        = 20
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
TARGET_VAL_ACC    = 0.995       # stop early if reached

# ── Real (collected) data ─────────────────────────────────────────────────────
REAL_DATA_DIR  = DEFAULT_DATA_DIR  # data/pieces/<label>/<uuid>.png
REAL_VAL_FRAC  = 0.15    # fraction of real crops held out for the real-val metric
REAL_OVERSAMPLE = 50     # repeat each real train crop N× so it isn't drowned by synth
REAL_SPLIT_SEED = 123    # deterministic real train/val split


def _to_tensor(images_bgr: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) BGR uint8  →  (N, 3, H, W) float32 RGB in [0, 1]."""
    rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))


def _load_real_split() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load collected crops and split into ``(tr_imgs, tr_lbls, val_imgs, val_lbls)``.

    The train half is oversampled by ``REAL_OVERSAMPLE`` so it carries weight
    against the much larger synthetic set. Returns empty arrays if no real data
    is present, in which case training falls back to synth-only.
    """
    imgs, lbls = load_real_dataset(REAL_DATA_DIR)
    n = len(imgs)
    if n == 0:
        empty_i = np.empty((0, *imgs.shape[1:]), dtype=imgs.dtype)
        empty_l = np.empty((0,), dtype=lbls.dtype)
        return empty_i, empty_l, empty_i.copy(), empty_l.copy()

    rng  = np.random.default_rng(REAL_SPLIT_SEED)
    perm = rng.permutation(n)
    imgs, lbls = imgs[perm], lbls[perm]

    n_val = max(1, int(round(n * REAL_VAL_FRAC))) if n > 1 else 0
    val_imgs, val_lbls = imgs[:n_val], lbls[:n_val]
    tr_imgs,  tr_lbls  = imgs[n_val:], lbls[n_val:]

    if REAL_OVERSAMPLE > 1 and len(tr_imgs):
        tr_imgs = np.repeat(tr_imgs, REAL_OVERSAMPLE, axis=0)
        tr_lbls = np.repeat(tr_lbls, REAL_OVERSAMPLE, axis=0)

    return tr_imgs, tr_lbls, val_imgs, val_lbls


def _eval(
    net: PieceCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    device: torch.device,
) -> float:
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i : i + batch].to(device, non_blocking=True)
            yb = y[i : i + batch].to(device, non_blocking=True)
            logits = net(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
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
        NUM_TRAIN_SAMPLES, n_workers=NUM_WORKERS, seed=0, desc="  train pregen",
    )
    val_imgs, val_lbls = pregenerate_dataset(
        NUM_VAL_SAMPLES, n_workers=NUM_WORKERS, seed=10**7, desc="    val pregen",
    )

    # Mix in collected real crops: oversampled into the train set, with a held-out
    # slice tracked as a separate "real val" accuracy.
    real_tr_imgs, real_tr_lbls, real_val_imgs, real_val_lbls = _load_real_split()
    n_real_tr  = len(real_tr_imgs)
    n_real_val = len(real_val_imgs)
    if n_real_tr or n_real_val:
        n_unique_tr = n_real_tr // REAL_OVERSAMPLE if REAL_OVERSAMPLE > 1 else n_real_tr
        print(f"[phase 1] real data: {n_unique_tr} train crops "
              f"(×{REAL_OVERSAMPLE} = {n_real_tr}) + {n_real_val} val crops")
        if n_real_tr:
            train_imgs = np.concatenate([train_imgs, real_tr_imgs], axis=0)
            train_lbls = np.concatenate([train_lbls, real_tr_lbls], axis=0)
    else:
        print(f"[phase 1] no real data found under {REAL_DATA_DIR} — synth only")

    print(f"[phase 1] dataset ready in {time.time() - t0:.1f}s "
          f"(RAM: {(train_imgs.nbytes + val_imgs.nbytes) / 1e6:.0f} MB)")

    # Keep the full tensors in pinned CPU RAM and copy each batch to the
    # GPU on-the-fly. Datasets large enough to exceed GPU memory (e.g.
    # 250k samples ≈ 26 GiB as float32) train fine this way; the PCIe
    # copy overlaps with the previous batch's compute via non_blocking=True.
    pin     = device.type == "cuda"
    train_x = _to_tensor(train_imgs).pin_memory() if pin else _to_tensor(train_imgs)
    train_y = torch.from_numpy(train_lbls).pin_memory() if pin else torch.from_numpy(train_lbls)
    val_x   = _to_tensor(val_imgs).pin_memory() if pin else _to_tensor(val_imgs)
    val_y   = torch.from_numpy(val_lbls).pin_memory() if pin else torch.from_numpy(val_lbls)
    del train_imgs, val_imgs   # free the uint8 staging copies

    has_real_val = n_real_val > 0
    if has_real_val:
        real_val_x = _to_tensor(real_val_imgs)
        real_val_y = torch.from_numpy(real_val_lbls)
        if pin:
            real_val_x = real_val_x.pin_memory()
            real_val_y = real_val_y.pin_memory()

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
        perm = torch.randperm(n_train)
        batch_starts = list(range(0, n_train, BATCH_SIZE))

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0
        ep_t0 = time.time()

        bar = tqdm(
            batch_starts,
            desc=f"  Epoch {epoch:>2d}/{NUM_EPOCHS}",
            leave=False,
            unit="batch",
        )
        for i in bar:
            idx = perm[i : i + BATCH_SIZE]
            x = train_x[idx].to(device, non_blocking=True)
            y = train_y[idx].to(device, non_blocking=True)

            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss    += loss.item()
            batch_correct  = (logits.argmax(dim=1) == y).sum().item()
            epoch_correct += batch_correct
            n_batches     += 1

            bar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{batch_correct / len(y):.3f}",
            )

        train_acc = epoch_correct / n_train
        val_acc   = _eval(net, val_x, val_y, batch=BATCH_SIZE, device=device)
        real_acc  = (
            _eval(net, real_val_x, real_val_y, batch=BATCH_SIZE, device=device)
            if has_real_val else None
        )
        ep_time   = time.time() - ep_t0

        # When real data is present, checkpoint on real-val accuracy — that's the
        # metric we actually care about. Otherwise fall back to synth-val.
        monitor = real_acc if has_real_val else val_acc

        flag = ""
        if monitor > best_acc:
            best_acc = monitor
            torch.save(net.state_dict(), out_path)
            flag = "  ↳ saved"

        real_str = f"  real_val_acc={real_acc:.4f}" if has_real_val else ""
        print(f"  Epoch {epoch:>2d}  loss={epoch_loss / n_batches:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}{real_str}  "
              f"({ep_time:.1f}s){flag}")

        if monitor >= TARGET_VAL_ACC:
            metric = "real_val_acc" if has_real_val else "val_acc"
            print(f"[phase 2] hit target {metric} {TARGET_VAL_ACC}, stopping early")
            break

    best_metric = "real_val_acc" if has_real_val else "val_acc"
    print(f"\n[done] best {best_metric}={best_acc:.4f}  "
          f"total train time {time.time() - train_t0:.1f}s  "
          f"weights → {out_path}")


if __name__ == "__main__":
    main()
