"""Fit ValueNet to Monte Carlo returns from stored episode trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import param
from blockblaster.model.checkpoint import load_if_exists, save
from blockblaster.model.value_net import ValueNet
from blockblaster.train.dataset import EpisodeDataset
from blockblaster.train.logger import log_epoch


def train() -> None:
    device = torch.device(param.DEVICE)

    # ── Datasets & loaders ──────────────────────────────────────────────
    print("Loading training data...")
    train_ds = EpisodeDataset(split="train")
    test_ds = EpisodeDataset(split="test")
    print(
        f"  Train samples: {len(train_ds):,}  |  Test samples: {len(test_ds):,}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(param.DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=param.BATCH_SIZE * 4,
        shuffle=False,
        num_workers=0,
    )

    # ── Model & optimiser ───────────────────────────────────────────────
    net = ValueNet().to(device)
    meta = load_if_exists(net)
    start_epoch = 0
    # Always reset best_test_loss per run — the data distribution changes every
    # round (greedy episodes have larger returns than random ones), so MSE values
    # are not comparable across runs. We still load weights for the warm start.
    best_test_loss = math.inf
    if meta is not None:
        start_epoch = meta.get("epoch", 0)
        print(f"  Loaded weights from checkpoint (epoch {start_epoch})")

    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=param.LEARNING_RATE,
        weight_decay=param.WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\nTraining for {param.NUM_EPOCHS} epochs on {param.DEVICE}...")
    for epoch in range(start_epoch + 1, start_epoch + param.NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        for states, targets in tqdm(
            train_loader,
            desc=f"  Epoch {epoch:>4d}",
            leave=False,
            unit="batch",
        ):
            states = states.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimiser.zero_grad()
            preds = net(states)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Eval ────────────────────────────────────────────────────────
        test_loss: float | None = None
        if epoch % param.EVAL_INTERVAL_EPOCHS == 0:
            net.eval()
            total_loss = 0.0
            n_test_batches = 0
            with torch.no_grad():
                for states, targets in test_loader:
                    states = states.to(device)
                    targets = targets.to(device).unsqueeze(1)
                    preds = net(states)
                    total_loss += loss_fn(preds, targets).item()
                    n_test_batches += 1
            test_loss = total_loss / max(n_test_batches, 1)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save(net, epoch, best_test_loss)
                print(f"  -> Checkpoint saved (epoch={epoch}, test_loss={best_test_loss:.4f})")

        log_epoch(epoch, avg_train_loss, test_loss, best_test_loss)

    # Always persist the final model so the next simulate round uses it,
    # even if no eval epoch happened to beat best_test_loss this run.
    final_epoch = start_epoch + param.NUM_EPOCHS
    if best_test_loss == math.inf:
        save(net, final_epoch, best_test_loss)
        print(f"  -> Checkpoint saved (epoch={final_epoch}, no eval yet)")

    print(f"\nTraining complete.  Best test loss this run: {best_test_loss:.4f}")
    print(f"Checkpoint: {param.CHECKPOINT_PATH}")
