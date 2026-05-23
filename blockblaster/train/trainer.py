"""Fit ValueNet to n-step TD targets bootstrapped from a frozen target net.

For each training sample `(s_t, s_next, n_step_sum, phi_t, phi_next, bootstrap)`
from `EpisodeDataset`, the shaped target the net is trained against is

    target_F(s_t) = n_step_sum
                  + bootstrap * γ^n * (V_target(s_next) + phi_next)
                  - phi_t

The net outputs V_F = V* - Φ (same convention as the previous MC trainer), so
the action-selection code in `blockblaster.agent.policy` keeps working unchanged:
V*(s) = V_F(s) + Φ(s).

`V_target` is a frozen copy of the live net, refreshed every
`param.TARGET_REFRESH_BATCHES` minibatches.  This is what lets training make
progress even when the replay buffer is dominated by one policy: the targets
shift as the net improves, instead of being pinned to the fixed-point of
V^π_buffer.
"""

from __future__ import annotations

import copy
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


def _compute_target(
    target_net: ValueNet,
    s_next: torch.Tensor,
    n_step_sum: torch.Tensor,
    phi_t: torch.Tensor,
    phi_next: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma_n: float,
) -> torch.Tensor:
    """target_F = n_step_sum + bootstrap * γ^n * (V_target(s_next) + φ_next) - φ_t.

    Computed under `no_grad` because the target net is frozen between
    refreshes; gradients only flow through the *live* net's prediction.
    """
    with torch.no_grad():
        v_next_F = target_net(s_next).squeeze(1)
        v_next_star = v_next_F + phi_next
        target = n_step_sum + bootstrap * gamma_n * v_next_star - phi_t
    return target


def train() -> None:
    device = torch.device(param.DEVICE)

    print("Loading training data...")
    train_ds = EpisodeDataset(split="train")
    test_ds = EpisodeDataset(split="test")
    print(
        f"  Train samples: {len(train_ds):,}  |  Test samples: {len(test_ds):,}"
    )
    print(
        f"  TD n-step: {train_ds.n_step}  |  γ^n: {train_ds.gamma_n:.4f}  |  "
        f"target refresh: every {param.TARGET_REFRESH_BATCHES} batches"
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
    best_test_loss = math.inf
    if meta is not None:
        start_epoch = meta.get("epoch", 0)
        print(f"  Loaded weights from checkpoint (epoch {start_epoch})")

    # Target net: frozen copy of the live net, refreshed periodically.
    # `deepcopy` shares no parameters with `net`, so updates to `net`
    # don't affect `target_net` until we explicitly copy weights.
    target_net = copy.deepcopy(net).to(device)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    # Adam state is intentionally NOT restored across rounds.  Stale
    # second-moment estimates over hundreds of epochs collapse the effective
    # per-parameter LR and prevent the net from moving when it needs to;
    # we'd rather pay the "fresh Adam warmup" cost each round than be stuck.
    # See `blockblaster.model.checkpoint` for matching save-side behaviour.
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=param.LEARNING_RATE,
        weight_decay=param.WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    gamma_n = train_ds.gamma_n
    refresh_every = max(1, param.TARGET_REFRESH_BATCHES)
    global_step = 0

    print(f"\nTraining for {param.NUM_EPOCHS} epochs on {param.DEVICE}...")
    for epoch in range(start_epoch + 1, start_epoch + param.NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in tqdm(
            train_loader,
            desc=f"  Epoch {epoch:>4d}",
            leave=False,
            unit="batch",
        ):
            s_t, s_next, n_step_sum, phi_t, phi_next, bootstrap = batch
            s_t = s_t.to(device, non_blocking=True)
            s_next = s_next.to(device, non_blocking=True)
            n_step_sum = n_step_sum.to(device, non_blocking=True)
            phi_t = phi_t.to(device, non_blocking=True)
            phi_next = phi_next.to(device, non_blocking=True)
            bootstrap = bootstrap.to(device, non_blocking=True)

            target = _compute_target(
                target_net, s_next, n_step_sum, phi_t, phi_next, bootstrap, gamma_n
            )

            optimiser.zero_grad()
            preds = net(s_t).squeeze(1)
            loss = loss_fn(preds, target)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Refresh target network periodically.
            if global_step % refresh_every == 0:
                target_net.load_state_dict(net.state_dict())

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Eval ────────────────────────────────────────────────────────
        test_loss: float | None = None
        if epoch % param.EVAL_INTERVAL_EPOCHS == 0:
            net.eval()
            total_loss = 0.0
            n_test_batches = 0
            with torch.no_grad():
                for batch in test_loader:
                    s_t, s_next, n_step_sum, phi_t, phi_next, bootstrap = batch
                    s_t = s_t.to(device, non_blocking=True)
                    s_next = s_next.to(device, non_blocking=True)
                    n_step_sum = n_step_sum.to(device, non_blocking=True)
                    phi_t = phi_t.to(device, non_blocking=True)
                    phi_next = phi_next.to(device, non_blocking=True)
                    bootstrap = bootstrap.to(device, non_blocking=True)

                    target = _compute_target(
                        target_net, s_next, n_step_sum, phi_t, phi_next, bootstrap, gamma_n
                    )
                    preds = net(s_t).squeeze(1)
                    total_loss += loss_fn(preds, target).item()
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
