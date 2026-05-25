"""Evaluate the piece-CNN against synthetic and (optionally) real captures.

Prints:
  - overall accuracy + clean-only / corrupt-only accuracy
  - per-class accuracy (sorted by worst first)
  - top-N confusion pairs (which classes get mistaken for which)
  - optional: same metrics computed over a folder of real captures

Also dumps misclassified samples to ``tmp/misclassified/`` (synth) and
``tmp/misclassified_real/`` (real) so you can eyeball what's failing.

Real-capture layout
-------------------
Place hand-labeled PNG/JPG crops in::

    eval_set/<class_id>_<piece_name>/<anything>.png

e.g. ``eval_set/37_3x2/cap01.png`` for a real 3x2 capture. The first
``_<class_id>`` of the parent folder name is the ground-truth label. Folder
naming after the underscore is free-form. If ``eval_set/`` doesn't exist
the script just runs synth-only.

Usage::

    uv run scripts/eval_piece_cnn.py
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from blockblaster.game.pieces import PIECES
from blockblaster.piece_cnn.config import CLEAN_SAMPLE_FRACTION, NUM_CLASSES
from blockblaster.piece_cnn.model import (
    DEFAULT_WEIGHT_PATH,
    PieceCNN,
    preprocess_batch,
)
from blockblaster.piece_cnn.synth import (
    EMPTY_CLASS_ID,
    class_id_for,
    piece_for_class,
    render_piece_sample,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_SYNTH_SAMPLES = 5_000
BATCH_SIZE        = 256
SEED              = 12345
EMPTY_FRACTION    = 1 / NUM_CLASSES
TOP_CONFUSION     = 20
SYNTH_MISS_DIR    = Path("tmp/misclassified")
REAL_DIR          = Path("eval_set")
REAL_MISS_DIR     = Path("tmp/misclassified_real")
WEIGHTS_PATH      = DEFAULT_WEIGHT_PATH


# ── Helpers ──────────────────────────────────────────────────────────────────

def _class_name(cid: int) -> str:
    if cid == EMPTY_CLASS_ID:
        return "empty"
    p = piece_for_class(cid)
    return p.name if p is not None else f"class{cid}"


def _load_net(device: torch.device) -> PieceCNN:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"weights not found at {WEIGHTS_PATH.resolve()} — train first via "
            f"`uv run train_piece_cnn.py`"
        )
    net = PieceCNN().to(device).eval()
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    return net


@torch.no_grad()
def _predict(
    net: PieceCNN,
    images_bgr: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_class_ids, top1_probs) for a (N, H, W, 3) uint8 BGR array."""
    preds_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    for i in range(0, len(images_bgr), BATCH_SIZE):
        chunk = images_bgr[i : i + BATCH_SIZE]
        x = preprocess_batch(chunk).to(device)
        logits = net(x)
        probs  = F.softmax(logits, dim=1)
        top_p, top_c = probs.max(dim=1)
        preds_all.append(top_c.cpu().numpy())
        probs_all.append(top_p.cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(probs_all)


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_overall(name: str, labels: np.ndarray, preds: np.ndarray) -> None:
    n = len(labels)
    acc = float((preds == labels).mean()) if n else 0.0
    print(f"\n── {name}  (n={n}) ───────────────────────────────")
    print(f"  overall accuracy: {acc:.4f}  ({int((preds == labels).sum())}/{n})")


def _print_per_class(labels: np.ndarray, preds: np.ndarray) -> None:
    print("\n  per-class accuracy (worst first):")
    print(f"    {'class':<16}{'acc':>8}{'n':>8}")
    rows: list[tuple[str, float, int]] = []
    for cid in range(NUM_CLASSES):
        mask = labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float((preds[mask] == cid).mean())
        rows.append((_class_name(cid), acc, n))
    rows.sort(key=lambda r: (r[1], -r[2]))
    for name, acc, n in rows[:15]:
        print(f"    {name:<16}{acc:>8.3f}{n:>8d}")


def _print_confusions(
    labels: np.ndarray,
    preds: np.ndarray,
    top: int = TOP_CONFUSION,
) -> None:
    confusions: Counter[tuple[int, int]] = Counter()
    for t, p in zip(labels, preds):
        if t != p:
            confusions[(int(t), int(p))] += 1
    if not confusions:
        print("\n  no confusions — perfect classification on this set.")
        return
    print(f"\n  top {top} confusion pairs  (true → predicted, count):")
    print(f"    {'true':<16}{'→  predicted':<20}{'count':>8}")
    for (t, p), c in confusions.most_common(top):
        print(f"    {_class_name(t):<16}{'→  ' + _class_name(p):<20}{c:>8d}")


def _dump_misclassified(
    out_dir: Path,
    images_bgr: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    max_per_pair: int = 8,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_counts: Counter[tuple[int, int]] = Counter()
    written = 0
    for i in range(len(labels)):
        t, p = int(labels[i]), int(preds[i])
        if t == p:
            continue
        if pair_counts[(t, p)] >= max_per_pair:
            continue
        pair_counts[(t, p)] += 1
        fname = (
            f"true{t:02d}_{_class_name(t)}__"
            f"pred{p:02d}_{_class_name(p)}__"
            f"p{probs[i]:.2f}__{i:05d}.png"
        )
        cv2.imwrite(str(out_dir / fname), images_bgr[i])
        written += 1
    return written


# ── Synthetic eval set ───────────────────────────────────────────────────────

def _render_synth_eval(
    n_samples: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (images_bgr, labels, is_clean_mask)."""
    images: list[np.ndarray] = []
    labels = np.empty(n_samples, dtype=np.int64)
    is_clean = np.empty(n_samples, dtype=bool)
    for i in range(n_samples):
        piece = None if rng.random() < EMPTY_FRACTION else rng.choice(PIECES)
        clean = rng.random() < CLEAN_SAMPLE_FRACTION
        img = render_piece_sample(piece, rng, clean=clean)
        images.append(img)
        labels[i] = class_id_for(piece)
        is_clean[i] = clean
    return np.stack(images), labels, is_clean


# ── Real-capture eval set ────────────────────────────────────────────────────

def _load_real_eval(root: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not root.exists():
        return None
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        prefix = class_dir.name.split("_", 1)[0]
        try:
            cid = int(prefix)
        except ValueError:
            print(f"  [warn] skipping {class_dir} (folder name must start with '<class_id>_')")
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [warn] failed to read {f}")
                continue
            images.append(img)
            labels.append(cid)
    if not images:
        return None
    # cv2.imread sizes vary; preprocess_batch handles resize. Keep originals
    # for the misclassified-dump step.
    return np.array(images, dtype=object), np.array(labels, dtype=np.int64)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  weights={WEIGHTS_PATH.resolve()}")
    net = _load_net(device)
    print(f"[eval] model params: {sum(p.numel() for p in net.parameters()):,}")

    # ── Synthetic eval ───────────────────────────────────────────────────
    rng = random.Random(SEED)
    print(f"\n[eval] rendering {NUM_SYNTH_SAMPLES:,} synthetic eval samples…")
    imgs, labels, is_clean = _render_synth_eval(NUM_SYNTH_SAMPLES, rng)

    preds, probs = _predict(net, imgs, device)
    _print_overall("SYNTH  (clean + corrupt)", labels, preds)

    clean_mask = is_clean
    _print_overall("SYNTH  clean-only",        labels[clean_mask],  preds[clean_mask])
    _print_overall("SYNTH  corrupt-only",      labels[~clean_mask], preds[~clean_mask])

    _print_per_class(labels, preds)
    _print_confusions(labels, preds)

    n_miss = _dump_misclassified(SYNTH_MISS_DIR, imgs, labels, preds, probs)
    print(f"\n  dumped {n_miss} misclassified synth samples → {SYNTH_MISS_DIR}/")

    # ── Real-capture eval ────────────────────────────────────────────────
    print(f"\n[eval] looking for real captures in {REAL_DIR.resolve()}/")
    real = _load_real_eval(REAL_DIR)
    if real is None:
        print("  no real captures found — create eval_set/<class_id>_<name>/*.png to enable")
        return

    real_imgs_obj, real_labels = real
    # preprocess_batch accepts a list of BGR uint8 arrays of any size
    real_imgs_list = list(real_imgs_obj)
    print(f"  loaded {len(real_imgs_list)} real captures")
    real_preds_all: list[int] = []
    real_probs_all: list[float] = []
    with torch.no_grad():
        for i in range(0, len(real_imgs_list), BATCH_SIZE):
            chunk = real_imgs_list[i : i + BATCH_SIZE]
            x = preprocess_batch(chunk).to(device)
            logits = net(x)
            probs  = F.softmax(logits, dim=1)
            top_p, top_c = probs.max(dim=1)
            real_preds_all.extend(top_c.cpu().tolist())
            real_probs_all.extend(top_p.cpu().tolist())
    real_preds = np.array(real_preds_all, dtype=np.int64)
    real_probs = np.array(real_probs_all, dtype=np.float32)

    _print_overall("REAL captures", real_labels, real_preds)
    _print_per_class(real_labels, real_preds)
    _print_confusions(real_labels, real_preds)

    # Dump misclassified reals — resize originals to a uniform size first
    real_imgs_uniform = np.stack([
        cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
        for im in real_imgs_list
    ])
    n_miss = _dump_misclassified(
        REAL_MISS_DIR, real_imgs_uniform, real_labels, real_preds, real_probs,
        max_per_pair=999,
    )
    print(f"\n  dumped {n_miss} misclassified real samples → {REAL_MISS_DIR}/")


if __name__ == "__main__":
    main()
