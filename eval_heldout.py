"""Honest held-out eval: replicate the training real-val split and score CNN."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from blockblaster.piece_cnn import load_real_dataset
from blockblaster.piece_cnn.realdata import DEFAULT_DATA_DIR
from blockblaster.piece_cnn.synth import piece_for_class
from blockblaster.piece_cnn.model import PieceClassifier, preprocess_batch
import torch
import torch.nn.functional as F

# Must match train_piece_cnn.py
REAL_VAL_FRAC = 0.15
REAL_SPLIT_SEED = 123


def main() -> None:
    imgs, lbls = load_real_dataset(DEFAULT_DATA_DIR)
    n = len(imgs)
    rng = np.random.default_rng(REAL_SPLIT_SEED)
    perm = rng.permutation(n)
    imgs, lbls = imgs[perm], lbls[perm]
    n_val = max(1, int(round(n * REAL_VAL_FRAC)))
    val_imgs, val_lbls = imgs[:n_val], lbls[:n_val]
    print(f"total real crops: {n} | held-out val: {n_val}")

    clf = PieceClassifier()
    net = clf.net
    assert net is not None

    with torch.no_grad():
        batch = preprocess_batch(list(val_imgs)).to(clf.device)
        logits = net(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    correct = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    confusions: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(n_val):
        pred_cid = int(probs[i].argmax())
        conf = float(probs[i][pred_cid])
        true_p = piece_for_class(int(val_lbls[i]))
        pred_p = piece_for_class(pred_cid)
        true_name = true_p.name if true_p else "EMPTY"
        pred_name = pred_p.name if pred_p else "EMPTY"
        if pred_cid == int(val_lbls[i]):
            correct += 1
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)
            confusions[(true_name, pred_name)] += 1

    print(f"\nheld-out CNN accuracy: {correct}/{n_val} = {correct / n_val:.1%}")
    if conf_correct:
        print(f"  mean conf (correct): {np.mean(conf_correct):.3f}")
    if conf_wrong:
        print(f"  mean conf (wrong)  : {np.mean(conf_wrong):.3f}  "
              f"max wrong conf: {max(conf_wrong):.3f}")
    print("\nconfusions (true -> pred):")
    for (t, p), k in sorted(confusions.items(), key=lambda x: -x[1]):
        print(f"  {t:>8} -> {p:<8} x{k}")


if __name__ == "__main__":
    main()
