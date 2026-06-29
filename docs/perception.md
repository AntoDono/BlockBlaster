# Perception

[← back to README](../README.md)

How a live mirrored frame becomes a board grid, a recognised queue, and a move suggestion. All of this runs on a background thread ([`assist/vision/analyzer.py`](../blockblaster/assist/vision/analyzer.py)); the UI only reads its latest snapshot.

```
frame ─▶ detect_interactables ─▶ split_roles ─▶ scan_board ─▶ 8×8 grid ┐
                                      └─▶ tray crops ─▶ piece CNN ─▶ queue ┤
                                                                          ├─▶ advisor ─▶ Suggestion
                                                                          ┘
```

## 1 · Interactable detection — [`detection.py`](../blockblaster/assist/vision/detection.py)

No hand-calibrated grid. The whole screen is segmented by **background subtraction**:

1. Sample a small strip at the very bottom of the frame (always empty game background) → median background BGR.
2. Mask pixels whose summed absolute BGR difference exceeds `BG_DIFF_THRESHOLD`.
3. Morphological **open** (`OPEN_KERNEL`) to kill speckle, then **close** with an **ellipse** kernel (`CLOSE_KERNEL = 17`) to fuse a piece's cells — including corner-touching S/Z/diagonal cells — into one solid blob.
4. Connected components above a min area / dimension → `Element`s, largest first.

`split_roles` classifies the largest blob as the **board** and the (up to 3) largest blobs below the board line as **pieces**. The board element is cached (`CACHED_BOARD`) and reused on subsequent frames unless re-detection is forced; `reset_board_cache()` clears it (wired to the **Recalibrate** button).

## 2 · Board scan — [`scanner.py`](../blockblaster/assist/vision/scanner.py)

Given the board bbox, the crop is resized and split into 8×8 cells. Each cell's centre patch is sampled in HSV; a cell is **filled** when it's bright (`V`) **and** either saturated (a coloured block) or near-white. Output: an `(8, 8)` bool occupancy grid.

## 3 · Piece classifier — [`piece_cnn/`](../blockblaster/piece_cnn/)

A small CNN classifies each tray crop into one of `NUM_CLASSES = 43` labels (42 pieces + empty).

**Architecture** ([`model.py`](../blockblaster/piece_cnn/model.py)) — deliberately shallow and **resolution-preserving** (`INPUT_SIZE = 64`):

```
Conv3×3(3→16) → ReLU → Conv3×3(16→16) → ReLU → Dropout → Linear(16·64·64 → 43)
```

No pooling anywhere. Counting cells (1x4 vs 1x5, L vs bar) depends on the thin gaps *between* cells; pooling/averaging destroyed that signal in the old deeper net, which systematically miscounted long bars. Keeping full resolution to a single linear head fixed it.

**`allow_empty` (default False):** callers only feed crops where a piece was already detected, so the EMPTY class is masked out of the argmax and the net always returns its best *piece* guess (confidence still returned for downstream gating). The empty class is kept in the model as a negative/anomaly signal — flip `allow_empty=True` to let it report "empty".

**Training** ([`synth.py`](../blockblaster/piece_cnn/synth.py), [`train_piece_cnn.py`](../train_piece_cnn.py)): mostly **synthetic** renders — chamfered/bevelled cell grids on randomised backgrounds with shadows, borders, low-contrast regimes, rotation/shear, JPEG/colour-cast/occlusion corruption, and a `clean` fraction matching the pristine in-game look — mixed with **real collected crops** (`data/pieces/<label>/`) oversampled and loss-weighted, with a held-out real-val split. Result: ~99% on held-out real crops, with the bar-length confusions essentially gone.

**Fallback:** if `piece_cnn.pt` is missing/unloadable, [`piece_recognizer.py`](../blockblaster/assist/vision/piece_recognizer.py) falls back to a geometric projection / template-matching heuristic ([`piece_mask.py`](../blockblaster/assist/vision/piece_mask.py)). (It's far less accurate — ~66% — and is only a safety net; see `eval_recognizers.py`.)

**Evaluation harnesses:** [`eval_recognizers.py`](../eval_recognizers.py) (geometric vs CNN over all of `data/pieces/`) and [`eval_heldout.py`](../eval_heldout.py) (honest accuracy on the real-val split only).

## 4 · Advisor — [`assist/advisor.py`](../blockblaster/assist/advisor.py)

Loads the trained `ValueNet` champion and exposes `suggest(board_grid, queue) → Suggestion(slot, row, col, piece)`. The advisor uses the same 3-piece lookahead as the training policy (shared via [`game/lookahead.py`](../blockblaster/game/lookahead.py)): for every distinct ordering of the tray it simulates each candidate placement with row/column clears and scores the resulting sequence as `r₀ + γ·r₁ + γ²·r₂ + γ³·V*(s₃)`, where `r_k` is the immediate placement-plus-clear reward and `V*(s) = V_F(s) + Φ(s)`.
Selection is **feasibility-first**:
1. **Primary** — a first move is *safe* if at least one continuation places all three tray pieces. If any safe first move exists, only safe moves are considered and the best one wins by total discounted score.
2. **Tie-break** — within the safe pool, the discounted step rewards and `V*` of the leaf jointly pick the winner (so clearing a row inside the 3-piece window is rewarded the same way it is during training).
3. **Fallback** — when every first move forces a game-over within the 3-piece window, the advisor still returns the best-scoring terminal path (best-of-a-bad-situation).
On sparse boards (every tray piece has many legal positions) feasibility is trivially satisfied and the call uses the same `BEAM_WIDTH` beam the training policy does; on cramped boards every distinct first move is retained at the top of the search so the only feasible move is never pruned. Results are cached on `(board_grid, queue)` so an unchanged scene doesn't re-run the net.

## 5 · Latching & gating — [`analyzer.py`](../blockblaster/assist/vision/analyzer.py)

The worker re-analyses only on a *new* captured frame, and:

- **Holds the suggestion** until the debounced **(board, queue)** state changes — a dragged piece keeps the board scan churning so the state never stabilises and the suggestion survives the whole move; a settled placement or a freshly-dealt set triggers a recompute. This prevents both flicker and stale suggestions that point at a piece no longer in the tray.
- **Pauses during big-motion events** (drop/clear animations, flagged by the frame-diff tracker) and re-analyses the instant the event clears.
- Supports a **manual board override** (`set_board_override`) used by the GUI's *Edit Board* mode.

See [assist-gui.md](assist-gui.md) for the controls and [visual-servo.md](visual-servo.md) for what happens after a suggestion is chosen.
