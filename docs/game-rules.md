# Game Rules

[← back to README](../README.md)

## Rules

| Rule | Detail |
|------|--------|
| Grid | 8 × 8, starts empty |
| Queue | 3 pieces shown at once; a fresh batch of 3 is drawn when all are placed |
| Placement | A piece may be placed anywhere its cells fit on empty squares |
| Line clear | Any fully-filled row **and** column clears simultaneously after each placement |
| Multi-clear bonus | Extra reward when ≥ 2 lines clear from a single placement |
| Game over | When no queued piece can be legally placed anywhere |

## Pieces

42 canonical shapes:

- single cell
- horizontal/vertical bars (1×2 … 1×5, 2×1 … 5×1)
- 2×2 and 3×3 squares; 2×3 / 3×2 rectangles
- L/J variants with 2-cell, 4-cell (3-leg + foot), and 5-cell (3-leg + 3-leg) bodies
- S/Z, T, plus
- corner-touching diagonals of length 2 and 3

See [`blockblaster/game/pieces.py`](../blockblaster/game/pieces.py) for the full enumeration.

The reward / shaping interaction with these pieces is covered in
[algorithm.md → Reward shaping](algorithm.md#reward-shaping-potential-based).
