# Game Rules

[← back to README](../README.md)

The simulator in [`blockblaster/game/`](../blockblaster/game/) implements Block Blast.

## Board & queue

- **Board:** `BOARD_SIZE × BOARD_SIZE = 8 × 8` grid of cells, each empty or filled.
- **Queue:** `QUEUE_SIZE = 3` upcoming pieces. You may place them in any order; a new set of 3 is dealt once all three are placed.
- **Placement:** a piece drops onto a set of empty cells matching its shape (no rotation in this variant — every rotation is its own piece id). A placement is legal iff all its cells land on empty squares.
- **Line clear:** after a placement, any fully-filled row or column clears simultaneously.
- **Game over:** when none of the remaining queue pieces has a legal placement.

## Pieces

42 canonical pieces ([`pieces.py`](../blockblaster/game/pieces.py)), each a frozen set of `(row, col)` cell offsets with a stable `piece_id`:

- Single `1x1`
- Horizontal bars `1x2 … 1x5`, vertical bars `2x1 … 5x1`
- Squares `2x2`, `3x3`; rectangles `2x3`, `3x2`
- L-shapes (2-, 3-, and 5-cell variants, all rotations)
- S / Z (horizontal and vertical)
- T-shapes (4 rotations), Plus
- Diagonals (corner-touching, length 2 and 3)

`PIECE_BY_ID` maps id → `Piece`; ids are contiguous `0…41`, so they double as the piece-classifier's class labels.

## Scoring

From [`scoring.py`](../blockblaster/game/scoring.py) and [`param.py`](../param.py):

```
step_reward = cells_placed · REWARD_PER_CELL
            + (lines_cleared · REWARD_PER_LINE  +  MULTI_CLEAR_BONUS[lines])   if lines > 0
```

| Constant | Value |
|----------|-------|
| `REWARD_PER_CELL` | 1.0 |
| `REWARD_PER_LINE` | 25.0 |
| `MULTI_CLEAR_BONUS` | `{1:0, 2:50, 3:150, 4:350, 5:700}` |

`lines_cleared` counts rows **plus** columns cleared by a single placement, so simultaneous multi-line clears earn the large bonuses. These are the **real game rewards** used both for environment scoring and as the basis for the agent's training targets (the potential shaping in [algorithm.md](algorithm.md) is layered on top of these, not in place of them).
