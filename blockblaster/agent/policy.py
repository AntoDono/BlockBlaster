"""3-piece beam-search lookahead policy with epsilon-exploration.

The value net is trained against *shaped* MC targets G_t - Phi(s_t), so it
predicts V_F(s) = V*(s) - Phi(s).  To recover the original optimal policy we
add Phi(s') back at action-selection time:

    V*(s) = V_F(s) + Phi(s)

The beam search scores each candidate sequence as the **true discounted
return** it would produce:

    score = r_0 + γ·r_1 + γ²·r_2 + γ³·V*(s_3)

where r_k is the immediate reward earned by placing the k-th piece (cells
placed + line-clear bonuses, per `blockblaster.game.scoring`).  Earlier
versions of this policy dropped the per-placement rewards and scored only
V*(s_3), which under-weighted line-clears inside the 3-piece window — a
disaster on cramped late-game boards where clearing *now* is literally
survival.  See `docs/policy.md` for the full rationale.

The shared search + selection lives in :mod:`blockblaster.game.lookahead`
and is also used by the live assist advisor; this module only handles the
training-loop wiring (env access, epsilon exploration, partial-queue
fallback).

Game-over handling
==================
A "dead-end" leaf — where the next piece in the planned ordering has no
legal position on the resulting board — is the search's view of *game
over*.  These leaves used to be scored as just ``accum_reward`` (V*=0),
which let a high-immediate-reward suicide path (e.g. clear two rows now,
then can't fit the next piece) beat a low-immediate-reward survival
path on the raw argmax.  We now flag dead-end leaves as terminal and
filter them at selection time: if **any** non-terminal continuation
exists for **any** action, we pick from those alone.  Only when every
action is forced into a terminal leaf (unavoidable death within the
3-piece window) do we fall back to ranking among terminal leaves.

See `blockblaster/game/potential.py`, `blockblaster/game/scoring.py`,
and `blockblaster/train/dataset.py`.
"""

from __future__ import annotations

import random
from typing import Optional

import param
from blockblaster.game.env import BlockBlastEnv
from blockblaster.game.lookahead import enumerate_leaves, select_first_move
from blockblaster.model.value_net import ValueNet


def select_action(
    env: BlockBlastEnv,
    net: Optional[ValueNet],
    epsilon: float = 0.0,
    device: str | None = None,
    temperature: float = 0.0,
    top_m: int | None = None,
) -> tuple[int, int, int]:
    """Choose a (slot, row, col) action via beam-search 3-piece lookahead.

    Strategy:
      - With probability `epsilon` or if `net` is None: uniform random from
        the 1-step legal action set.
      - Otherwise: beam search over all distinct orderings of the queue.
        At each of the 3 depths, candidates are scored as V*(board, remaining)
        and only the top BEAM_WIDTH are expanded at the next depth.
        Dead-end beams (no legal position for the next piece) score V=0.
      - Final selection: if `temperature == 0` (default), pick the argmax over
        all leaf candidates (deterministic; eval-safe).  If `temperature > 0`,
        dedupe leaf candidates by first_action and sample one of the top-`top_m`
        with probability proportional to `exp(score / temperature)` — used
        during data-collection rounds to inject state diversity into the buffer.
    """
    actions = env.legal_actions()
    if not actions:
        raise RuntimeError("select_action called on a terminal state (no legal actions)")

    if net is None or random.random() < epsilon:
        return random.choice(actions)

    if top_m is None:
        top_m = param.SIM_EXPLORE_TOP_M

    tray = list(enumerate(env.queue))
    leaves = enumerate_leaves(
        env.board.grid, tray, net,
        beam_width=param.BEAM_WIDTH,
        chunk_size=param.LOOKAHEAD_MAX_BATCH,
    )
    pick = select_first_move(leaves, temperature=temperature, top_m=top_m)
    if pick is None:
        # No leaves enumerated (e.g. first piece has zero legal positions)
        # — fall back to a uniformly-random legal action.
        return random.choice(actions)
    path, _score, _terminal = pick
    first = path[0]
    return (first.slot, first.row, first.col)
