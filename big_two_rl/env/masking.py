"""
Action Mask Generator for Capsa Banting (1696-dimensional Discrete Action Space).
Enforces game legality and the Strategic Anti-Feeding Hard Constraint.
"""
from typing import Sequence, Optional
import numpy as np

from big_two_rl.env.actions import (
    NUM_ACTIONS,
    cards_to_action,
    ACTION_TO_CARDS,
)


def generate_action_mask(
    legal_moves: Sequence[Sequence[int]],
    hand: Sequence[int],
    next_player_card_count: Optional[int] = None,
    as_int8: bool = False,
    enforce_anti_feed: bool = True
) -> np.ndarray:
    """
    Generate an action mask of shape (1696,) where mask[a] = True (or 1)
    ONLY if action a is legal.

    Strategic Anti-Feed Hard Constraint:
    If the next player has exactly 1 card remaining (next_player_card_count == 1),
    and a Single move is chosen, all Single moves are LOCKED EXCEPT the single
    move with the absolute highest card value in the agent's hand.
    Pairs, Triples, 5-Card combos, and legal Passes remain available.
    """
    dtype = np.int8 if as_int8 else bool
    mask = np.zeros(NUM_ACTIONS, dtype=dtype)

    for move in legal_moves:
        act = cards_to_action(move)
        mask[act] = 1 if as_int8 else True

    # Apply Strategic Anti-Feeding Hard Constraint
    if enforce_anti_feed and next_player_card_count == 1 and len(hand) > 0:
        highest_card = max(hand)
        highest_single_act = cards_to_action((highest_card,))

        # Lock all single actions (actions 1 to 52) except the highest single
        for act in range(1, 53):
            if act != highest_single_act:
                mask[act] = 0 if as_int8 else False

    # Safety check: ensure at least one legal action remains
    if not np.any(mask):
        # Fallback to unrestricted legal moves
        for move in legal_moves:
            act = cards_to_action(move)
            mask[act] = 1 if as_int8 else True

    return mask
