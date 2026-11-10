"""
Observation vector encoder (234-dimensional float32) for Capsa Banting (Big Two).

Observation Space: Box(0.0, 1.0, shape=(234,), dtype=np.float32)
1. [0:52]: Self Hand Tensor (52-dim binary)
2. [52:104]: Discard Pile Tensor (52-dim binary)
3. [104:156]: Current Active Trick Tensor (52-dim binary)
4. [156:208]: Current Boss Indicators (52-dim binary)
5. [208:234]: Contextual Game Features (26-dim):
   - [208:212]: Trick length one-hot (1, 2, 3, 5 cards)
   - [212:220]: Trick type one-hot (Single, Pair, Triple, Straight, Flush, FullHouse, Quads, StraightFlush)
   - [220]: Highest rank on active trick (normalized r / 12.0)
   - [221:224]: Remaining cards of 3 opponents in turn order (normalized / 13.0)
   - [224:227]: Critical indicator opponents with exactly 1 card remaining (binary)
   - [227:231]: Pass status of all 4 players on current trick (binary)
   - [231]: Free lead status (1.0 if free lead, 0.0 otherwise)
   - [232]: Optimal partition steps N_step(H) / 13.0
   - [233]: Count of 2s held / 4.0
"""
from typing import Sequence
import numpy as np
from gymnasium import spaces

from big_two_rl.engine.card import get_rank, get_suit
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    get_optimal_partition_steps,
)
from big_two_rl.engine.game_state import GameState

OBS_DIM = 234
OBSERVATION_SPACE = spaces.Box(
    low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
)


def encode_observation(game_state: GameState, player_idx: int) -> np.ndarray:
    """
    Encode the current game state from the perspective of player_idx
    into a 234-dimensional float32 vector.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # 1. [0:52]: Self Hand Tensor (one-hot)
    hand = game_state.hands[player_idx]
    for c in hand:
        if 0 <= c < 52:
            obs[c] = 1.0

    # 2. [52:104]: Discard Pile Tensor (one-hot)
    for c in game_state.discard_pile:
        if 0 <= c < 52:
            obs[52 + c] = 1.0

    # 3. [104:156]: Current Active Trick Tensor (one-hot)
    for c in game_state.current_trick_cards:
        if 0 <= c < 52:
            obs[104 + c] = 1.0

    # 4. [156:208]: Current Boss Indicators (52-dim binary)
    # 1.0 if card in player's hand is strictly higher than all unseen cards
    boss_cards = game_state.get_boss_cards(player_idx)
    for c in boss_cards:
        if 0 <= c < 52:
            obs[156 + c] = 1.0

    # 5. [208:234]: Contextual Game Features (26-dim)
    combo = game_state.current_trick_combo
    is_free_lead = game_state.is_free_lead()

    if combo is not None and combo.combo_type != HandType.PASS and not is_free_lead:
        # Trick length one-hot [208:212]
        trick_len = len(combo.cards)
        if trick_len == 1:
            obs[208] = 1.0
        elif trick_len == 2:
            obs[209] = 1.0
        elif trick_len == 3:
            obs[210] = 1.0
        elif trick_len == 5:
            obs[211] = 1.0

        # Trick type one-hot [212:220]
        # HandType: SINGLE=1, PAIR=2, TRIPLE=3, STRAIGHT=4, FLUSH=5, FULL_HOUSE=6, QUADS=7, STRAIGHT_FLUSH=8
        ct_idx = int(combo.combo_type) - 1
        if 0 <= ct_idx < 8:
            obs[212 + ct_idx] = 1.0

        # Highest rank on active trick [220] (r / 12.0)
        if combo.primary_card >= 0:
            obs[220] = float(get_rank(combo.primary_card)) / 12.0

    # Opponents remaining cards in clockwise order [221:224] & Critical 1-card indicator [224:227]
    for i in range(1, 4):
        opp_idx = (player_idx + i) % 4
        opp_count = len(game_state.hands[opp_idx])
        obs[220 + i] = float(opp_count) / 13.0
        if opp_count == 1:
            obs[223 + i] = 1.0

    # Pass history on current trick [227:231]
    for p in range(4):
        if game_state.passes[p]:
            obs[227 + p] = 1.0

    # Free lead status [231]
    obs[231] = 1.0 if is_free_lead else 0.0

    # Optimal partition steps N_step(H) / 13.0 [232]
    n_step = get_optimal_partition_steps(hand)
    obs[232] = float(n_step) / 13.0

    # Count of 2s held / 4.0 [233]
    twos_held = sum(1 for c in hand if get_rank(c) == 12)
    obs[233] = float(twos_held) / 4.0

    return obs
