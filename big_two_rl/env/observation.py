"""
Observation vector encoder (308-dimensional float32) for Capsa Banting (Big Two).
Implements Canonical Seat Invariance, Relative Trick Leadership,
Bilateral Tactical Awareness, and Adjacent Opponent Profiling (Left, Right, Opposite).

Observation Space: Box(0.0, 1.0, shape=(308,), dtype=np.float32)
1. [0:52]: Self Hand Tensor (52-dim binary)
2. [52:104]: Discard Pile Tensor (52-dim binary)
3. [104:156]: Current Active Trick Tensor (52-dim binary)
4. [156:208]: Current Boss Indicators (52-dim binary)
5. [208:234]: Core Contextual Features (26-dim):
   - [208:212]: Trick length one-hot (1, 2, 3, 5 cards)
   - [212:220]: Trick type one-hot (Single, Pair, Triple, Straight, Flush, FullHouse, Quads, StraightFlush)
   - [220]: Highest rank on active trick (normalized r / 12.0)
   - [221:224]: Remaining cards of 3 opponents in relative clockwise order (Left, Opposite, Right / 13.0)
   - [224:227]: Critical indicator opponents with exactly 1 card remaining (Left, Opposite, Right)
   - [227:231]: Canonical Relative Pass status on current trick (Self=0, Left=1, Opposite=2, Right=3)
   - [231]: Free lead status (1.0 if free lead, 0.0 otherwise)
   - [232]: Optimal partition steps N_step(H) / 13.0
   - [233]: Count of 2s held / 4.0
6. [234:260]: Bilateral Adjacent Tactical Features (26-dim):
   - [234:238]: Relative Trick Leader One-Hot (Self=0, Left=1, Opposite=2, Right=3; 0 if free lead)
   - [238:241]: Opponents <= 2 cards critical threat (Left, Opposite, Right)
   - [241:244]: Opponents <= 3 cards alert indicator (Left, Opposite, Right)
   - [244:248]: Left Player (Downstream) Round Pass Memory (Single, Pair, Triple, 5-Card)
   - [248:252]: Right Player (Upstream) Round Pass Memory (Single, Pair, Triple, 5-Card)
   - [252:256]: Opposite Player Round Pass Memory (Single, Pair, Triple, 5-Card)
   - [256]: Downstream Anti-Feed Urgency Score max(0, 4 - k_left) / 3.0
   - [257]: Upstream Pressure Metric (trick rank / 12.0 if Right led, else 0.0)
   - [258]: Interception Opportunity Alert (1.0 if Upstream/Opposite led, Left <= 3 cards, Self can beat)
   - [259]: Downstream Combo Starvation Exploitability (1.0 if Free Lead & Self holds combo Left passed)
7. [260:308]: Adjacent Opponent Profiling & Behavioral Fingerprints (48-dim):
   - [260:276]: Left Player Profile Z_left (16-dim: identity, win rate, penalties, momentum, aggressiveness, combo prefs, hoarding timing, pass frequency, conversion)
   - [276:292]: Opposite Player Profile Z_opp (16-dim)
   - [292:308]: Right Player Profile Z_right (16-dim)
"""
from typing import Sequence, Set, Dict, List, Optional, Any
from collections import Counter
import numpy as np
from gymnasium import spaces

from big_two_rl.engine.card import get_rank, get_suit
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    get_optimal_partition_steps,
    evaluate_hand,
)
from big_two_rl.engine.game_state import GameState

OBS_DIM = 308
OBSERVATION_SPACE = spaces.Box(
    low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
)

FIVE_CARD_TYPES = {
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.QUADS,
    HandType.STRAIGHT_FLUSH,
}

# Completely neutral 16-dim opponent profile for uninitialized / pre-training states
DEFAULT_OPPONENT_PROFILE_16 = np.zeros(16, dtype=np.float32)


def encode_observation(
    game_state: GameState,
    player_idx: int,
    profile_manager: Optional[Any] = None
) -> np.ndarray:
    """
    Encode the current game state from the perspective of player_idx
    into a 308-dimensional float32 vector with bilateral adjacent player awareness
    and explicit opponent profiling (Left, Right, Opposite).
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
    boss_cards = game_state.get_boss_cards(player_idx)
    for c in boss_cards:
        if 0 <= c < 52:
            obs[156 + c] = 1.0

    # 5. [208:234]: Core Contextual Features (26-dim)
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
        ct_idx = int(combo.combo_type) - 1
        if 0 <= ct_idx < 8:
            obs[212 + ct_idx] = 1.0

        # Highest rank on active trick [220] (r / 12.0)
        if combo.primary_card >= 0:
            obs[220] = float(get_rank(combo.primary_card)) / 12.0

    # Opponents remaining cards in relative clockwise order [221:224] (rel 1=Left, 2=Opp, 3=Right)
    # Critical 1-card indicator [224:227]
    # Critical <=2 cards [238:241]
    # Alert <=3 cards [241:244]
    for rel_i in range(1, 4):
        opp_idx = (player_idx + rel_i) % 4
        opp_count = len(game_state.hands[opp_idx])
        obs[220 + rel_i] = float(opp_count) / 13.0
        if opp_count == 1:
            obs[223 + rel_i] = 1.0
        if opp_count <= 2:
            obs[237 + rel_i] = 1.0
        if opp_count <= 3:
            obs[240 + rel_i] = 1.0

    # Canonical Relative Pass history on CURRENT trick [227:231]
    # rel 0=Self, rel 1=Left, rel 2=Opposite, rel 3=Right
    for rel_i in range(4):
        p_idx = (player_idx + rel_i) % 4
        if game_state.passes[p_idx]:
            obs[227 + rel_i] = 1.0

    # Free lead status [231]
    obs[231] = 1.0 if is_free_lead else 0.0

    # Optimal partition steps N_step(H) / 13.0 [232]
    n_step = get_optimal_partition_steps(hand)
    obs[232] = float(n_step) / 13.0

    # Count of 2s held / 4.0 [233]
    twos_held = sum(1 for c in hand if get_rank(c) == 12)
    obs[233] = float(twos_held) / 4.0

    # 6. [234:260]: Bilateral Adjacent Tactical Features (26-dim)
    # [234:238]: Relative Trick Leader One-Hot
    if not is_free_lead and combo is not None and combo.combo_type != HandType.PASS:
        rel_leader = (game_state.trick_lead_player - player_idx) % 4
        obs[234 + rel_leader] = 1.0

    # Pass memory throughout the round for Left, Right, Opposite
    passed_combos = getattr(game_state, "passed_combos", [set(), set(), set(), set()])

    def encode_pass_memory(p: int, base_idx: int):
        s = passed_combos[p]
        if HandType.SINGLE in s:
            obs[base_idx] = 1.0
        if HandType.PAIR in s:
            obs[base_idx + 1] = 1.0
        if HandType.TRIPLE in s:
            obs[base_idx + 2] = 1.0
        if any(t in s for t in FIVE_CARD_TYPES):
            obs[base_idx + 3] = 1.0

    left_p = (player_idx + 1) % 4
    right_p = (player_idx + 3) % 4
    opp_p = (player_idx + 2) % 4

    # [244:248]: Left Player Round Pass Memory
    encode_pass_memory(left_p, 244)
    # [248:252]: Right Player Round Pass Memory
    encode_pass_memory(right_p, 248)
    # [252:256]: Opposite Player Round Pass Memory
    encode_pass_memory(opp_p, 252)

    # [256]: Downstream Anti-Feed Urgency Score: max(0, 4 - k_left) / 3.0
    left_cards = len(game_state.hands[left_p])
    obs[256] = float(max(0, 4 - left_cards)) / 3.0

    # [257]: Upstream Pressure Metric (trick rank / 12.0 if Right led, else 0.0)
    if not is_free_lead and combo is not None and combo.combo_type != HandType.PASS:
        if (game_state.trick_lead_player - player_idx) % 4 == 3 and combo.primary_card >= 0:
            obs[257] = float(get_rank(combo.primary_card)) / 12.0

    # [258]: Interception Opportunity Alert
    if not is_free_lead and combo is not None and combo.combo_type != HandType.PASS:
        rel_leader = (game_state.trick_lead_player - player_idx) % 4
        if rel_leader in (2, 3) and left_cards <= 3:
            legal_moves = game_state.get_legal_actions(player_idx)
            if any(len(m) > 0 for m in legal_moves):
                obs[258] = 1.0

    # [259]: Downstream Combo Starvation Exploitability
    if is_free_lead and len(hand) > 0:
        left_s = passed_combos[left_p]
        if left_s:
            rc = Counter(get_rank(c) for c in hand)
            has_pair = any(cnt >= 2 for cnt in rc.values())
            has_triple = any(cnt >= 3 for cnt in rc.values())
            has_5card = len(hand) >= 5

            if (HandType.PAIR in left_s and has_pair) or \
               (HandType.TRIPLE in left_s and has_triple) or \
               (any(t in left_s for t in FIVE_CARD_TYPES) and has_5card):
                obs[259] = 1.0

    # 7. [260:308]: Adjacent Opponent Profiling & Behavioral Fingerprints (48-dim)
    # [260:276]: Left Player Profile Z_left (16-dim)
    # [276:292]: Opposite Player Profile Z_opp (16-dim)
    # [292:308]: Right Player Profile Z_right (16-dim)
    if profile_manager is not None:
        profiles_48 = profile_manager.get_adjacent_profiles(player_idx)
        obs[260:308] = profiles_48
    else:
        # Default baseline neutral profiles if no profiler attached
        obs[260:276] = DEFAULT_OPPONENT_PROFILE_16
        obs[276:292] = DEFAULT_OPPONENT_PROFILE_16
        obs[292:308] = DEFAULT_OPPONENT_PROFILE_16

    return obs
