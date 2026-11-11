"""
Dynamic Hand Strength & Opponent Equity Tracker for Big Two (Capsa Banting).
Provides real-time mathematical deduction of:
1. Self Hand Strength (Initial luck, current remaining strength, partition steps, penalty hazard).
2. Global Power Card Depletion (Remaining unrevealed 2s and Aces).
3. Opponent Demonstrated Power & Relative Equity (Bilateral Left, Opposite, Right).
"""
from typing import Sequence, Dict, Any, Optional, List
from collections import Counter
import numpy as np

from big_two_rl.engine.card import get_rank
from big_two_rl.engine.luck_meter import calculate_initial_hand_luck
from big_two_rl.engine.evaluator import HandType, HandCombo, compute_hand_partition_metrics

EQUITY_FEATURE_DIM = 18
DEFAULT_EQUITY_VECTOR_18 = np.zeros(EQUITY_FEATURE_DIM, dtype=np.float32)
# Neutral luck differential is 0.5 (indices 8, 12, 16 in the 18-dim vector)
DEFAULT_EQUITY_VECTOR_18[8] = 0.5
DEFAULT_EQUITY_VECTOR_18[12] = 0.5
DEFAULT_EQUITY_VECTOR_18[16] = 0.5


class DynamicEquityTracker:
    """
    Tracks and computes real-time Hand Strength & Opponent Equity metrics
    for all 4 seats in a Big Two game.
    """
    def __init__(self):
        self.initial_luck: List[float] = [130.0, 130.0, 130.0, 130.0]
        self.demonstrated_luck: List[float] = [0.0, 0.0, 0.0, 0.0]
        self.twos_played: List[int] = [0, 0, 0, 0]
        self.aces_played: List[int] = [0, 0, 0, 0]
        self.cards_played: List[List[int]] = [[], [], [], []]

    def reset(self, initial_hands: Sequence[Sequence[int]]):
        """Reset the tracker at the beginning of a new game round."""
        self.initial_luck = [
            float(calculate_initial_hand_luck(hand)) if hand else 130.0
            for hand in initial_hands
        ]
        self.demonstrated_luck = [0.0, 0.0, 0.0, 0.0]
        self.twos_played = [0, 0, 0, 0]
        self.aces_played = [0, 0, 0, 0]
        self.cards_played = [[], [], [], []]

    def record_play(
        self,
        player_idx: int,
        move: Sequence[int],
        combo: Optional[HandCombo] = None
    ):
        """
        Record a non-pass play made by player_idx.
        Computes the incremental demonstrated power/luck from revealed cards.
        """
        if not move:
            return

        p = player_idx
        move_cards = list(move)
        self.cards_played[p].extend(move_cards)

        ranks = [get_rank(c) for c in move_cards]
        rank_counts = Counter(ranks)

        # 1. Base rank contribution
        move_power = float(sum(ranks))

        # 2. Power cards: 2s (rank 12)
        n_twos = rank_counts.get(12, 0)
        self.twos_played[p] += n_twos
        move_power += n_twos * 25.0
        if 51 in move_cards:  # 2 of Spades Supreme Card
            move_power += 10.0

        # 3. Power cards: Aces (rank 11)
        n_aces = rank_counts.get(11, 0)
        self.aces_played[p] += n_aces
        move_power += n_aces * 8.0

        # 4. Combo tier bonuses
        if combo is not None:
            c_type = combo.combo_type
            if c_type == HandType.STRAIGHT_FLUSH:
                move_power += 50.0
            elif c_type == HandType.QUADS:
                move_power += 35.0
            elif c_type == HandType.FULL_HOUSE:
                move_power += 20.0
            elif c_type == HandType.FLUSH:
                move_power += 15.0
            elif c_type == HandType.STRAIGHT:
                move_power += 10.0
            elif c_type == HandType.TRIPLE:
                move_power += 8.0
            elif c_type == HandType.PAIR:
                move_power += 3.0

        self.demonstrated_luck[p] += move_power

    def get_equity_vector(
        self,
        focal_seat: int,
        current_hands: Sequence[Sequence[int]],
        discard_pile: Sequence[int]
    ) -> np.ndarray:
        """
        Produce a normalized 18-dimensional feature vector from the perspective
        of focal_seat:
        - [0:4]: Self Hand Strength (Initial luck, current strength, steps, penalty danger)
        - [4:6]: Global Power Depletion (Unrevealed 2s, unrevealed Aces)
        - [6:10]: Left Opponent Equity (Demonstrated luck, 2s played, relative luck diff, threat index)
        - [10:14]: Opposite Opponent Equity
        - [14:18]: Right Opponent Equity
        All features strictly bounded in [0.0, 1.0].
        """
        vec = np.zeros(EQUITY_FEATURE_DIM, dtype=np.float32)

        # 1. Self Hand Strength [0:4]
        self_hand = current_hands[focal_seat] if focal_seat < len(current_hands) else []
        self_init_luck = self.initial_luck[focal_seat] if focal_seat < len(self.initial_luck) else 130.0
        vec[0] = float(np.clip(self_init_luck / 250.0, 0.0, 1.0))

        # Current remaining hand strength
        if self_hand:
            self_ranks = [get_rank(c) for c in self_hand]
            self_twos = sum(1 for r in self_ranks if r == 12)
            cur_strength = sum(self_ranks) + self_twos * 20.0
            vec[1] = float(np.clip(cur_strength / 150.0, 0.0, 1.0))
            steps, _ = compute_hand_partition_metrics(tuple(sorted(self_hand)))
            vec[2] = float(np.clip(steps / 13.0, 0.0, 1.0))
        else:
            vec[1] = 0.0
            vec[2] = 0.0

        # Penalty hazard indicator S_4
        n_self_cards = len(self_hand)
        if n_self_cards >= 10:
            vec[3] = 1.0  # Critical Penalty Evacuation Zone!
        elif n_self_cards >= 8:
            vec[3] = 0.5  # Approaching Penalty Zone
        else:
            vec[3] = 0.0

        # 2. Global Power Depletion [4:6]
        # Twos and Aces unrevealed globally across entire deck
        known_twos = sum(self.twos_played) + sum(1 for c in self_hand if get_rank(c) == 12)
        known_aces = sum(self.aces_played) + sum(1 for c in self_hand if get_rank(c) == 11)
        unrevealed_twos = max(0, 4 - known_twos)
        unrevealed_aces = max(0, 4 - known_aces)
        vec[4] = float(unrevealed_twos / 4.0)
        vec[5] = float(unrevealed_aces / 4.0)

        # 3. Bilateral Opponents [6:18]
        rel_offsets = [
            (1, 6),   # Left opponent: offset 1, slice [6:10]
            (2, 10),  # Opposite opponent: offset 2, slice [10:14]
            (3, 14),  # Right opponent: offset 3, slice [14:18]
        ]

        for offset, start_idx in rel_offsets:
            opp_seat = (focal_seat + offset) % 4
            opp_demonstrated = self.demonstrated_luck[opp_seat] if opp_seat < len(self.demonstrated_luck) else 0.0
            opp_twos = self.twos_played[opp_seat] if opp_seat < len(self.twos_played) else 0
            opp_hand_len = len(current_hands[opp_seat]) if opp_seat < len(current_hands) else 13

            # E_1: Demonstrated luck normalized
            vec[start_idx] = float(np.clip(opp_demonstrated / 200.0, 0.0, 1.0))
            # E_2: Twos played
            vec[start_idx + 1] = float(np.clip(opp_twos / 4.0, 0.0, 1.0))
            # E_3: Relative luck differential Delta L (0.5 is neutral parity)
            diff = (self_init_luck - opp_demonstrated + 200.0) / 400.0
            vec[start_idx + 2] = float(np.clip(diff, 0.0, 1.0))
            # E_4: Threat Index
            cards_shed = max(0, 13 - opp_hand_len)
            threat = (cards_shed / 13.0) * (1.0 + 0.5 * (opp_twos / 4.0))
            vec[start_idx + 3] = float(np.clip(threat, 0.0, 1.0))

        return vec
