"""
Reward & Penalty Engine for Capsa Banting (Big Two) Reinforcement Learning.
Implements:
1. Terminal Zero-Sum Payoffs with Quadratic Multiplier & 2-Hoarding Penalty.
2. Potential-Based Reward Shaping (PBRS) based on Ng et al. (1999).
3. Grandmaster Tactical Penalties & Incentives (Anti-Feed, Combo Break, Boss Convert, Strategic Pass, Hoard 2).
"""
from typing import Sequence, Dict, Any, Optional, Tuple, Set, List
import numpy as np

from big_two_rl.engine.card import get_rank
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    evaluate_hand,
    compute_hand_partition_metrics,
)
from big_two_rl.engine.game_state import GameState, calculate_penalty


class RewardEngine:
    """
    Computes R_i(t) = R_terminal,i(t) + F_i(s_t, a_t, s_{t+1}) + sum_m Pi_{m,i}(s_t, a_t).
    """
    def __init__(
        self,
        gamma: float = 0.995,
        w1: float = 0.40,   # N_step weight
        w2: float = 0.30,   # B_count (boss cards) weight
        w3: float = 0.15,   # U_singles (unpaired low singles) weight
    ):
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def compute_potential(
        self,
        hand: Sequence[int],
        boss_cards: Set[int]
    ) -> float:
        """
        Compute Phi(s) = -w1 * N_step(H) + w2 * B_count(H, D) - w3 * U_singles(H).
        If hand is empty (terminal victory), Phi = 0.0.
        """
        if not hand:
            return 0.0

        hand_tuple = tuple(sorted(hand))
        n_step, u_singles = compute_hand_partition_metrics(hand_tuple)
        b_count = len(boss_cards)

        phi = -self.w1 * float(n_step) + self.w2 * float(b_count) - self.w3 * float(u_singles)
        return float(phi)

    def compute_step_reward(
        self,
        player_idx: int,
        cards_played: Sequence[int],
        hand_before: Sequence[int],
        hand_after: Sequence[int],
        boss_cards_before: Set[int],
        boss_cards_after: Set[int],
        game_state: GameState,
        can_beat_trick: bool,
        lead_regained_by_boss: bool = False,
        trick_leader_before: Optional[int] = None,
        is_free_lead_before: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total step reward for player_idx:
        R_step = F_pbrs + sum_m Pi_m.
        """
        details: Dict[str, float] = {}

        # 1. Potential-Based Reward Shaping (PBRS)
        phi_before = self.compute_potential(hand_before, boss_cards_before)
        phi_after = self.compute_potential(hand_after, boss_cards_after)
        f_pbrs = self.gamma * phi_after - phi_before
        details["f_pbrs"] = float(f_pbrs)

        pi_total = 0.0

        is_pass = (len(cards_played) == 0)
        next_player = (player_idx + 1) % 4
        next_player_cards = len(game_state.hands[next_player])

        # 2. Tactical Penalty: PENALTY_FEEDING (-30.0)
        # Playing a single low card (rank < Jack, i.e. rank < 8) when next opponent has exactly 1 card
        if len(cards_played) == 1 and next_player_cards == 1:
            card_rank = get_rank(cards_played[0])
            if card_rank < 8:  # Jack is rank index 8
                details["penalty_feeding"] = -30.0
                pi_total += -30.0

        # 2b. Tactical Penalty: PENALTY_FEEDING_TIER2 (-12.0)
        # Playing a low single when downstream has 2 cards and agent holds viable combo options
        if len(cards_played) == 1 and next_player_cards == 2:
            card_rank = get_rank(cards_played[0])
            if card_rank < 8:
                from collections import Counter
                rc = Counter(get_rank(c) for c in hand_before)
                if any(cnt >= 2 for cnt in rc.values()):
                    details["penalty_feeding_tier2"] = -12.0
                    pi_total += -12.0

        # 2c. Tactical Bonus: BONUS_COMBO_STARVE (+2.5)
        # On free lead, playing a combination type that downstream has previously passed
        if is_free_lead_before and len(cards_played) > 1:
            played_combo = evaluate_hand(cards_played)
            passed_set = getattr(game_state, "passed_combos", [set()]*4)[next_player]
            if played_combo is not None and played_combo.combo_type in passed_set:
                details["bonus_combo_starve"] = +2.5
                pi_total += 2.5

        # 2d. Tactical Bonus: BONUS_INTERCEPT (+2.0)
        # Intercepting trick led by Upstream/Opposite when downstream is in danger (<= 3 cards)
        if not is_free_lead_before and len(cards_played) > 0:
            effective_leader = trick_leader_before if trick_leader_before is not None else game_state.trick_lead_player
            if (effective_leader - player_idx) % 4 in (2, 3) and next_player_cards <= 3:
                details["bonus_intercept"] = +2.0
                pi_total += 2.0

        # 2e. Tactical Penalty: PENALTY_CARELESS_PASS (-10.0)
        # Passing on Upstream lead despite holding power cards when downstream has <= 2 cards
        if is_pass and can_beat_trick and next_player_cards <= 2:
            effective_leader = trick_leader_before if trick_leader_before is not None else game_state.trick_lead_player
            if (effective_leader - player_idx) % 4 in (2, 3):
                has_power = any(c in boss_cards_before or get_rank(c) == 12 for c in hand_before)
                if has_power:
                    details["penalty_careless_pass"] = -10.0
                    pi_total += -10.0

        # 3. Tactical Penalty: PENALTY_COMBO_BREAK (-5.0)
        # Breaking a high 5-card combo (Full House or Quads) into singles without taking boss status
        if len(cards_played) == 1 and len(hand_before) >= 5:
            # Check if hand_before had a Full House or Quads containing the played card
            had_quads_or_fh = self._has_quads_or_fullhouse(hand_before)
            has_now = self._has_quads_or_fullhouse(hand_after)
            if had_quads_or_fh and not has_now:
                # Broke the combination
                if cards_played[0] not in boss_cards_before:
                    details["penalty_combo_break"] = -5.0
                    pi_total += -5.0

        # 4. Tactical Bonus: BONUS_BOSS_CONVERT (+3.0)
        # Playing a boss card and regaining lead / forcing 3 passes
        if lead_regained_by_boss:
            details["bonus_boss_convert"] = +3.0
            pi_total += 3.0

        # 5. Tactical Bonus: BONUS_STRATEGIC_PASS (+1.0)
        # Choosing Pass when holding cards capable of beating the trick, preserving better future partitions
        if is_pass and can_beat_trick:
            n_step_before = compute_hand_partition_metrics(tuple(sorted(hand_before)))[0]
            n_step_after = compute_hand_partition_metrics(tuple(sorted(hand_after)))[0]
            if n_step_after <= n_step_before:
                details["bonus_strategic_pass"] = +1.0
                pi_total += 1.0

        # 6. Tactical Penalty: PENALTY_HOARD_2 (-2.0)
        # Still holding rank 2 when any opponent has <= 3 cards remaining
        opp_counts = [len(game_state.hands[p]) for p in range(4) if p != player_idx]
        min_opp_cards = min(opp_counts) if opp_counts else 13
        still_has_2 = any(get_rank(c) == 12 for c in hand_after)

        if still_has_2 and min_opp_cards <= 3:
            details["penalty_hoard_2"] = -2.0
            pi_total += -2.0

        details["pi_tactical"] = float(pi_total)
        total_reward = float(f_pbrs + pi_total)
        return total_reward, details

    def compute_terminal_payoffs(self, game_state: GameState) -> List[float]:
        """
        Compute zero-sum terminal payoffs for all 4 players:
        R_terminal,i(T) = sum_{j != i} C(k_j) if k_i == 0 else -C(k_i).
        """
        if not game_state.game_over or game_state.winner is None:
            return [0.0, 0.0, 0.0, 0.0]

        penalties = [
            calculate_penalty(
                len(game_state.hands[p]),
                has_2=any(get_rank(c) == 12 for c in game_state.hands[p])
            )
            for p in range(4)
        ]
        w = game_state.winner
        payoffs = [-float(penalties[p]) for p in range(4)]
        payoffs[w] = float(sum(penalties[p] for p in range(4) if p != w))
        return payoffs

    def _has_quads_or_fullhouse(self, hand: Sequence[int]) -> bool:
        if len(hand) < 5:
            return False
        from collections import Counter
        rc = Counter(get_rank(c) for c in hand)
        counts = set(rc.values())
        return (4 in counts) or (3 in counts and 2 in counts)
