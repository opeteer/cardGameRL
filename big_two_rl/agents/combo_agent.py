"""
Bernstein - The Combo Architect (Hand Partitioner & Structural Purist).
Decomposes hands into optimal combinations (5-Card, Triples, Pairs, Singles),
preserves combination integrity by refusing to break pairs/triples for singles,
and prioritizes discarding large combination packages on free leads.
"""
from typing import Any, List, Tuple, Sequence, Set
from collections import defaultdict
import numpy as np

from big_two_rl.agents.base_agent import BaseAgent
from big_two_rl.engine.card import get_rank
from big_two_rl.engine.evaluator import evaluate_hand, HandType
from big_two_rl.engine.game_state import get_all_combinations_in_hand
from big_two_rl.env.actions import cards_to_action


def decompose_hand(hand: Sequence[int]) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]], List[int]]:
    """
    Greedily partition hand into disjoint:
    - combos_5: List of 5-card combinations (Straight, Flush, Full House, Quads, SF)
    - combos_3: List of Triples
    - combos_2: List of Pairs
    - singles: List of orphan single cards
    """
    used: Set[int] = set()
    combos_5: List[Tuple[int, ...]] = []

    all_combos = get_all_combinations_in_hand(hand)
    # Find 5-card combos: prioritize combos that do not waste Rank '2' cards
    c5_candidates = [c for c in all_combos if len(c.cards) == 5]
    c5_candidates.sort(
        key=lambda c: (
            any(get_rank(card) == 12 for card in c.cards), # False (0) before True (1)
            c.cmp_score
        )
    )

    for c in c5_candidates:
        if not any(card in used for card in c.cards):
            combos_5.append(c.cards)
            used.update(c.cards)

    # Remaining cards
    remaining = [c for c in hand if c not in used]
    by_rank = defaultdict(list)
    for c in remaining:
        by_rank[get_rank(c)].append(c)

    # Triples
    combos_3: List[Tuple[int, ...]] = []
    for r in sorted(by_rank.keys()):
        cards = by_rank[r]
        if len(cards) >= 3:
            trip = tuple(sorted(cards[:3]))
            combos_3.append(trip)
            used.update(trip)

    # Pairs
    remaining = [c for c in hand if c not in used]
    by_rank = defaultdict(list)
    for c in remaining:
        by_rank[get_rank(c)].append(c)

    combos_2: List[Tuple[int, ...]] = []
    for r in sorted(by_rank.keys()):
        cards = by_rank[r]
        if len(cards) >= 2:
            pair = tuple(sorted(cards[:2]))
            combos_2.append(pair)
            used.update(pair)

    # Singles
    singles = sorted([c for c in hand if c not in used])
    return combos_5, combos_3, combos_2, singles


class BernsteinAgent(BaseAgent):
    """
    Bernstein: The Combo Architect.
    - Preserves structural integrity of hands.
    - Does not break pairs or triples to respond to single challenges.
    - On free lead, discards high-volume packages first (5-card hands, then triples, then pairs).
    """
    def __init__(self, name: str = "Bernstein"):
        super().__init__(name=name)

    def select_action(
        self,
        env: Any,
        player_idx: int,
        legal_actions: np.ndarray,
        legal_moves: List[Tuple[int, ...]]
    ) -> int:
        if len(legal_actions) == 0:
            return 0

        hand = env.game_state.hands[player_idx]
        combos_5, combos_3, combos_2, singles = decompose_hand(hand)

        non_pass_moves = [m for m in legal_moves if len(m) > 0]
        can_pass = (0 in legal_actions)

        current_trick = env.game_state.current_trick_combo
        is_free_lead = (current_trick is None or current_trick.combo_type == HandType.PASS)

        # -------------------------------------------------------------
        # SCENARIO 1: FREE LEAD
        # -------------------------------------------------------------
        if is_free_lead:
            if not non_pass_moves:
                return 0

            # Discard volume priority: 5-card > 3-card > 2-card > singles
            if combos_5:
                # Pick lowest 5-card combo to shed cards smoothly
                c5_sorted = sorted(combos_5, key=lambda x: evaluate_hand(x).cmp_score)
                chosen = c5_sorted[0]
                if chosen in legal_moves:
                    return cards_to_action(chosen)

            if combos_3:
                c3_sorted = sorted(combos_3, key=lambda x: evaluate_hand(x).cmp_score)
                chosen = c3_sorted[0]
                if chosen in legal_moves:
                    return cards_to_action(chosen)

            if combos_2:
                c2_sorted = sorted(combos_2, key=lambda x: evaluate_hand(x).cmp_score)
                chosen = c2_sorted[0]
                if chosen in legal_moves:
                    return cards_to_action(chosen)

            if singles:
                chosen = (singles[0],)
                if chosen in legal_moves:
                    return cards_to_action(chosen)

            # Fallback
            return cards_to_action(non_pass_moves[0])

        # -------------------------------------------------------------
        # SCENARIO 2: CHALLENGING
        # -------------------------------------------------------------
        if not non_pass_moves:
            return 0

        if not can_pass:
            # First turn: Must play 3♦. Check if 3♦ belongs to a combo in partition
            moves_with_3d = [m for m in non_pass_moves if 0 in m]
            # Prefer 5-card with 3d, then pair with 3d, then single 3d
            moves_with_3d.sort(key=lambda m: (-len(m), evaluate_hand(m).cmp_score))
            return cards_to_action(moves_with_3d[0])

        trick_len = len(current_trick.cards)

        if trick_len == 1:
            # Table is SINGLE: Only play from singles pool! Do NOT break pairs/triples!
            legal_singles = [m for m in non_pass_moves if len(m) == 1 and m[0] in singles]
            if legal_singles:
                # Pick lowest legal single from pool
                legal_singles.sort(key=lambda m: evaluate_hand(m).cmp_score)
                return cards_to_action(legal_singles[0])
            else:
                # No orphan single can beat the table -> PASS to preserve structure!
                return 0

        elif trick_len == 2:
            # Table is PAIR: Only play pairs from combos_2!
            legal_pairs = [m for m in non_pass_moves if len(m) == 2 and m in combos_2]
            if legal_pairs:
                legal_pairs.sort(key=lambda m: evaluate_hand(m).cmp_score)
                return cards_to_action(legal_pairs[0])
            else:
                return 0

        elif trick_len == 3:
            # Table is TRIPLE: Only play triples from combos_3!
            legal_trips = [m for m in non_pass_moves if len(m) == 3 and m in combos_3]
            if legal_trips:
                legal_trips.sort(key=lambda m: evaluate_hand(m).cmp_score)
                return cards_to_action(legal_trips[0])
            else:
                return 0

        elif trick_len == 5:
            # Table is 5-CARD: Only play 5-card hands from combos_5!
            legal_5s = [m for m in non_pass_moves if len(m) == 5 and m in combos_5]
            if legal_5s:
                legal_5s.sort(key=lambda m: evaluate_hand(m).cmp_score)
                return cards_to_action(legal_5s[0])
            else:
                return 0

        # General fallback if nothing matched
        return 0
