"""
Sullivan - The Conservative Hoarder (Defensive Minimalist & Late-Game Executioner).
Hoards boss cards (Rank 2 and Ace), climbs with minimal ranks,
voluntarily passes on high challenges, and unleashes full power only in the endgame.
"""
from typing import Any, List, Tuple
import numpy as np

from big_two_rl.agents.base_agent import BaseAgent
from big_two_rl.engine.card import get_rank
from big_two_rl.engine.evaluator import evaluate_hand, HandType
from big_two_rl.env.actions import cards_to_action


class SullivanAgent(BaseAgent):
    """
    Sullivan: Defensive and patient hoarder agent.
    - Conserves high cards (Rank '2' and 'A') during early and mid game.
    - Minimal Beating: When countering, uses the lowest legal card/combo.
    - Passing Discipline: Voluntarily passes if table challenge rank >= Jack (rank 8),
      allowing aggressive opponents to exhaust each other.
    - Late-Game Executioner: Enters Crisis Mode when any opponent (or Sullivan)
      has <= 3 cards, aggressively playing top cards to close the game or block a win.
    """
    def __init__(self, name: str = "Sullivan", high_pass_threshold: int = 8):
        super().__init__(name=name)
        # Rank 8 = 'J', 9 = 'Q', 10 = 'K', 11 = 'A', 12 = '2'
        self.high_pass_threshold = high_pass_threshold

    def select_action(
        self,
        env: Any,
        player_idx: int,
        legal_actions: np.ndarray,
        legal_moves: List[Tuple[int, ...]]
    ) -> int:
        if len(legal_actions) == 0:
            return 0

        non_pass_moves = [m for m in legal_moves if len(m) > 0]
        can_pass = (0 in legal_actions)

        # Assess game urgency
        opp_card_counts = [len(env.game_state.hands[i]) for i in range(4) if i != player_idx]
        min_opp_cards = min(opp_card_counts) if opp_card_counts else 13
        my_card_count = len(env.game_state.hands[player_idx])
        is_crisis = (min_opp_cards <= 3) or (my_card_count <= 3)

        current_trick = env.game_state.current_trick_combo
        is_free_lead = (current_trick is None or current_trick.combo_type == HandType.PASS)

        # -------------------------------------------------------------
        # SCENARIO 1: FREE LEAD
        # -------------------------------------------------------------
        if is_free_lead:
            if not non_pass_moves:
                return 0

            # If Sullivan can win in this turn, play the winning combo!
            winning_moves = [m for m in non_pass_moves if len(m) == my_card_count]
            if winning_moves:
                # Pick the winning move
                return cards_to_action(winning_moves[0])

            if is_crisis:
                # In crisis, play highest combo or highest boss card to maintain absolute control
                sorted_moves = sorted(
                    non_pass_moves,
                    key=lambda x: evaluate_hand(x).cmp_score,
                    reverse=True
                )
                return cards_to_action(sorted_moves[0])
            else:
                # Normal mode: Shed lowest junk cards (lowest single or lowest pair)
                # Group by length
                singles = [m for m in non_pass_moves if len(m) == 1]
                pairs = [m for m in non_pass_moves if len(m) == 2]

                if singles:
                    sorted_singles = sorted(singles, key=lambda x: evaluate_hand(x).cmp_score)
                    return cards_to_action(sorted_singles[0])
                elif pairs:
                    sorted_pairs = sorted(pairs, key=lambda x: evaluate_hand(x).cmp_score)
                    return cards_to_action(sorted_pairs[0])
                else:
                    # Lowest combo overall
                    sorted_all = sorted(non_pass_moves, key=lambda x: evaluate_hand(x).cmp_score)
                    return cards_to_action(sorted_all[0])

        # -------------------------------------------------------------
        # SCENARIO 2: CHALLENGING (FOLLOWING A TRICK)
        # -------------------------------------------------------------
        if not non_pass_moves:
            return 0  # Forced pass

        if not can_pass:
            # First turn: Must play 3♦
            sorted_moves = sorted(non_pass_moves, key=lambda x: evaluate_hand(x).cmp_score)
            return cards_to_action(sorted_moves[0])

        # Sullivan can choose to Pass or Counter
        if is_crisis:
            # Crisis Mode: An opponent is about to win! Sullivan MUST intervene.
            # Play the smallest beating move that can stop them.
            sorted_moves = sorted(non_pass_moves, key=lambda x: evaluate_hand(x).cmp_score)
            chosen_move = sorted_moves[0]
            action_id = cards_to_action(chosen_move)
            if action_id in legal_actions:
                return action_id
            return int(legal_actions[0])

        # Normal Hoarding Mode:
        # 1. If the current table trick rank is already high (>= Jack), PASS!
        if current_trick and current_trick.primary_card >= 0:
            trick_top_rank = get_rank(current_trick.primary_card)
            if trick_top_rank >= self.high_pass_threshold:
                return 0  # Strategic pass, hoard cards!

        # 2. Hoard Aces (rank 11) and 2s (rank 12):
        # Filter moves that do NOT consume an Ace or a 2.
        safe_moves = [
            m for m in non_pass_moves
            if all(get_rank(c) < 11 for c in m)
        ]

        if safe_moves:
            # Play the lowest safe move (minimal beating)
            sorted_safe = sorted(safe_moves, key=lambda x: evaluate_hand(x).cmp_score)
            chosen_move = sorted_safe[0]
            action_id = cards_to_action(chosen_move)
            if action_id in legal_actions:
                return action_id

        # 3. If beating requires wasting an Ace or 2, Sullivan prefers to PASS!
        return 0
