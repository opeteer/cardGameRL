"""
Albert - The Aggressive Rusher (Tempo & Boss Slammer).
Relentlessly challenges opponents, never passes when moves are available,
and slams high rank cards and powerful combinations to dominate the board.
"""
from typing import Any, List, Tuple
import numpy as np

from big_two_rl.agents.base_agent import BaseAgent
from big_two_rl.engine.evaluator import evaluate_hand, HandType
from big_two_rl.env.actions import cards_to_action


class AlbertAgent(BaseAgent):
    """
    Albert: Aggressive tempo-oriented agent.
    - Hates passing; always counters if a legal beating move exists.
    - In challenges: Plays the highest available legal move to seize trick leadership.
    - On free lead: Prioritizes high-impact combinations (highest 5-card hands, high pairs, boss singles).
    """
    def __init__(self, name: str = "Albert"):
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

        # Separate pass () from active card moves
        non_pass_moves = [m for m in legal_moves if len(m) > 0]
        if not non_pass_moves:
            return 0  # Forced pass

        # Determine if this is a free lead or challenge
        current_trick = env.game_state.current_trick_combo
        is_free_lead = (current_trick is None or current_trick.combo_type == HandType.PASS)

        if is_free_lead:
            # On free lead, Albert wants to attack with high-impact hands:
            # Hierarchy preference: 5-Card Hands > Triples > Pairs > Singles
            moves_by_cat = {
                5: [],
                3: [],
                2: [],
                1: []
            }
            for m in non_pass_moves:
                moves_by_cat[len(m)].append(m)

            for length in [5, 3, 2, 1]:
                if moves_by_cat[length]:
                    # Pick the strongest within this category
                    sorted_moves = sorted(
                        moves_by_cat[length],
                        key=lambda x: evaluate_hand(x).cmp_score,
                        reverse=True
                    )
                    chosen_move = sorted_moves[0]
                    return cards_to_action(chosen_move)

            # Fallback
            return cards_to_action(non_pass_moves[0])

        else:
            # Challenging: Albert wants to shut down the trick immediately by playing
            # the STRONGEST legal beating move available.
            sorted_moves = sorted(
                non_pass_moves,
                key=lambda x: evaluate_hand(x).cmp_score,
                reverse=True
            )
            chosen_move = sorted_moves[0]
            action_id = cards_to_action(chosen_move)
            if action_id in legal_actions:
                return action_id

            # Fallback to any valid action
            return int(np.random.choice(legal_actions))
