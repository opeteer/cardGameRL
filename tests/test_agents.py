"""
Unit tests for the 4 Distinct Strategic Agents:
- Curt (RL Agent)
- Albert (Aggressive Rusher)
- Sullivan (Conservative Hoarder)
- Bernstein (Combo Architect)
"""
import pytest
import numpy as np

from big_two_rl.engine.card import make_card
from big_two_rl.engine.evaluator import evaluate_hand
from big_two_rl.env.big_two_env import raw_env
from big_two_rl.agents import (
    create_tournament_agents,
    CurtAgent,
    AlbertAgent,
    SullivanAgent,
    BernsteinAgent
)
from big_two_rl.agents.combo_agent import decompose_hand


class TestAgents:
    def test_hand_decomposition_bernstein(self):
        """Test that decompose_hand correctly partitions into disjoint sets."""
        # Hand with:
        # Straight: 3♦, 4♣, 5♥, 6♠, 7♦ -> cards 0, 5, 10, 15, 16
        # Pair of Kings: K♦, K♠ -> cards 40, 43
        # Single 2♠: card 51
        hand = [0, 5, 10, 15, 16, 40, 43, 51]
        c5, c3, c2, singles = decompose_hand(hand)

        total_cards = sum(len(c) for c in c5) + sum(len(c) for c in c3) + sum(len(c) for c in c2) + len(singles)
        assert total_cards == len(hand), "Partition must conserve card count"
        assert len(c5) == 1, "Should identify the 5-card straight"
        assert len(c2) == 1, "Should identify the pair of Kings"
        assert 51 in singles, "Single 2♠ should be in singles"

    def test_agents_instantiation(self):
        # When all agents share the unified PPO RL Policy
        agents = create_tournament_agents(model_path=None)
        assert len(agents) == 4
        for a in agents:
            assert isinstance(a, CurtAgent) # RLAgent
        assert agents[0].name == "Curt"
        assert agents[1].name == "Albert"
        assert agents[2].name == "Sullivan"
        assert agents[3].name == "Bernstein"

    def test_full_game_with_4_distinct_agents(self):
        """Run multiple full games using the 4 distinct strategies to ensure zero crashes/illegal moves."""
        for seed in [101, 202, 303]:
            env = raw_env()
            env.reset(seed=seed)
            agents = create_tournament_agents(model_path=None)

            step_count = 0
            while not env.game_state.game_over and step_count < 200:
                current_agent_name = env.agent_selection
                p_idx = env.agent_name_to_idx[current_agent_name]
                agent = agents[p_idx]

                mask = env.action_mask(current_agent_name, as_int8=True)
                legal_actions = np.where(mask == 1)[0]
                assert len(legal_actions) > 0, f"No legal actions for {agent.name}"

                legal_moves = env.game_state.get_legal_actions(p_idx)
                action_id = agent.select_action(env, p_idx, legal_actions, legal_moves)

                assert action_id in legal_actions, f"{agent.name} selected illegal action {action_id}"
                env.step(action_id)
                step_count += 1

            assert env.game_state.game_over, "Game should finish cleanly"
            assert env.game_state.winner in [0, 1, 2, 3]

    def test_sullivan_avoids_wasting_2(self):
        """Sullivan should not play a 2 on a low trick if he has alternatives or can pass."""
        sullivan = SullivanAgent()
        env = raw_env()
        env.reset(seed=42)
        # Give Sullivan a 2♠ and some small cards
        # When table has a 5, and Sullivan can pass, check he doesn't throw 2♠ if table is high
        p_idx = 2
        # Mock trick with a King (rank 10)
        env.game_state.hands[p_idx] = [51] # 2♠
        env.game_state.current_trick_combo = evaluate_hand([40]) # K♦
        legal_moves = env.game_state.get_legal_actions(p_idx) # [(), (51,)]
        legal_actions = np.array([0, 52]) # 0 is pass, 52 is 2♠
        action = sullivan.select_action(env, p_idx, legal_actions, legal_moves)
        assert action == 0, "Sullivan should pass instead of burning 2♠ against King when not in crisis"
