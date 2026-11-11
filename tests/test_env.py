"""
Unit tests for the Big Two PettingZoo AEC Environment, action space,
observation encoder, and action masking.
"""
import pytest
import numpy as np
from pettingzoo.test import api_test

from big_two_rl.env.actions import (
    NUM_ACTIONS,
    ACTION_TO_CARDS,
    action_to_cards,
    cards_to_action,
    get_action_mask,
)
from big_two_rl.env.observation import OBS_DIM, OBSERVATION_SPACE, encode_observation
from big_two_rl.env.big_two_env import env, raw_env
from big_two_rl.engine.card import make_card, get_rank, get_suit


class TestActionSpace:
    def test_action_space_dimensions_and_ranges(self):
        assert NUM_ACTIONS == 1696
        assert len(ACTION_TO_CARDS) == 1696

        # ID 0: Pass
        assert action_to_cards(0) == ()
        assert cards_to_action(()) == 0

        # ID 1-52: Singles
        for a in range(1, 53):
            cards = action_to_cards(a)
            assert len(cards) == 1
            assert cards[0] == a - 1
            assert cards_to_action(cards) == a

        # ID 53-130: Pairs (78 variations)
        assert 130 - 53 + 1 == 78
        for a in range(53, 131):
            cards = action_to_cards(a)
            assert len(cards) == 2
            assert get_rank(cards[0]) == get_rank(cards[1])
            assert cards_to_action(cards) == a

        # ID 131-182: Triples (52 variations)
        assert 182 - 131 + 1 == 52
        for a in range(131, 183):
            cards = action_to_cards(a)
            assert len(cards) == 3
            assert get_rank(cards[0]) == get_rank(cards[1]) == get_rank(cards[2])
            assert cards_to_action(cards) == a

        # ID 183-1695: 5-Card Hands (1513 variations)
        assert 1695 - 183 + 1 == 1513
        for a in range(183, 1696):
            cards = action_to_cards(a)
            assert len(cards) == 5

    def test_cards_to_action_various_types(self):
        # Pass
        assert cards_to_action(()) == 0
        # Single 3♦
        assert cards_to_action((0,)) == 1
        # Single 2♠
        assert cards_to_action((51,)) == 52
        # Pair 3♦ 3♣
        assert cards_to_action((0, 1)) == 53
        # Triple 3♦ 3♣ 3♥
        assert cards_to_action((0, 1, 2)) == 131


class TestObservation:
    def test_observation_dimensions_and_bounds(self):
        raw = raw_env()
        raw.reset(seed=42)
        obs = raw.observe("player_0")

        assert OBS_DIM == 326
        assert obs.shape == (326,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

        # Check slices:
        # [0:52]: Hand cards
        assert np.sum(obs[0:52]) == 13.0  # Player has 13 cards at start
        # [52:104]: Discard pile initially empty
        assert np.sum(obs[52:104]) == 0.0
        # [221:224]: 3 opponents each have 13 cards (13 / 13 = 1.0)
        np.testing.assert_allclose(obs[221:224], [1.0, 1.0, 1.0])
        # [238:241]: Opponents <= 2 cards initially all 0.0
        np.testing.assert_allclose(obs[238:241], [0.0, 0.0, 0.0])
        # [241:244]: Opponents <= 3 cards initially all 0.0
        np.testing.assert_allclose(obs[241:244], [0.0, 0.0, 0.0])
        # [256]: Downstream urgency initially 0.0
        assert obs[256] == 0.0
        # [260:308]: 48-dim opponent profiles (Left, Opp, Right)
        assert len(obs[260:308]) == 48
        assert np.all(obs[260:308] >= 0.0)
        assert np.all(obs[260:308] <= 1.0)


class TestActionMasking:
    def test_action_mask_format(self):
        raw = raw_env()
        raw.reset(seed=42)
        mask = raw.action_masks()

        assert mask.shape == (1696,)
        assert mask.dtype == bool
        assert np.any(mask)  # Must have at least one legal action

    def test_first_turn_and_free_lead_masking(self):
        raw = raw_env()
        raw.reset(seed=100)

        # First turn: 3♦ must be played, pass (0) is illegal
        start_agent = raw.agent_selection
        mask = raw.action_mask(start_agent)
        assert mask[0] is np.False_ or mask[0] == False

        # All True actions must correspond to hands containing 3♦ (card 0)
        legal_actions = np.where(mask)[0]
        for a in legal_actions:
            cards = action_to_cards(a, hand=raw.game_state.hands[raw.agent_name_to_idx[start_agent]])
            assert 0 in cards


class TestPettingZooCompliance:
    def test_pettingzoo_api_compliance(self):
        # Run standard PettingZoo API compliance test
        test_env = env()
        api_test(test_env, num_cycles=10)

    def test_full_game_random_masked_actions(self):
        # Play out complete games using random valid masked actions
        for seed in range(3):
            test_env = env()
            test_env.reset(seed=seed)

            for agent in test_env.agent_iter(max_iter=1000):
                obs, reward, terminated, truncated, info = test_env.last()
                if terminated or truncated:
                    test_env.step(None)
                else:
                    mask = info["action_mask"]
                    assert np.any(mask), f"Agent {agent} has no legal actions"
                    legal_indices = np.where(mask)[0]
                    action = np.random.choice(legal_indices)
                    test_env.step(action)

            # At the end, verify zero-sum property
            rewards = [test_env.unwrapped.final_rewards[a] for a in test_env.possible_agents]
            assert sum(rewards) == 0
