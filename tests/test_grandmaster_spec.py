"""
Comprehensive Unit Tests for the Grandmaster Maskable PPO Specification:
1. Optimal Hand Partitioning Solver N_step(H) & U_singles(H)
2. Boss Cards Detection B_count(H, D)
3. Strategic Anti-Feeding Hard Constraint in Action Masking
4. Potential-Based Reward Shaping (PBRS) & Tactical Penalties
5. 234-dimensional Observation Vector Engineering
6. MaskableActorCritic [512, 512, 256] ELU + LayerNorm Architecture
"""
import pytest
import numpy as np
import torch

from big_two_rl.engine.card import make_card, get_rank, get_suit
from big_two_rl.engine.evaluator import (
    compute_hand_partition_metrics,
    get_optimal_partition_steps,
    get_unpaired_low_singles,
    evaluate_hand,
    HandType,
)
from big_two_rl.engine.game_state import GameState, calculate_penalty
from big_two_rl.env.observation import encode_observation, OBS_DIM
from big_two_rl.env.masking import generate_action_mask
from big_two_rl.env.actions import cards_to_action
from big_two_rl.env.reward_engine import RewardEngine
from big_two_rl.models.maskable_actor_critic import MaskableActorCritic


class TestHandPartitionSolver:
    def test_partition_pure_singles(self):
        # 5 distinct disconnected cards (ranks 3, 5, 7, 9, K)
        hand = [make_card(0, 0), make_card(2, 1), make_card(4, 2), make_card(6, 3), make_card(10, 0)]
        n_step = get_optimal_partition_steps(hand)
        assert n_step == 5, f"Expected 5 steps for disjoint singles, got {n_step}"

    def test_partition_straight_and_pair(self):
        # Straight (3, 4, 5, 6, 7) + Pair of 2s (card 48, 49) -> 2 steps total!
        straight = [make_card(0, 0), make_card(1, 1), make_card(2, 2), make_card(3, 3), make_card(4, 0)]
        pair_twos = [make_card(12, 0), make_card(12, 1)]
        hand = straight + pair_twos
        n_step = get_optimal_partition_steps(hand)
        assert n_step == 2, f"Expected 2 steps (Straight + Pair), got {n_step}"

    def test_unpaired_low_singles_detection(self):
        # Hand with 3♦ (rank 0), 4♣ (rank 1), and a pair of Aces (rank 11)
        hand = [make_card(0, 0), make_card(1, 1), make_card(11, 0), make_card(11, 1)]
        u_singles = get_unpaired_low_singles(hand)
        assert u_singles == 2, f"Expected 2 unpaired low singles (3 and 4), got {u_singles}"


class TestBossCardsDetection:
    def test_boss_cards_when_holding_2_spades(self):
        gs = GameState()
        gs.reset(seed=42)
        # Find player holding 2♠ (card 51)
        holder = next(p for p in range(4) if 51 in gs.hands[p])
        bosses = gs.get_boss_cards(holder)
        assert 51 in bosses, "2♠ must be an absolute boss card"

    def test_boss_cards_empty_unseen(self):
        gs = GameState()
        gs.reset(seed=42)
        # If all other cards are in discard pile, all held cards are bosses
        p = 0
        gs.discard_pile = list(set(range(52)) - set(gs.hands[p]))
        bosses = gs.get_boss_cards(p)
        assert bosses == set(gs.hands[p])


class TestAntiFeedHardConstraint:
    def test_anti_feed_locks_low_singles_when_opponent_has_1_card(self):
        hand = [make_card(0, 0), make_card(5, 1), make_card(12, 3)]  # 3♦, 8♣, 2♠
        legal_moves = [(), (hand[0],), (hand[1],), (hand[2],)]

        # Next opponent has 1 card: only 2♠ (highest) should remain legal among singles
        mask = generate_action_mask(
            legal_moves=legal_moves,
            hand=hand,
            next_player_card_count=1,
            enforce_anti_feed=True
        )

        act_3 = cards_to_action((hand[0],))
        act_8 = cards_to_action((hand[1],))
        act_2 = cards_to_action((hand[2],))

        assert not mask[act_3], "Low single 3♦ must be locked"
        assert not mask[act_8], "Low single 8♣ must be locked"
        assert mask[act_2], "Highest single 2♠ must remain legal"
        assert mask[0], "Pass must remain legal"


class TestRewardEnginePBRS:
    def test_potential_and_pbrs(self):
        engine = RewardEngine(gamma=0.995)
        hand1 = [0, 5, 10, 15, 20]
        hand2 = [0, 5, 10]
        phi1 = engine.compute_potential(hand1, set())
        phi2 = engine.compute_potential(hand2, set())
        f_pbrs = engine.gamma * phi2 - phi1
        assert isinstance(f_pbrs, float)

    def test_feeding_penalty_trigger(self):
        engine = RewardEngine()
        gs = GameState()
        gs.reset(seed=42)
        # Force next player to have 1 card
        gs.hands[1] = [51]

        r, details = engine.compute_step_reward(
            player_idx=0,
            cards_played=[make_card(2, 0)],  # 5♦ (rank 2 < 8)
            hand_before=[make_card(2, 0), make_card(12, 3)],
            hand_after=[make_card(12, 3)],
            boss_cards_before=set(),
            boss_cards_after=set(),
            game_state=gs,
            can_beat_trick=False
        )
        assert details.get("penalty_feeding") == -30.0, "Must trigger PENALTY_FEEDING (-30.0)"


class TestObservation234:
    def test_observation_dimensions_and_features(self):
        gs = GameState()
        gs.reset(seed=42)
        obs = encode_observation(gs, 0)
        assert obs.shape == (234,)
        assert OBS_DIM == 234
        # Range checks
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


class TestMaskableActorCriticArchitecture:
    def test_network_shapes_and_masked_logits(self):
        model = MaskableActorCritic(obs_dim=234, action_dim=1696)
        obs_t = torch.randn(4, 234)
        mask_t = torch.zeros(4, 1696, dtype=torch.bool)
        mask_t[:, :5] = True  # Only actions 0..4 legal

        actions, values, log_probs, entropy = model.get_action(obs_t, action_masks=mask_t)
        assert actions.shape == (4,)
        assert values.shape == (4,)
        assert log_probs.shape == (4,)
        assert (actions < 5).all(), "All sampled actions must be <= 4"
