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

    def test_feeding_tier2_penalty(self):
        engine = RewardEngine()
        gs = GameState()
        gs.reset(seed=42)
        gs.hands[1] = [50, 51]  # Next player has 2 cards

        # Hand before has a low single and a pair of 8s (could play combo)
        hand_b = [make_card(2, 0), make_card(5, 0), make_card(5, 1)]
        r, details = engine.compute_step_reward(
            player_idx=0,
            cards_played=[make_card(2, 0)],  # Single 5♦ (rank 2 < 8)
            hand_before=hand_b,
            hand_after=[make_card(5, 0), make_card(5, 1)],
            boss_cards_before=set(),
            boss_cards_after=set(),
            game_state=gs,
            can_beat_trick=False
        )
        assert details.get("penalty_feeding_tier2") == -12.0, "Must trigger PENALTY_FEEDING_TIER2 (-12.0)"

    def test_combo_starve_bonus(self):
        engine = RewardEngine()
        gs = GameState()
        gs.reset(seed=42)
        # Next player (p=1) previously passed on PAIR
        gs.passed_combos[1].add(HandType.PAIR)

        r, details = engine.compute_step_reward(
            player_idx=0,
            cards_played=[make_card(3, 0), make_card(3, 1)],  # Pair of 6s
            hand_before=[make_card(3, 0), make_card(3, 1)],
            hand_after=[],
            boss_cards_before=set(),
            boss_cards_after=set(),
            game_state=gs,
            can_beat_trick=False,
            is_free_lead_before=True
        )
        assert details.get("bonus_combo_starve") == 2.5, "Must trigger BONUS_COMBO_STARVE (+2.5)"

    def test_interception_bonus(self):
        engine = RewardEngine()
        gs = GameState()
        gs.reset(seed=42)
        # Downstream player has 2 cards (in danger)
        gs.hands[1] = [50, 51]
        # Trick led by upstream player (player 3)
        trick_leader = 3

        r, details = engine.compute_step_reward(
            player_idx=0,
            cards_played=[make_card(12, 3)],  # Intercept with 2♠
            hand_before=[make_card(12, 3)],
            hand_after=[],
            boss_cards_before={make_card(12, 3)},
            boss_cards_after=set(),
            game_state=gs,
            can_beat_trick=True,
            trick_leader_before=trick_leader,
            is_free_lead_before=False
        )
        assert details.get("bonus_intercept") == 2.0, "Must trigger BONUS_INTERCEPT (+2.0)"


class TestObservation308:
    def test_observation_dimensions_and_features(self):
        gs = GameState()
        gs.reset(seed=42)
        obs = encode_observation(gs, 0)
        assert obs.shape == (308,)
        assert OBS_DIM == 308
        # Range checks
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_bilateral_awareness_features(self):
        gs = GameState()
        gs.reset(seed=42)

        # 1. Left player has 1 card -> Urgency should be 1.0, 1-card indicator True
        gs.hands[1] = [51]
        obs = encode_observation(gs, 0)
        assert obs[224] == 1.0  # Left critical 1-card
        assert obs[238] == 1.0  # Left <= 2 cards
        assert obs[241] == 1.0  # Left <= 3 cards
        assert obs[256] == 1.0  # Downstream urgency = max(0, 4-1)/3 = 1.0

        # 2. Pass memory encoding
        gs.passed_combos[1].add(HandType.PAIR)
        obs2 = encode_observation(gs, 0)
        assert obs2[245] == 1.0  # Left passed Pair
        assert obs2[244] == 0.0  # Left did not pass Single

        # 3. Relative trick leader: Right player (seat 3 relative to seat 0 is rel 3)
        gs.current_trick_cards = (make_card(0, 0),)
        gs.current_trick_combo = evaluate_hand(gs.current_trick_cards)
        gs.trick_lead_player = 3
        obs3 = encode_observation(gs, 0)
        assert obs3[237] == 1.0  # Leader is relative 3 (Right player)
        assert obs3[234] == 0.0  # Leader is not Self

    def test_adjacent_opponent_profiling_integration(self):
        from big_two_rl.engine.opponent_profiler import OpponentProfileManager
        profiler = OpponentProfileManager()
        # Seat 0: Curt, Seat 1: Albert, Seat 2: Bernstein, Seat 3: Sullivan
        profiler.set_seat_mapping({0: "Curt", 1: "Albert", 2: "Bernstein", 3: "Sullivan"})

        # Record Albert winning 2 games and playing aggressively
        profiler.profiles["Albert"].record_game_end(won=True, remaining_cards=0, penalty=0, other_players=["Curt", "Bernstein", "Sullivan"])
        profiler.profiles["Albert"].record_game_end(won=True, remaining_cards=0, penalty=0, other_players=["Curt", "Bernstein", "Sullivan"])

        gs = GameState()
        gs.reset(seed=42)
        # Curt (seat 0) observes table
        obs = encode_observation(gs, 0, profile_manager=profiler)
        assert obs.shape == (308,)

        # Left player is seat 1 (Albert): identity one-hot [260:264] has Albert=1.0 (index 1)
        assert obs[260] == 0.0  # Curt
        assert obs[261] == 1.0  # Albert!
        assert obs[262] == 0.0  # Sullivan
        assert obs[263] == 0.0  # Bernstein
        # Albert's win rate [264] should be 1.0 (2 wins / 2 games)
        assert obs[264] == 1.0

        # Right player is seat 3 (Sullivan): identity one-hot [292:296] has Sullivan=1.0 (index 2)
        assert obs[292] == 0.0  # Curt
        assert obs[293] == 0.0  # Albert
        assert obs[294] == 1.0  # Sullivan!
        assert obs[295] == 0.0  # Bernstein

    def test_positional_adjacent_vulnerability_learning(self):
        """
        Verify scenario: 'Ketika Sullivan ada di samping kanan saya, saya sering kalah.'
        Curt must directly observe high positional loss rate against Sullivan on right side.
        """
        from big_two_rl.engine.opponent_profiler import OpponentProfileManager
        profiler = OpponentProfileManager()
        # Seat 0: Curt, Seat 1: Albert, Seat 2: Bernstein, Seat 3: Sullivan (Sullivan on Curt's right!)
        profiler.set_seat_mapping({0: "Curt", 1: "Albert", 2: "Bernstein", 3: "Sullivan"})

        # Simulate 10 rounds where Sullivan (seat 3) wins 8 times, and Curt (seat 0) loses all 10 rounds
        for _ in range(8):
            profiler.record_round_results(
                winner_seat=3,  # Sullivan wins
                remaining_cards_per_seat=[5, 4, 3, 0],
                penalties_per_seat=[5, 4, 3, 0]
            )
        for _ in range(2):
            profiler.record_round_results(
                winner_seat=1,  # Albert wins
                remaining_cards_per_seat=[6, 0, 4, 2],
                penalties_per_seat=[6, 0, 4, 2]
            )

        gs = GameState()
        gs.reset(seed=42)
        obs = encode_observation(gs, 0, profile_manager=profiler)

        # Right opponent is Sullivan: profile slice [292:308]
        # obs[298] (vec[6] of Right Profile) is Curt's positional loss rate when Sullivan is on his right
        assert obs[298] == 1.0, f"Expected 100% loss rate for Curt when Sullivan is on right, got {obs[298]}"
        # obs[299] (vec[7] of Right Profile) is Sullivan's win rate when on the right of Curt
        assert obs[299] == 0.8, f"Expected 80% win rate for Sullivan on Curt's right, got {obs[299]}"


class TestMaskableActorCriticArchitecture:
    def test_network_shapes_and_masked_logits(self):
        model = MaskableActorCritic(obs_dim=308, action_dim=1696)
        obs_t = torch.randn(4, 308)
        mask_t = torch.zeros(4, 1696, dtype=torch.bool)
        mask_t[:, :5] = True  # Only actions 0..4 legal

        actions, values, log_probs, entropy = model.get_action(obs_t, action_masks=mask_t)
        assert actions.shape == (4,)
        assert values.shape == (4,)
        assert log_probs.shape == (4,)
        assert (actions < 5).all(), "All sampled actions must be <= 4"

    def test_dynamic_shape_adapter_backward_compatibility(self):
        # A legacy model initialized with obs_dim=234 must accept 308-dim obs seamlessly
        legacy_model = MaskableActorCritic(obs_dim=234, action_dim=1696)
        obs_new = torch.randn(2, 308)
        mask_t = torch.ones(2, 1696, dtype=torch.bool)
        actions, values, log_probs, _ = legacy_model.get_action(obs_new, action_masks=mask_t)
        assert actions.shape == (2,)
