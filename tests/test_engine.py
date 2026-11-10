"""
Unit tests for the Big Two standalone rule engine.
Tests card representation, combination evaluator, hierarchy comparisons,
edge cases, and full game flow to 100% correctness.
"""
import pytest
from big_two_rl.engine.card import (
    Card,
    make_card,
    get_rank,
    get_suit,
    is_red,
    card_to_str,
    str_to_card,
    cards_to_str,
    cards_to_bitmask,
    bitmask_to_cards,
    RANK_CHARS,
    SUIT_CHARS,
)
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    evaluate_hand,
    compare_combos,
    is_valid_combo,
)
from big_two_rl.engine.game_state import (
    GameState,
    calculate_penalty,
    get_legal_moves,
)


class TestCard:
    def test_card_creation_and_monotonicity(self):
        # 3♦ must be 0, 2♠ must be 51
        c_min = make_card(0, 0)
        c_max = make_card(12, 3)
        assert c_min == 0
        assert c_max == 51

        # Check all 52 cards are strictly increasing by rank then suit
        cards = [make_card(r, s) for r in range(13) for s in range(4)]
        assert cards == list(range(52))

    def test_rank_and_suit_extraction(self):
        for r in range(13):
            for s in range(4):
                c = make_card(r, s)
                assert get_rank(c) == r
                assert get_suit(c) == s

    def test_string_conversion(self):
        assert card_to_str(0, use_unicode=False) == "3D"
        assert card_to_str(51, use_unicode=False) == "2S"
        assert card_to_str(str_to_card("10H"), use_unicode=False) == "10H"
        assert str_to_card("3♦") == 0
        assert str_to_card("2♠") == 51
        assert str_to_card("AC") == make_card(11, 1)

    def test_card_object_wrapper(self):
        c1 = Card("3D")
        c2 = Card("2S")
        assert c1.val == 0
        assert c2.val == 51
        assert c1 < c2
        assert c1.rank_char == '3'
        assert c2.rank_char == '2'
        assert c1.is_red is True
        assert c2.is_red is False

    def test_bitmask_roundtrip(self):
        test_cards = [0, 5, 12, 25, 51]
        mask = cards_to_bitmask(test_cards)
        recovered = bitmask_to_cards(mask)
        assert recovered == test_cards


class TestEvaluator:
    def test_singles(self):
        h3d = evaluate_hand([str_to_card("3D")])
        h3s = evaluate_hand([str_to_card("3S")])
        h2s = evaluate_hand([str_to_card("2S")])

        assert h3d.combo_type == HandType.SINGLE
        assert h3s.combo_type == HandType.SINGLE
        assert h3s.beats(h3d)
        assert h2s.beats(h3s)

    def test_pairs(self):
        p3_dc = evaluate_hand([str_to_card("3D"), str_to_card("3C")])
        p3_hs = evaluate_hand([str_to_card("3H"), str_to_card("3S")])
        p4_dc = evaluate_hand([str_to_card("4D"), str_to_card("4C")])
        p2_hs = evaluate_hand([str_to_card("2H"), str_to_card("2S")])

        assert p3_dc.combo_type == HandType.PAIR
        assert p3_hs.beats(p3_dc)  # Same rank, higher suit (Spade > Club)
        assert p4_dc.beats(p3_hs)  # Higher rank
        assert p2_hs.beats(p4_dc)  # Rank 2 is highest

    def test_triples(self):
        t3 = evaluate_hand([str_to_card("3D"), str_to_card("3C"), str_to_card("3H")])
        t4 = evaluate_hand([str_to_card("4D"), str_to_card("4C"), str_to_card("4S")])
        t2 = evaluate_hand([str_to_card("2D"), str_to_card("2H"), str_to_card("2S")])

        assert t3.combo_type == HandType.TRIPLE
        assert t4.beats(t3)
        assert t2.beats(t4)

    def test_invalid_combinations(self):
        # 2 cards with different ranks
        assert evaluate_hand([str_to_card("3D"), str_to_card("4D")]) is None
        # 4 cards (not a legal combo length in Big Two)
        assert evaluate_hand([str_to_card("3D"), str_to_card("3C"), str_to_card("3H"), str_to_card("3S")]) is None
        # 5 cards with random non-poker ranks
        assert evaluate_hand([str_to_card("3D"), str_to_card("5C"), str_to_card("7H"), str_to_card("9S"), str_to_card("2D")]) is None

    def test_straights_regular_and_wraps(self):
        # Regular straight: 3-4-5-6-7
        s_37 = evaluate_hand([str_to_card("3D"), str_to_card("4C"), str_to_card("5H"), str_to_card("6S"), str_to_card("7D")])
        assert s_37.combo_type == HandType.STRAIGHT

        # Highest standard straight: 10-J-Q-K-A
        s_10a = evaluate_hand([str_to_card("10D"), str_to_card("JC"), str_to_card("QH"), str_to_card("KS"), str_to_card("AD")])
        assert s_10a.combo_type == HandType.STRAIGHT

        # Highest possible straight with 2: J-Q-K-A-2
        s_j2 = evaluate_hand([str_to_card("JD"), str_to_card("QC"), str_to_card("KH"), str_to_card("AS"), str_to_card("2D")])
        assert s_j2.combo_type == HandType.STRAIGHT
        assert s_j2.beats(s_10a)
        assert s_10a.beats(s_37)

        # Wrap straights: A-2-3-4-5 and 2-3-4-5-6
        s_a5 = evaluate_hand([str_to_card("AD"), str_to_card("2C"), str_to_card("3H"), str_to_card("4S"), str_to_card("5D")])
        s_26 = evaluate_hand([str_to_card("2D"), str_to_card("3C"), str_to_card("4H"), str_to_card("5S"), str_to_card("6D")])
        assert s_a5.combo_type == HandType.STRAIGHT
        assert s_26.combo_type == HandType.STRAIGHT

        # Straight comparison: 2-3-4-5-6 vs A-2-3-4-5
        # 2-3-4-5-6 has top card 6, A-2-3-4-5 has top card 5 -> 2-3-4-5-6 beats A-2-3-4-5
        assert s_26.beats(s_a5)
        # 3-4-5-6-7 (top 7) beats 2-3-4-5-6 (top 6)
        assert s_37.beats(s_26)

    def test_flushes(self):
        f_d = evaluate_hand([str_to_card("3D"), str_to_card("5D"), str_to_card("7D"), str_to_card("9D"), str_to_card("KD")])
        f_c = evaluate_hand([str_to_card("3C"), str_to_card("5C"), str_to_card("7C"), str_to_card("9C"), str_to_card("KC")])
        f_s = evaluate_hand([str_to_card("3S"), str_to_card("5S"), str_to_card("7S"), str_to_card("9S"), str_to_card("KS")])

        assert f_d.combo_type == HandType.FLUSH
        assert f_c.combo_type == HandType.FLUSH
        assert f_s.combo_type == HandType.FLUSH

        # Penentu: suit, lalu rank
        assert f_c.beats(f_d)  # Club > Diamond
        assert f_s.beats(f_c)  # Spade > Club

        # Same suit flush comparison: higher rank wins
        f_s_low = evaluate_hand([str_to_card("3S"), str_to_card("4S"), str_to_card("6S"), str_to_card("8S"), str_to_card("10S")])
        assert f_s.beats(f_s_low)

    def test_full_house(self):
        # 8-8-8-3-3 vs 7-7-7-A-A
        fh_8_3 = evaluate_hand([str_to_card("8D"), str_to_card("8C"), str_to_card("8H"), str_to_card("3D"), str_to_card("3S")])
        fh_7_a = evaluate_hand([str_to_card("7D"), str_to_card("7C"), str_to_card("7H"), str_to_card("AD"), str_to_card("AS")])
        fh_2_3 = evaluate_hand([str_to_card("2D"), str_to_card("2C"), str_to_card("2H"), str_to_card("3D"), str_to_card("3S")])

        assert fh_8_3.combo_type == HandType.FULL_HOUSE
        assert fh_7_a.combo_type == HandType.FULL_HOUSE
        # Triplet 8 beats triplet 7 (even though pair is AA vs 33)
        assert fh_8_3.beats(fh_7_a)
        assert fh_2_3.beats(fh_8_3)

    def test_four_of_a_kind_quads(self):
        q8 = evaluate_hand([str_to_card("8D"), str_to_card("8C"), str_to_card("8H"), str_to_card("8S"), str_to_card("2S")])
        q7 = evaluate_hand([str_to_card("7D"), str_to_card("7C"), str_to_card("7H"), str_to_card("7S"), str_to_card("AS")])

        assert q8.combo_type == HandType.QUADS
        assert q7.combo_type == HandType.QUADS
        assert q8.beats(q7)

    def test_straight_flush(self):
        sf_d = evaluate_hand([str_to_card("3D"), str_to_card("4D"), str_to_card("5D"), str_to_card("6D"), str_to_card("7D")])
        sf_s = evaluate_hand([str_to_card("3S"), str_to_card("4S"), str_to_card("5S"), str_to_card("6S"), str_to_card("7S")])

        assert sf_d.combo_type == HandType.STRAIGHT_FLUSH
        assert sf_s.combo_type == HandType.STRAIGHT_FLUSH
        # Penentu: suit, lalu rank
        assert sf_s.beats(sf_d)

    def test_5_card_category_hierarchy(self):
        # Hierarchy: Straight < Flush < Full House < Quads < Straight Flush
        s = evaluate_hand([str_to_card("JD"), str_to_card("QC"), str_to_card("KH"), str_to_card("AS"), str_to_card("2D")])  # Straight
        f = evaluate_hand([str_to_card("3D"), str_to_card("5D"), str_to_card("7D"), str_to_card("9D"), str_to_card("JD")])  # Flush
        fh = evaluate_hand([str_to_card("3D"), str_to_card("3C"), str_to_card("3H"), str_to_card("4D"), str_to_card("4S")])  # Full House
        q = evaluate_hand([str_to_card("3D"), str_to_card("3C"), str_to_card("3H"), str_to_card("3S"), str_to_card("4D")])  # Quads
        sf = evaluate_hand([str_to_card("3D"), str_to_card("4D"), str_to_card("5D"), str_to_card("6D"), str_to_card("7D")])  # Straight Flush

        assert f.beats(s)
        assert fh.beats(f)
        assert q.beats(fh)
        assert sf.beats(q)

    def test_incompatible_lengths(self):
        single = evaluate_hand([str_to_card("2S")])
        pair = evaluate_hand([str_to_card("3D"), str_to_card("3C")])
        fh = evaluate_hand([str_to_card("3D"), str_to_card("3C"), str_to_card("3H"), str_to_card("4D"), str_to_card("4S")])

        # Pair cannot beat single, 5-card cannot beat pair
        assert compare_combos(pair, single) == -1
        assert compare_combos(fh, pair) == -1


class TestGameState:
    def test_reset_and_dealing(self):
        gs = GameState(seed=42)
        start_p = gs.reset()

        # Each player has exactly 13 cards
        assert [len(h) for h in gs.hands] == [13, 13, 13, 13]

        # All 52 cards are accounted for with no duplicates
        all_cards = set()
        for h in gs.hands:
            all_cards.update(h)
        assert len(all_cards) == 52

        # 3♦ (card 0) is held by starting player
        assert 0 in gs.hands[start_p]
        assert gs.current_player == start_p
        assert gs.is_first_turn is True

    def test_first_turn_requires_3_diamonds(self):
        gs = GameState(seed=123)
        start_p = gs.reset()

        legal_moves = gs.get_legal_actions()
        assert len(legal_moves) > 0
        # All legal moves on first turn MUST contain card 0
        for move in legal_moves:
            assert 0 in move
        # Pass is NOT allowed on first turn
        assert () not in legal_moves

    def test_free_lead_disallows_pass(self):
        gs = GameState(seed=999)
        gs.reset()
        # On first turn or when trick resets, pass is forbidden
        assert () not in gs.get_legal_actions()

    def test_consecutive_passes_reset_trick(self):
        gs = GameState(seed=777)
        start_p = gs.reset()

        # Player plays 3♦ as single
        gs.step((0,))
        assert gs.current_trick_cards == (0,)
        assert gs.pass_count == 0

        # Next 3 players pass
        for _ in range(3):
            assert () in gs.get_legal_actions()
            gs.step(())

        # After 3 passes, trick resets, leader gets Free Lead
        assert gs.current_trick_cards == ()
        assert gs.current_player == start_p
        assert gs.is_free_lead() is True

    def test_penalty_scoring_rules(self):
        # < 8: 1x
        assert calculate_penalty(0) == 0
        assert calculate_penalty(5) == 5
        assert calculate_penalty(7) == 7
        # 8 <= sisa < 10: 2x
        assert calculate_penalty(8) == 16
        assert calculate_penalty(9) == 18
        # 10 <= sisa < 13: 3x
        assert calculate_penalty(10) == 30
        assert calculate_penalty(12) == 36
        # 13: 4x (52)
        assert calculate_penalty(13) == 52

    def test_zero_sum_payoffs(self):
        gs = GameState(seed=1)
        gs.reset()
        # Force an end game state
        gs.hands[0] = []
        gs.hands[1] = [1, 2, 3]                  # 3 cards -> penalty = 3
        gs.hands[2] = [4, 5, 6, 7, 8, 9, 10, 11]  # 8 cards -> penalty = 16
        gs.hands[3] = list(range(12, 25))        # 13 cards -> penalty = 52
        gs.game_over = True
        gs.winner = 0

        payoffs = gs.get_payoffs()
        assert payoffs[1] == -3
        assert payoffs[2] == -16
        assert payoffs[3] == -52
        assert payoffs[0] == 3 + 16 + 52  # 71
        assert sum(payoffs) == 0

    def test_full_random_game_simulation(self):
        # Run 5 full games with random legal choices to verify engine stability
        for seed in range(5):
            gs = GameState(seed=seed)
            gs.reset()
            steps = 0
            while not gs.game_over and steps < 1000:
                actions = gs.get_legal_actions()
                assert len(actions) > 0, f"No legal actions available for player {gs.current_player}"
                # Sample a random move
                move_idx = gs.rng.choice(len(actions))
                move = actions[move_idx]
                gs.step(move)
                steps += 1

            assert gs.game_over is True
            assert gs.winner is not None
            assert len(gs.hands[gs.winner]) == 0
            payoffs = gs.get_payoffs()
            assert sum(payoffs) == 0
