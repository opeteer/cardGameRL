"""
Unit tests for Big Two Luck Meter (Hand Luck Evaluation).
"""
import pytest
from big_two_rl.engine.card import make_card
from big_two_rl.engine.luck_meter import calculate_initial_hand_luck, get_hand_luck_breakdown


def test_hand_luck_empty_and_valid():
    assert calculate_initial_hand_luck([]) == 0


def test_hand_luck_god_hand():
    # God hand: all 4 twos (ranks 12), all 4 Aces (ranks 11), 3 Kings, 2 Queens
    # (Full House, Quads, Twos, 2 of spades, etc.)
    god_hand = [
        make_card(12, 0), make_card(12, 1), make_card(12, 2), make_card(12, 3), # 4 Twos (incl 2♠)
        make_card(11, 0), make_card(11, 1), make_card(11, 2), make_card(11, 3), # 4 Aces
        make_card(10, 0), make_card(10, 1), make_card(10, 2),                    # 3 Kings
        make_card(9, 0), make_card(9, 1),                                        # 2 Queens
    ]
    assert len(god_hand) == 13
    score = calculate_initial_hand_luck(god_hand)
    breakdown = get_hand_luck_breakdown(god_hand)

    assert score > 250
    assert breakdown["twos"] == 4
    assert breakdown["has_2s"] is True
    assert breakdown["aces"] == 4
    assert "2♠ Dewa" in breakdown["highlights"]
    assert "Quads Bom" in breakdown["highlights"]


def test_hand_luck_unlucky_hand():
    # Trash hand: scattered low singles, no pairs, no 2s, no Aces
    trash_hand = [
        make_card(0, 1), # 3♣ (no 3♦)
        make_card(1, 0), # 4♦
        make_card(2, 1), # 5♣
        make_card(3, 2), # 6♥
        make_card(4, 3), # 7♠
        make_card(5, 0), # 8♦
        make_card(6, 1), # 9♣
        make_card(7, 2), # 10♥
        make_card(8, 3), # J♠
        make_card(9, 0), # Q♦
        make_card(1, 2), # 4♥ (one pair 4)
        make_card(2, 3), # 5♠ (one pair 5)
        make_card(0, 2), # 3♥ (one pair 3)
    ]
    assert len(trash_hand) == 13
    score = calculate_initial_hand_luck(trash_hand)
    breakdown = get_hand_luck_breakdown(trash_hand)

    assert score < 120
    assert breakdown["twos"] == 0
    assert breakdown["has_2s"] is False
    assert breakdown["aces"] == 0


def test_hand_luck_relative_order():
    # A hand with 2♠ and 3♦ should score higher than a hand with pure low cards
    strong_hand = [
        make_card(12, 3), # 2♠ (card 51)
        make_card(12, 2), # 2♥
        make_card(11, 3), # A♠
        make_card(11, 2), # A♥
        make_card(0, 0),  # 3♦ (card 0)
        make_card(5, 0), make_card(5, 1), make_card(5, 2), # Triple 8
        make_card(6, 0), make_card(6, 1),                   # Pair 9
        make_card(7, 0), make_card(8, 0), make_card(9, 0)
    ]
    weak_hand = [
        make_card(0, 1), make_card(0, 2),
        make_card(1, 1), make_card(1, 2),
        make_card(2, 1), make_card(2, 2),
        make_card(3, 1), make_card(3, 2),
        make_card(4, 1), make_card(4, 2),
        make_card(5, 1), make_card(5, 2), make_card(6, 1)
    ]
    assert calculate_initial_hand_luck(strong_hand) > calculate_initial_hand_luck(weak_hand)
