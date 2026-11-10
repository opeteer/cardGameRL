"""
Low-latency standalone rule engine for Big Two (Capsa Banting).
"""
from big_two_rl.engine.card import (
    Card,
    card_to_str,
    str_to_card,
    get_rank,
    get_suit,
    make_card,
    cards_to_str,
    RANK_CHARS,
    SUIT_CHARS,
    SUIT_SYMBOLS,
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
    get_legal_moves,
    calculate_penalty,
)
from big_two_rl.engine.entropy import (
    HardwareEntropyHarvester,
    get_hardware_seed,
    get_hardware_rng,
)

__all__ = [
    "Card",
    "card_to_str",
    "str_to_card",
    "get_rank",
    "get_suit",
    "make_card",
    "cards_to_str",
    "RANK_CHARS",
    "SUIT_CHARS",
    "SUIT_SYMBOLS",
    "HandType",
    "HandCombo",
    "evaluate_hand",
    "compare_combos",
    "is_valid_combo",
    "GameState",
    "get_legal_moves",
    "calculate_penalty",
    "HardwareEntropyHarvester",
    "get_hardware_seed",
    "get_hardware_rng",
]
