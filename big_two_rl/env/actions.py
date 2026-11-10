"""
Action space mapping for Capsa Banting (Big Two).

Action Space: gymnasium.spaces.Discrete(1696)
- ID 0: Pass (tuple ())
- ID 1-52: Singles (52 variations, 1 card each)
- ID 53-130: Pairs (78 variations, 13 ranks * 6 suit pairs)
- ID 131-182: Triples (52 variations, 13 ranks * 4 suit triples)
- ID 183-1695: 5-Card hands (1513 valid variations):
  - 183 to 1469: Flushes (1287 combinations)
  - 1470 to 1625: Full Houses (156 combinations)
  - 1626 to 1677: Quads (52 combinations)
  - 1678 to 1687: Straights (10 variations)
  - 1688 to 1695: Straight Flushes (8 variations)
"""
import itertools
from typing import Tuple, Dict, List, Optional, Sequence, Any
import numpy as np
from gymnasium import spaces

from big_two_rl.engine.card import make_card, get_rank, get_suit
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    evaluate_hand,
    STRAIGHT_RANK_MAP,
)

NUM_ACTIONS = 1696

# Precomputed action tables
ACTION_TO_CARDS: List[Tuple[int, ...]] = [()] * NUM_ACTIONS
CARDS_TO_ACTION: Dict[Tuple[int, ...], int] = {}
# Abstract category and signature to action ID for flexible mapping
SIGNATURE_TO_ACTION: Dict[Tuple[Any, ...], int] = {}


def _init_action_space():
    global ACTION_TO_CARDS, CARDS_TO_ACTION, SIGNATURE_TO_ACTION

    idx = 0

    # 1. Action 0: Pass
    ACTION_TO_CARDS[0] = ()
    CARDS_TO_ACTION[()] = 0
    SIGNATURE_TO_ACTION[(HandType.PASS,)] = 0
    idx += 1

    # 2. Actions 1-52: Singles
    for c in range(52):
        tup = (c,)
        ACTION_TO_CARDS[idx] = tup
        CARDS_TO_ACTION[tup] = idx
        SIGNATURE_TO_ACTION[(HandType.SINGLE, c)] = idx
        idx += 1
    assert idx == 53, f"Expected idx 53 after singles, got {idx}"

    # 3. Actions 53-130: Pairs (78 variations)
    for r in range(13):
        for s1, s2 in itertools.combinations(range(4), 2):
            c1 = make_card(r, s1)
            c2 = make_card(r, s2)
            tup = (c1, c2)
            ACTION_TO_CARDS[idx] = tup
            CARDS_TO_ACTION[tup] = idx
            SIGNATURE_TO_ACTION[(HandType.PAIR, r, s1, s2)] = idx
            idx += 1
    assert idx == 131, f"Expected idx 131 after pairs, got {idx}"

    # 4. Actions 131-182: Triples (52 variations)
    for r in range(13):
        for suits in itertools.combinations(range(4), 3):
            cards = tuple(make_card(r, s) for s in suits)
            ACTION_TO_CARDS[idx] = cards
            CARDS_TO_ACTION[cards] = idx
            SIGNATURE_TO_ACTION[(HandType.TRIPLE, r, suits)] = idx
            idx += 1
    assert idx == 183, f"Expected idx 183 after triples, got {idx}"

    # 5. Actions 183-1695: 5-Card Hands (1513 variations)
    # A) 1287 Flushes: all 5-combinations of 13 ranks (suits default to Diamond=0 for canonical)
    for rank_comb in itertools.combinations(range(13), 5):
        canonical_cards = tuple(make_card(r, 0) for r in rank_comb)
        ACTION_TO_CARDS[idx] = canonical_cards
        CARDS_TO_ACTION[canonical_cards] = idx
        SIGNATURE_TO_ACTION[(HandType.FLUSH, rank_comb)] = idx
        idx += 1
    assert idx == 183 + 1287  # 1470

    # B) 156 Full Houses: all pairs of (triplet_rank, pair_rank) with r1 != r2
    for r_trip in range(13):
        for r_pair in range(13):
            if r_trip == r_pair:
                continue
            canonical_cards = (
                make_card(r_trip, 0), make_card(r_trip, 1), make_card(r_trip, 2),
                make_card(r_pair, 0), make_card(r_pair, 1)
            )
            canonical_cards = tuple(sorted(canonical_cards))
            ACTION_TO_CARDS[idx] = canonical_cards
            CARDS_TO_ACTION[canonical_cards] = idx
            SIGNATURE_TO_ACTION[(HandType.FULL_HOUSE, r_trip, r_pair)] = idx
            idx += 1
    assert idx == 1470 + 156  # 1626

    # C) 52 Quads: 13 quad ranks * 4 kicker ranks/suits
    for r_quad in range(13):
        for k_idx in range(4):
            # choose kicker rank distinct from quad rank
            k_rank = (r_quad + 1 + k_idx) % 13
            canonical_cards = (
                make_card(r_quad, 0), make_card(r_quad, 1),
                make_card(r_quad, 2), make_card(r_quad, 3),
                make_card(k_rank, 0)
            )
            canonical_cards = tuple(sorted(canonical_cards))
            ACTION_TO_CARDS[idx] = canonical_cards
            CARDS_TO_ACTION[canonical_cards] = idx
            SIGNATURE_TO_ACTION[(HandType.QUADS, r_quad, k_idx)] = idx
            idx += 1
    assert idx == 1626 + 52  # 1678

    # D) 10 Straights: from STRAIGHT_RANK_MAP (excluding J-Q-K-A-2 or first 10)
    straight_rank_keys = list(STRAIGHT_RANK_MAP.keys())[:10]
    for str_ranks in straight_rank_keys:
        # Canonical cards: suits cycling 0, 1, 2, 3, 0
        canonical_cards = tuple(make_card(str_ranks[i], i % 4) for i in range(5))
        canonical_cards = tuple(sorted(canonical_cards))
        ACTION_TO_CARDS[idx] = canonical_cards
        CARDS_TO_ACTION[canonical_cards] = idx
        SIGNATURE_TO_ACTION[(HandType.STRAIGHT, str_ranks)] = idx
        idx += 1
    assert idx == 1678 + 10  # 1688

    # E) 8 Straight Flushes
    sf_rank_keys = list(STRAIGHT_RANK_MAP.keys())[:8]
    for sf_ranks in sf_rank_keys:
        canonical_cards = tuple(make_card(r, 0) for r in sf_ranks)
        canonical_cards = tuple(sorted(canonical_cards))
        ACTION_TO_CARDS[idx] = canonical_cards
        CARDS_TO_ACTION[canonical_cards] = idx
        SIGNATURE_TO_ACTION[(HandType.STRAIGHT_FLUSH, sf_ranks)] = idx
        idx += 1
    assert idx == NUM_ACTIONS, f"Expected {NUM_ACTIONS} total actions, got {idx}"


_init_action_space()


def action_to_cards(
    action_id: int,
    hand: Optional[Sequence[int]] = None,
    legal_moves: Optional[Sequence[Sequence[int]]] = None
) -> Tuple[int, ...]:
    """
    Map an action ID (0..1695) to a tuple of cards.
    If legal_moves is provided, selects the valid legal move in legal_moves
    that corresponds to this action_id.
    If hand is provided, searches hand for a combination matching this action.
    Otherwise returns the canonical cards tuple.
    """
    assert 0 <= action_id < NUM_ACTIONS, f"Action ID must be in [0, {NUM_ACTIONS-1}], got {action_id}"
    if action_id == 0:
        return ()

    # If legal_moves is provided, find the move in legal_moves that maps to this action_id
    if legal_moves is not None:
        for m in legal_moves:
            if cards_to_action(m) == action_id:
                return tuple(m)

    canonical = ACTION_TO_CARDS[action_id]
    if hand is None:
        return canonical

    hand_set = set(hand)
    # If the canonical cards are directly in the hand, return them
    if all(c in hand_set for c in canonical):
        return canonical

    # If action is Single, Pair, Triple: they are strictly concrete
    if action_id < 183:
        return canonical

    # For 5-card actions (Flushes, Straights, Full Houses, Quads, SF):
    for key, act in SIGNATURE_TO_ACTION.items():
        if act == action_id:
            sig = key
            break
    else:
        return canonical

    from big_two_rl.engine.game_state import get_all_combinations_in_hand
    for combo in get_all_combinations_in_hand(hand):
        if combo_to_signature(combo) == sig:
            return combo.cards

    return canonical


def combo_to_signature(combo: HandCombo) -> Tuple[Any, ...]:
    """Extract signature for matching with SIGNATURE_TO_ACTION."""
    ct = combo.combo_type
    if ct == HandType.PASS:
        return (HandType.PASS,)
    elif ct == HandType.SINGLE:
        return (HandType.SINGLE, combo.cards[0])
    elif ct == HandType.PAIR:
        r = get_rank(combo.cards[0])
        s1, s2 = get_suit(combo.cards[0]), get_suit(combo.cards[1])
        return (HandType.PAIR, r, s1, s2)
    elif ct == HandType.TRIPLE:
        r = get_rank(combo.cards[0])
        suits = tuple(get_suit(c) for c in combo.cards)
        return (HandType.TRIPLE, r, suits)
    elif ct == HandType.FLUSH:
        ranks = tuple(sorted(get_rank(c) for c in combo.cards))
        return (HandType.FLUSH, ranks)
    elif ct == HandType.FULL_HOUSE:
        from collections import Counter
        counts = Counter(get_rank(c) for c in combo.cards)
        trip_r = next(r for r, count in counts.items() if count == 3)
        pair_r = next(r for r, count in counts.items() if count == 2)
        return (HandType.FULL_HOUSE, trip_r, pair_r)
    elif ct == HandType.QUADS:
        from collections import Counter
        counts = Counter(get_rank(c) for c in combo.cards)
        quad_r = next(r for r, count in counts.items() if count == 4)
        kicker_r = next(r for r, count in counts.items() if count == 1)
        k_idx = (kicker_r - quad_r - 1) % 13
        k_idx = min(k_idx, 3)
        return (HandType.QUADS, quad_r, k_idx)
    elif ct == HandType.STRAIGHT:
        ranks = tuple(sorted(get_rank(c) for c in combo.cards))
        return (HandType.STRAIGHT, ranks)
    elif ct == HandType.STRAIGHT_FLUSH:
        ranks = tuple(sorted(get_rank(c) for c in combo.cards))
        return (HandType.STRAIGHT_FLUSH, ranks)
    return (ct,)


def cards_to_action(cards_tuple: Sequence[int]) -> int:
    """
    Map any valid combination (cards_tuple) to an action ID in [0, 1695].
    """
    tup = tuple(sorted(cards_tuple))
    if tup in CARDS_TO_ACTION:
        return CARDS_TO_ACTION[tup]

    combo = evaluate_hand(tup)
    if combo is None:
        raise ValueError(f"Cards {cards_tuple} do not form a valid combination")

    sig = combo_to_signature(combo)
    if sig in SIGNATURE_TO_ACTION:
        return SIGNATURE_TO_ACTION[sig]

    # Fallback for straights or straight flushes with alternative rank keys
    if combo.combo_type in (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH):
        ranks = tuple(sorted(get_rank(c) for c in tup))
        for key, act in SIGNATURE_TO_ACTION.items():
            if key[0] == combo.combo_type:
                return act

    # Fallback for quads with different kicker offset
    if combo.combo_type == HandType.QUADS:
        quad_r = sig[1]
        for k_idx in range(4):
            if (HandType.QUADS, quad_r, k_idx) in SIGNATURE_TO_ACTION:
                return SIGNATURE_TO_ACTION[(HandType.QUADS, quad_r, k_idx)]

    raise ValueError(f"Could not map combo {combo} to an action ID")


def get_action_mask(legal_moves: Sequence[Sequence[int]], as_int8: bool = False) -> np.ndarray:
    """
    Generate an action mask of shape (1696,) where
    mask[a] = True (or 1) ONLY if action a is legal.
    If as_int8 is True, returns dtype np.int8 (required by Gymnasium Discrete.sample).
    Otherwise returns dtype bool.
    """
    dtype = np.int8 if as_int8 else bool
    mask = np.zeros(NUM_ACTIONS, dtype=dtype)
    for move in legal_moves:
        act = cards_to_action(move)
        mask[act] = 1 if as_int8 else True
    return mask
