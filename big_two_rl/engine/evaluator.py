"""
Combination evaluator and comparator for Capsa Banting (Big Two).

Valid Combinations:
- Single: 1 card
- Pair: 2 cards of same rank
- Triple / Tris: 3 cards of same rank
- 5-Card Hands (Package 5):
  1. Straight (5 consecutive ranks, suit free)
  2. Flush (5 cards of same suit)
  3. Full House (3 of same rank + 2 of same rank)
  4. Four of a Kind / Quads (4 of same rank + 1 kicker)
  5. Straight Flush (5 consecutive cards of same suit)

5-Card Package Hierarchy:
Straight < Flush < Full House < Quads < Straight Flush
"""
import functools
from enum import IntEnum
from typing import Sequence, Tuple, Optional, Dict, Any, List, Set
from collections import Counter, defaultdict
from big_two_rl.engine.card import get_rank, get_suit, card_to_str


class HandType(IntEnum):
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    QUADS = 7
    STRAIGHT_FLUSH = 8


STRAIGHT_RANK_MAP: Dict[Tuple[int, ...], Tuple[int, int]] = {
    (0, 1, 2, 11, 12): (0, 2),   # A-2-3-4-5 -> top rank is 5 (index 2)
    (0, 1, 2, 3, 12):  (1, 3),   # 2-3-4-5-6 -> top rank is 6 (index 3)
    (0, 1, 2, 3, 4):   (2, 4),   # 3-4-5-6-7 -> top rank is 7 (index 4)
    (1, 2, 3, 4, 5):   (3, 5),   # 4-5-6-7-8 -> top rank is 8 (index 5)
    (2, 3, 4, 5, 6):   (4, 6),   # 5-6-7-8-9 -> top rank is 9 (index 6)
    (3, 4, 5, 6, 7):   (5, 7),   # 6-7-8-9-10 -> top rank is 10 (index 7)
    (4, 5, 6, 7, 8):   (6, 8),   # 7-8-9-10-J -> top rank is J (index 8)
    (5, 6, 7, 8, 9):   (7, 9),   # 8-9-10-J-Q -> top rank is Q (index 9)
    (6, 7, 8, 9, 10):  (8, 10),  # 9-10-J-Q-K -> top rank is K (index 10)
    (7, 8, 9, 10, 11): (9, 11),  # 10-J-Q-K-A -> top rank is A (index 11)
    (8, 9, 10, 11, 12):(10, 12), # J-Q-K-A-2 -> top rank is 2 (index 12)
}


class HandCombo:
    """
    Evaluated card combination holding combo type, sorted cards,
    and a normalized comparison score tuple for fast comparisons.
    """
    __slots__ = ('combo_type', 'cards', 'primary_card', 'cmp_score')

    def __init__(
        self,
        combo_type: HandType,
        cards: Sequence[int],
        primary_card: int,
        cmp_score: Tuple[Any, ...]
    ):
        self.combo_type = combo_type
        self.cards = tuple(sorted(cards))
        self.primary_card = primary_card
        self.cmp_score = cmp_score

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        card_names = [card_to_str(c) for c in self.cards]
        return f"HandCombo({self.combo_type.name}, cards={card_names}, primary={card_to_str(self.primary_card) if self.primary_card >= 0 else 'None'})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandCombo):
            return False
        return (self.combo_type == other.combo_type and 
                self.cmp_score == other.cmp_score and
                self.cards == other.cards)

    def beats(self, other: 'HandCombo') -> bool:
        """
        Check if self beats other according to Big Two rules.
        """
        return compare_combos(self, other) > 0


def evaluate_hand(cards: Sequence[int]) -> Optional[HandCombo]:
    """
    Evaluate a sequence of card integers and return a HandCombo,
    or None if the combination is invalid.
    Cards do not need to be presorted.
    """
    n = len(cards)
    if n == 0:
        return HandCombo(HandType.PASS, (), -1, (int(HandType.PASS),))

    sorted_cards = sorted(cards)
    ranks = [get_rank(c) for c in sorted_cards]
    suits = [get_suit(c) for c in sorted_cards]

    if n == 1:
        # Single: determined by card value (monotonic with rank then suit)
        c = sorted_cards[0]
        return HandCombo(
            HandType.SINGLE,
            sorted_cards,
            primary_card=c,
            cmp_score=(int(HandType.SINGLE), ranks[0], suits[0])
        )

    if n == 2:
        # Pair: both cards must have the same rank
        if ranks[0] == ranks[1]:
            # Penentu: rank, lalu highest suit in the pair
            top_card = sorted_cards[1]
            return HandCombo(
                HandType.PAIR,
                sorted_cards,
                primary_card=top_card,
                cmp_score=(int(HandType.PAIR), ranks[0], suits[1])
            )
        return None

    if n == 3:
        # Triple: all 3 cards must have the same rank
        if ranks[0] == ranks[1] == ranks[2]:
            top_card = sorted_cards[2]
            return HandCombo(
                HandType.TRIPLE,
                sorted_cards,
                primary_card=top_card,
                cmp_score=(int(HandType.TRIPLE), ranks[0])
            )
        return None

    if n == 5:
        # 5-card hands: Straight, Flush, Full House, Quads, Straight Flush
        rank_counts = Counter(ranks)
        unique_ranks = tuple(sorted(rank_counts.keys()))
        is_flush = (len(set(suits)) == 1)
        straight_info = STRAIGHT_RANK_MAP.get(unique_ranks, None)
        is_straight = (straight_info is not None)

        # 5. Straight Flush
        if is_straight and is_flush:
            str_order, top_rank = straight_info
            top_card = next(c for c in sorted_cards if get_rank(c) == top_rank)
            top_suit = get_suit(top_card)
            return HandCombo(
                HandType.STRAIGHT_FLUSH,
                sorted_cards,
                primary_card=top_card,
                cmp_score=(int(HandType.STRAIGHT_FLUSH), top_suit, str_order)
            )

        # 4. Four of a Kind / Quads (4 + 1)
        if len(rank_counts) == 2 and 4 in rank_counts.values():
            quad_rank = next(r for r, count in rank_counts.items() if count == 4)
            quad_top_card = max(c for c in sorted_cards if get_rank(c) == quad_rank)
            return HandCombo(
                HandType.QUADS,
                sorted_cards,
                primary_card=quad_top_card,
                cmp_score=(int(HandType.QUADS), quad_rank)
            )

        # 3. Full House (3 + 2)
        if len(rank_counts) == 2 and 3 in rank_counts.values():
            triplet_rank = next(r for r, count in rank_counts.items() if count == 3)
            triplet_top_card = max(c for c in sorted_cards if get_rank(c) == triplet_rank)
            return HandCombo(
                HandType.FULL_HOUSE,
                sorted_cards,
                primary_card=triplet_top_card,
                cmp_score=(int(HandType.FULL_HOUSE), triplet_rank)
            )

        # 2. Flush
        if is_flush:
            flush_suit = suits[0]
            rev_ranks = tuple(sorted(ranks, reverse=True))
            top_card = sorted_cards[-1]
            return HandCombo(
                HandType.FLUSH,
                sorted_cards,
                primary_card=top_card,
                cmp_score=(int(HandType.FLUSH), flush_suit, rev_ranks)
            )

        # 1. Straight
        if is_straight:
            str_order, top_rank = straight_info
            top_card = next(c for c in sorted_cards if get_rank(c) == top_rank)
            top_suit = get_suit(top_card)
            return HandCombo(
                HandType.STRAIGHT,
                sorted_cards,
                primary_card=top_card,
                cmp_score=(int(HandType.STRAIGHT), str_order, top_suit)
            )

        return None

    # Big Two does not allow 4 cards or > 5 cards
    return None


def is_valid_combo(cards: Sequence[int]) -> bool:
    """Return True if the sequence of cards forms a valid Big Two combination."""
    return evaluate_hand(cards) is not None


def compare_combos(challenger: HandCombo, defender: HandCombo) -> int:
    """
    Compare two evaluated combinations.
    Returns:
       1 if challenger beats defender
       0 if equal
      -1 if defender beats challenger or incompatible types

    Rules:
    - If lengths differ, incompatible (cannot beat), returns -1.
    - If length in (1, 2, 3), combo_types must match.
    - If length == 5:
        If combo_types differ: higher HandType wins.
        If combo_types match: cmp_score determines winner.
    """
    if len(challenger) != len(defender):
        return -1

    if len(challenger) in (1, 2, 3):
        if challenger.combo_type != defender.combo_type:
            return -1
        if challenger.cmp_score > defender.cmp_score:
            return 1
        elif challenger.cmp_score < defender.cmp_score:
            return -1
        return 0

    if len(challenger) == 5:
        if challenger.combo_type > defender.combo_type:
            return 1
        elif challenger.combo_type < defender.combo_type:
            return -1
        else:
            if challenger.cmp_score > defender.cmp_score:
                return 1
            elif challenger.cmp_score < defender.cmp_score:
                return -1
            return 0

    return -1


@functools.lru_cache(maxsize=32768)
def compute_hand_partition_metrics(hand_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Solve for optimal partition of hand_tuple into disjoint valid combinations
    (Singles, Pairs, Triples, 5-Card Combos) to minimize total steps/tricks needed N_step(H),
    and count unpaired low singles (rank <= 8, index <= 5) U_singles(H).
    Returns:
        (N_step, U_singles)
    """
    if not hand_tuple:
        return 0, 0

    hand_set = set(hand_tuple)
    by_rank = defaultdict(list)
    by_suit = defaultdict(list)
    for c in hand_tuple:
        by_rank[get_rank(c)].append(c)
        by_suit[get_suit(c)].append(c)

    combos: List[Set[int]] = []

    # 1. Quads (+ 1 kicker)
    for r, cards_r in by_rank.items():
        if len(cards_r) == 4:
            for c_other in hand_tuple:
                if c_other not in cards_r:
                    combos.append(set(cards_r + [c_other]))

    # 2. Full Houses (Triple + Pair)
    triples = [cards_r for r, cards_r in by_rank.items() if len(cards_r) >= 3]
    for trip_cand in triples:
        t_comb = trip_cand[:3]
        for r2, cards_r2 in by_rank.items():
            if r2 != get_rank(t_comb[0]) and len(cards_r2) >= 2:
                combos.append(set(t_comb + cards_r2[:2]))

    # 3. Flushes (5 cards of same suit)
    for s, cards_s in by_suit.items():
        if len(cards_s) >= 5:
            combos.append(set(sorted(cards_s)[-5:]))

    # 4. Straights (5 consecutive ranks)
    unique_ranks = set(by_rank.keys())
    for rank_pat in STRAIGHT_RANK_MAP.keys():
        if all(r in unique_ranks for r in rank_pat):
            combos.append(set([by_rank[r][0] for r in rank_pat]))

    def solve_remainder(rem_cards: Set[int]) -> Tuple[int, int]:
        rc = Counter(get_rank(c) for c in rem_cards)
        steps = 0
        unpaired_low = 0
        for r, cnt in rc.items():
            if cnt == 4:
                steps += 2
            elif cnt == 3 or cnt == 2:
                steps += 1
            elif cnt == 1:
                steps += 1
                if r <= 5:  # rank <= 8 in Big Two (3, 4, 5, 6, 7, 8)
                    unpaired_low += 1
        return steps, unpaired_low

    # Case 0: No 5-card combos
    best_steps, best_u = solve_remainder(hand_set)

    # Case 1: Exactly 1 5-card combo
    for c1 in combos:
        s, u = solve_remainder(hand_set - c1)
        if 1 + s < best_steps or (1 + s == best_steps and u < best_u):
            best_steps = 1 + s
            best_u = u

    # Case 2: Exactly 2 disjoint 5-card combos
    num_c = len(combos)
    for i in range(num_c):
        c1 = combos[i]
        for j in range(i + 1, num_c):
            c2 = combos[j]
            if c1.isdisjoint(c2):
                s, u = solve_remainder(hand_set - (c1 | c2))
                if 2 + s < best_steps or (2 + s == best_steps and u < best_u):
                    best_steps = 2 + s
                    best_u = u

    return best_steps, best_u


def get_optimal_partition_steps(hand: Sequence[int]) -> int:
    """Convenience helper returning N_step(H)."""
    return compute_hand_partition_metrics(tuple(sorted(hand)))[0]


def get_unpaired_low_singles(hand: Sequence[int]) -> int:
    """Convenience helper returning U_singles(H)."""
    return compute_hand_partition_metrics(tuple(sorted(hand)))[1]
