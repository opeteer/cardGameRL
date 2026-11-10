"""
Card representations and utilities for Capsa Banting (Big Two).

Card integer formula: c = 4 * r + s
- r in [0, 12] -> '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2'
- s in [0, 3]  -> 0: ♦ (Diamond), 1: ♣ (Club), 2: ♥ (Heart), 3: ♠ (Spade)

Absolute card hierarchy is monotonic with the integer value:
c1 < c2 iff c1 has lower rank or same rank with lower suit.
- Lowest card: 3♦ (c = 0)
- Highest card: 2♠ (c = 51)
"""
from typing import List, Tuple, Union, Sequence

RANK_CHARS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUIT_SYMBOLS = ['♦', '♣', '♥', '♠']
SUIT_CHARS = ['D', 'C', 'H', 'S']
SUIT_NAMES = ['Diamond', 'Club', 'Heart', 'Spade']
IS_RED_SUIT = [True, False, True, False]

RANK_MAP = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
    '8': 5, '9': 6, '10': 7, 'T': 7, 't': 7,
    'J': 8, 'j': 8, 'Q': 9, 'q': 9,
    'K': 10, 'k': 10, 'A': 11, 'a': 11,
    '2': 12
}

SUIT_MAP = {
    '♦': 0, 'D': 0, 'd': 0, 'DIAMOND': 0,
    '♣': 1, 'C': 1, 'c': 1, 'CLUB': 1,
    '♥': 2, 'H': 2, 'h': 2, 'HEART': 2,
    '♠': 3, 'S': 3, 's': 3, 'SPADE': 3
}


def make_card(rank: int, suit: int) -> int:
    """Create card integer c = 4 * rank + suit."""
    assert 0 <= rank <= 12, f"Rank must be in [0, 12], got {rank}"
    assert 0 <= suit <= 3, f"Suit must be in [0, 3], got {suit}"
    return 4 * rank + suit


def get_rank(card: int) -> int:
    """Extract rank r in [0, 12] from card integer."""
    return card // 4


def get_suit(card: int) -> int:
    """Extract suit s in [0, 3] from card integer."""
    return card % 4


def is_red(card: int) -> bool:
    """Return True if card is Diamond (0) or Heart (2)."""
    return (card % 4) in (0, 2)


def card_to_str(card: int, use_unicode: bool = True) -> str:
    """Convert card integer to human-readable string representation."""
    r = get_rank(card)
    s = get_suit(card)
    suit_str = SUIT_SYMBOLS[s] if use_unicode else SUIT_CHARS[s]
    return f"{RANK_CHARS[r]}{suit_str}"


def str_to_card(card_str: str) -> int:
    """
    Parse a string into a card integer.
    Supports formats like '3D', '3♦', '10H', '10♥', '2S', '2♠', 'AS'.
    """
    s = card_str.strip()
    if not s:
        raise ValueError("Empty card string")
    
    if len(s) >= 3 and s[:2] == '10':
        rank_part = '10'
        suit_part = s[2:]
    else:
        rank_part = s[0]
        suit_part = s[1:]
    
    if rank_part not in RANK_MAP:
        raise ValueError(f"Unknown rank in card string '{card_str}'")
    if suit_part not in SUIT_MAP:
        raise ValueError(f"Unknown suit in card string '{card_str}'")
        
    return make_card(RANK_MAP[rank_part], SUIT_MAP[suit_part])


def cards_to_str(cards: Sequence[int], use_unicode: bool = True) -> str:
    """Format a sequence of card integers as a comma-separated string."""
    return "[" + ", ".join(card_to_str(c, use_unicode) for c in cards) + "]"


def cards_to_bitmask(cards: Sequence[int]) -> int:
    """Convert a sequence of card integers to a 64-bit integer mask."""
    mask = 0
    for c in cards:
        mask |= (1 << c)
    return mask


def bitmask_to_cards(bitmask: int) -> List[int]:
    """Convert a 64-bit integer mask back to a sorted list of card integers."""
    cards = []
    for c in range(52):
        if (bitmask >> c) & 1:
            cards.append(c)
    return cards


class Card:
    """
    Lightweight wrapper around an integer card representation.
    Cards can be compared directly using standard operators.
    """
    __slots__ = ('_val',)

    def __init__(self, val: Union[int, str]):
        if isinstance(val, str):
            self._val = str_to_card(val)
        else:
            assert 0 <= val <= 51, f"Card value must be in [0, 51], got {val}"
            self._val = int(val)

    @property
    def val(self) -> int:
        return self._val

    @property
    def rank(self) -> int:
        return get_rank(self._val)

    @property
    def suit(self) -> int:
        return get_suit(self._suit_idx)

    @property
    def _suit_idx(self) -> int:
        return self._val % 4

    @property
    def rank_char(self) -> str:
        return RANK_CHARS[self.rank]

    @property
    def suit_symbol(self) -> str:
        return SUIT_SYMBOLS[self._suit_idx]

    @property
    def suit_char(self) -> str:
        return SUIT_CHARS[self._suit_idx]

    @property
    def is_red(self) -> bool:
        return is_red(self._val)

    def __int__(self) -> int:
        return self._val

    def __index__(self) -> int:
        return self._val

    def __repr__(self) -> str:
        return f"Card({card_to_str(self._val)})"

    def __str__(self) -> str:
        return card_to_str(self._val)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self._val == other._val
        if isinstance(other, int):
            return self._val == other
        return False

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self._val < other._val
        if isinstance(other, int):
            return self._val < other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._val)
