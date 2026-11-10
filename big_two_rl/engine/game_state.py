"""
Game state management, card dealing, trick tracking, and legal move generation.
Pure Python and NumPy compatible, zero external dependencies.
"""
import itertools
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional, Dict, Set
import numpy as np

from big_two_rl.engine.card import get_rank, get_suit
from big_two_rl.engine.evaluator import (
    HandType,
    HandCombo,
    evaluate_hand,
    compare_combos,
    STRAIGHT_RANK_MAP,
)


def calculate_penalty(remaining_cards: int, has_2: bool = False) -> int:
    """
    Calculate penalty points for remaining cards according to Capsa Banting rules:
    - remaining < 8: 1x
    - 8 <= remaining < 10: 2x
    - 10 <= remaining < 13: 3x
    - 13: 4x (52 points)
    - Extra penalty: +5 if still holding rank 2 at game end (1_{2 in H_end})
    """
    assert 0 <= remaining_cards <= 13, f"Remaining cards must be in [0, 13], got {remaining_cards}"
    if remaining_cards < 8:
        base = remaining_cards * 1
    elif remaining_cards < 10:
        base = remaining_cards * 2
    elif remaining_cards < 13:
        base = remaining_cards * 3
    else:  # remaining_cards == 13
        base = 13 * 4  # 52

    if has_2 and remaining_cards > 0:
        base += 5
    return base


def get_all_combinations_in_hand(hand: Sequence[int]) -> List[HandCombo]:
    """
    Generate all valid non-pass combinations (Singles, Pairs, Triples, 5-Card hands)
    that can be formed from the given hand cards.
    """
    hand_sorted = sorted(hand)
    n = len(hand_sorted)
    if n == 0:
        return []

    combos: List[HandCombo] = []

    # 1. Singles
    for c in hand_sorted:
        combo = evaluate_hand((c,))
        if combo:
            combos.append(combo)

    # Group cards by rank and suit for fast pattern matching
    by_rank: Dict[int, List[int]] = defaultdict(list)
    by_suit: Dict[int, List[int]] = defaultdict(list)
    for c in hand_sorted:
        by_rank[get_rank(c)].append(c)
        by_suit[get_suit(c)].append(c)

    # 2. Pairs
    for r, cards in by_rank.items():
        if len(cards) >= 2:
            for pair in itertools.combinations(cards, 2):
                combo = evaluate_hand(pair)
                if combo:
                    combos.append(combo)

    # 3. Triples
    for r, cards in by_rank.items():
        if len(cards) >= 3:
            for trip in itertools.combinations(cards, 3):
                combo = evaluate_hand(trip)
                if combo:
                    combos.append(combo)

    # If fewer than 5 cards in hand, no 5-card hands possible
    if n < 5:
        return combos

    # 4. 5-Card Hands:
    # A) Straights & Straight Flushes
    hand_ranks = set(by_rank.keys())
    for target_ranks in STRAIGHT_RANK_MAP.keys():
        if all(r in hand_ranks for r in target_ranks):
            card_choices = [by_rank[r] for r in target_ranks]
            for str_cards in itertools.product(*card_choices):
                combo = evaluate_hand(str_cards)
                if combo:
                    combos.append(combo)

    # B) Flushes (excluding any already captured as straight flush)
    for s, cards in by_suit.items():
        if len(cards) >= 5:
            for flush_cards in itertools.combinations(cards, 5):
                combo = evaluate_hand(flush_cards)
                if combo and combo.combo_type == HandType.FLUSH:
                    combos.append(combo)

    # C) Full Houses
    triplet_ranks = [r for r, cards in by_rank.items() if len(cards) >= 3]
    pair_ranks = [r for r, cards in by_rank.items() if len(cards) >= 2]
    for r_trip in triplet_ranks:
        for trip_cards in itertools.combinations(by_rank[r_trip], 3):
            for r_pair in pair_ranks:
                if r_pair == r_trip:
                    # If this rank had 4 cards, we can take the 4th card with something else,
                    # but cannot use same card in trip and pair
                    continue
                for pair_cards in itertools.combinations(by_rank[r_pair], 2):
                    fh_cards = trip_cards + pair_cards
                    combo = evaluate_hand(fh_cards)
                    if combo:
                        combos.append(combo)

    # D) Quads (Four of a Kind)
    quad_ranks = [r for r, cards in by_rank.items() if len(cards) == 4]
    for r_quad in quad_ranks:
        quad_cards = tuple(by_rank[r_quad])
        for c in hand_sorted:
            if c not in quad_cards:
                quad_hand = quad_cards + (c,)
                combo = evaluate_hand(quad_hand)
                if combo:
                    combos.append(combo)

    return combos


def get_legal_moves(
    hand: Sequence[int],
    current_trick_combo: Optional[HandCombo],
    is_first_turn: bool = False
) -> List[Tuple[int, ...]]:
    """
    Generate all legal moves (as sorted tuples of card integers) for the player.
    - If is_first_turn is True: Must include 3♦ (card 0), Pass is NOT legal.
    - If current_trick_combo is None (Free Lead): Must play a valid combo, Pass is NOT legal.
    - If current_trick_combo is not None: Must beat current_trick_combo, Pass () is legal.
    """
    all_combos = get_all_combinations_in_hand(hand)
    legal_moves: List[Tuple[int, ...]] = []

    if is_first_turn:
        # Must play a combo containing card 0 (3♦)
        for combo in all_combos:
            if 0 in combo.cards:
                legal_moves.append(combo.cards)
        return legal_moves

    if current_trick_combo is None or current_trick_combo.combo_type == HandType.PASS:
        # Free lead: any valid non-pass combination
        for combo in all_combos:
            legal_moves.append(combo.cards)
        return legal_moves

    # Following a trick: Pass is always an option
    legal_moves.append(())

    # Filter combinations that beat the current trick
    for combo in all_combos:
        if combo.beats(current_trick_combo):
            legal_moves.append(combo.cards)

    return legal_moves


class GameState:
    """
    Complete game state for a 4-player Capsa Banting game.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.last_seed = seed
            self.entropy_info = None
        else:
            from big_two_rl.engine.entropy import HardwareEntropyHarvester
            self.entropy_info = HardwareEntropyHarvester.harvest_raw()
            self.last_seed = HardwareEntropyHarvester.generate_seed()
            self.rng = np.random.default_rng(self.last_seed)

        self.hands: List[List[int]] = [[], [], [], []]
        self.discard_pile: List[int] = []
        self.current_player: int = 0
        self.current_trick_cards: Tuple[int, ...] = ()
        self.current_trick_combo: Optional[HandCombo] = None
        self.trick_lead_player: int = 0
        self.pass_count: int = 0
        self.passes: List[bool] = [False, False, False, False]
        self.is_first_turn: bool = True
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None) -> int:
        """
        Deal 13 random cards to each of the 4 players using hardware entropy
        and find the player with 3♦.
        Returns the starting player index (owner of 3♦).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.last_seed = seed
            self.entropy_info = None
        else:
            # 5-Source True Hardware Entropy (RDRAND + RDSEED + CPU Jitter + Mem Latency + Cache Miss)
            from big_two_rl.engine.entropy import HardwareEntropyHarvester
            self.entropy_info = HardwareEntropyHarvester.harvest_raw()
            self.last_seed = HardwareEntropyHarvester.generate_seed()
            self.rng = np.random.default_rng(self.last_seed)

        deck = list(range(52))
        self.rng.shuffle(deck)

        # Realistic casino dealing (interleaved round-robin: 1 card to each player in turn)
        self.hands = [
            sorted(deck[p::4])
            for p in range(4)
        ]
        self.discard_pile = []
        self.current_trick_cards = ()
        self.current_trick_combo = None
        self.pass_count = 0
        self.passes = [False, False, False, False]
        self.is_first_turn = True
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.history = []

        # Find owner of 3♦ (card integer 0)
        start_player = 0
        for p in range(4):
            if 0 in self.hands[p]:
                start_player = p
                break

        self.current_player = start_player
        self.trick_lead_player = start_player
        return start_player

    def is_free_lead(self) -> bool:
        """Return True if the current player is on a Free Lead (empty trick)."""
        return (self.current_trick_combo is None or 
                self.current_trick_combo.combo_type == HandType.PASS)

    def get_legal_actions(self, player_idx: Optional[int] = None) -> List[Tuple[int, ...]]:
        """
        Get all legal moves for the specified player (defaults to current_player).
        """
        p = self.current_player if player_idx is None else player_idx
        if self.game_over:
            return []
        return get_legal_moves(
            self.hands[p],
            self.current_trick_combo,
            is_first_turn=self.is_first_turn
        )

    def step(self, move: Sequence[int]) -> Tuple[bool, Optional[int]]:
        """
        Execute a move (tuple of card ints, or empty tuple () for Pass).
        Returns:
            (game_over, winner)
        """
        if self.game_over:
            return True, self.winner

        p = self.current_player
        move_tuple = tuple(sorted(move))
        legal_moves = self.get_legal_actions(p)

        if move_tuple not in legal_moves:
            raise ValueError(
                f"Illegal move {move_tuple} for player {p}. "
                f"Hand: {self.hands[p]}, Current trick: {self.current_trick_cards}, "
                f"First turn: {self.is_first_turn}"
            )

        self.turn_count += 1
        record = {
            "turn": self.turn_count,
            "player": p,
            "move": move_tuple,
            "is_pass": (len(move_tuple) == 0)
        }

        if len(move_tuple) == 0:
            # Pass
            self.passes[p] = True
            self.pass_count += 1
            record["trick_reset"] = False

            # If 3 consecutive passes occurred, trick is reset
            if self.pass_count == 3:
                self.current_trick_cards = ()
                self.current_trick_combo = None
                self.pass_count = 0
                self.passes = [False, False, False, False]
                # The player who played the last card gets Free Lead
                self.current_player = self.trick_lead_player
                record["trick_reset"] = True
            else:
                self.current_player = (p + 1) % 4
        else:
            # Player plays cards
            combo = evaluate_hand(move_tuple)
            assert combo is not None, f"Move {move_tuple} is not a valid combination"

            # Remove cards from player's hand
            for c in move_tuple:
                self.hands[p].remove(c)

            # Add to discard pile if replacing previous trick cards
            if self.current_trick_cards:
                self.discard_pile.extend(self.current_trick_cards)

            self.current_trick_cards = move_tuple
            self.current_trick_combo = combo
            self.trick_lead_player = p
            self.pass_count = 0
            self.passes = [False, False, False, False]
            self.is_first_turn = False

            # Check if this player has won
            if len(self.hands[p]) == 0:
                self.game_over = True
                self.winner = p
                record["winner"] = p
                self.history.append(record)
                return True, p

            self.current_player = (p + 1) % 4

        self.history.append(record)
        return self.game_over, self.winner

    def get_payoffs(self) -> List[int]:
        """
        Calculate zero-sum payoffs at the end of the game:
        - Winner: + sum of penalties of all 3 opponents
        - Loser: - penalty of own remaining cards
        """
        if not self.game_over or self.winner is None:
            return [0, 0, 0, 0]

        penalties = [
            calculate_penalty(
                len(self.hands[p]),
                has_2=any(get_rank(c) == 12 for c in self.hands[p])
            )
            for p in range(4)
        ]
        w = self.winner
        payoffs = [-penalties[p] for p in range(4)]
        # Winner receives sum of all opponents' penalties
        payoffs[w] = sum(penalties[p] for p in range(4) if p != w)
        return payoffs

    def get_unseen_cards(self, player_idx: int) -> Set[int]:
        """Return the set of card IDs not yet visible to player_idx (opponents' hands)."""
        seen = set(self.discard_pile) | set(self.hands[player_idx])
        return set(range(52)) - seen

    def get_boss_cards(self, player_idx: int) -> Set[int]:
        """
        Return the set of cards in player_idx's hand that hold absolute boss status
        (no unseen card has a strictly higher rank/suit).
        """
        hand = self.hands[player_idx]
        if not hand:
            return set()
        unseen = self.get_unseen_cards(player_idx)
        if not unseen:
            return set(hand)
        max_unseen = max(unseen)
        return {c for c in hand if c > max_unseen}
