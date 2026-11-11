"""
Opponent Profiler & Behavioral Fingerprinting Engine for Capsa Banting (Big Two).
Tracks agent identity, win/loss history, card pattern distributions, and playing habits
for explicit bilateral opponent modeling in Multi-Agent RL.
"""
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np

from big_two_rl.engine.card import get_rank, get_suit
from big_two_rl.engine.evaluator import HandType

CANONICAL_AGENTS = ["Curt", "Albert", "Sullivan", "Bernstein"]
AGENT_NAME_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_AGENTS)}

PROFILE_DIM_PER_PLAYER = 16  # 16 features per opponent


class PlayerProfile:
    """
    Detailed statistical profile and behavioral fingerprint for an individual player.
    Generates a normalized 16-dimensional feature vector:
    - [0:4]: Agent Identity One-Hot (Curt, Albert, Sullivan, Bernstein)
    - [4]: Tournament Win Rate (wins / max(1, games))
    - [5]: Normalized Average Penalty Score (avg_penalty / 52.0)
    - [6]: Recent Form / Momentum (win rate in last 10 games)
    - [7]: Head-to-Head Win Rate against focal agent
    - [8]: Aggressiveness Ratio (plays / max(1, total_turns))
    - [9]: Single Play Preference Ratio (singles / max(1, total_plays))
    - [10]: Pair Play Preference Ratio (pairs / max(1, total_plays))
    - [11]: Triple Play Preference Ratio (triples / max(1, total_plays))
    - [12]: 5-Card Play Preference Ratio (5-cards / max(1, total_plays))
    - [13]: 2-Hoarding Timing Index (avg cards remaining when playing rank 2 / 13.0)
    - [14]: Strategic Pass Frequency (passes on beatable tricks / max(1, total_passes))
    - [15]: Critical Conversion Rate (wins / max(1, times_reached_le3_cards))
    """
    def __init__(self, name: str):
        self.name = name
        self.agent_id = AGENT_NAME_TO_ID.get(name, -1)

        # Macro Tournament Stats
        self.games_played: int = 0
        self.wins: int = 0
        self.total_penalties: int = 0
        self.recent_results: deque = deque(maxlen=10)  # 1 for win, 0 for loss
        self.head_to_head_wins: Dict[str, int] = {a: 0 for a in CANONICAL_AGENTS}
        self.head_to_head_games: Dict[str, int] = {a: 0 for a in CANONICAL_AGENTS}

        # Micro Behavioral Patterns
        self.total_turns: int = 0
        self.total_plays: int = 0
        self.total_passes: int = 0
        self.singles_played: int = 0
        self.pairs_played: int = 0
        self.triples_played: int = 0
        self.five_cards_played: int = 0

        # Timing & Tactics
        self.twos_played_remaining_cards: List[int] = []
        self.strategic_passes: int = 0
        self.times_reached_le3_cards: int = 0
        self.wins_from_le3_cards: int = 0

    def record_turn(
        self,
        cards_played: Tuple[int, ...],
        remaining_cards_before: int,
        can_beat: bool,
    ):
        """Record an individual turn decision by this player."""
        self.total_turns += 1
        n_cards = len(cards_played)

        if remaining_cards_before <= 3:
            self.times_reached_le3_cards += 1

        if n_cards == 0:
            self.total_passes += 1
            if can_beat:
                self.strategic_passes += 1
        else:
            self.total_plays += 1
            if n_cards == 1:
                self.singles_played += 1
                if get_rank(cards_played[0]) == 12:  # Rank 2
                    self.twos_played_remaining_cards.append(remaining_cards_before)
            elif n_cards == 2:
                self.pairs_played += 1
                if any(get_rank(c) == 12 for c in cards_played):
                    self.twos_played_remaining_cards.append(remaining_cards_before)
            elif n_cards == 3:
                self.triples_played += 1
            elif n_cards == 5:
                self.five_cards_played += 1

    def record_game_end(
        self,
        won: bool,
        remaining_cards: int,
        penalty: int,
        other_players: List[str]
    ):
        """Record game completion metrics."""
        self.games_played += 1
        if won:
            self.wins += 1
            self.recent_results.append(1)
            self.wins_from_le3_cards += 1
        else:
            self.recent_results.append(0)

        self.total_penalties += penalty

        for opp in other_players:
            if opp != self.name:
                self.head_to_head_games[opp] = self.head_to_head_games.get(opp, 0) + 1
                if won:
                    self.head_to_head_wins[opp] = self.head_to_head_wins.get(opp, 0) + 1

    def to_vector(
        self,
        focal_agent_name: Optional[str] = None,
        positional_stats: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Export a 16-dimensional float32 normalized feature vector.
        All elements are strictly bounded in [0.0, 1.0].
        Positional stats specifically condition on this opponent being at this
        exact relative seat position (Left, Right, Opposite) relative to the focal agent.
        """
        vec = np.zeros(PROFILE_DIM_PER_PLAYER, dtype=np.float32)

        # 1. Identity one-hot [0:4]
        if 0 <= self.agent_id < 4:
            vec[self.agent_id] = 1.0

        # 2. Tournament Win Rate [4]
        if self.games_played > 0:
            vec[4] = float(np.clip(self.wins / self.games_played, 0.0, 1.0))

        # 3. Average Penalty Score [5]
        if self.games_played > 0:
            avg_pen = self.total_penalties / self.games_played
            vec[5] = float(np.clip(avg_pen / 52.0, 0.0, 1.0))

        # 4. Positional Focal Loss Rate [6]:
        # Answers: "Ketika lawan ini ada di samping/posisi ini, seberapa sering saya kalah?"
        if positional_stats is not None and positional_stats["games"] > 0:
            vec[6] = float(np.clip(positional_stats["focal_losses"] / positional_stats["games"], 0.0, 1.0))
        elif self.recent_results:
            vec[6] = float(np.clip(sum(1 - r for r in self.recent_results) / len(self.recent_results), 0.0, 1.0))

        # 5. Positional Opponent Win Rate [7]:
        # Answers: "Seberapa sering lawan ini menang ketika duduk di posisi ini di samping saya?"
        if positional_stats is not None and positional_stats["games"] > 0:
            vec[7] = float(np.clip(positional_stats["opp_wins"] / positional_stats["games"], 0.0, 1.0))
        elif focal_agent_name and focal_agent_name in self.head_to_head_games:
            h2h_g = self.head_to_head_games[focal_agent_name]
            if h2h_g > 0:
                vec[7] = float(np.clip(self.head_to_head_wins[focal_agent_name] / h2h_g, 0.0, 1.0))

        # 6. Aggressiveness Ratio [8]
        if self.total_turns > 0:
            vec[8] = float(np.clip(self.total_plays / self.total_turns, 0.0, 1.0))

        # 7. Combo Preferences [9:13]
        # Purely data-driven: 0.0 before any cards played; reflects emergent preferences as training unfolds
        if self.total_plays > 0:
            vec[9] = float(np.clip(self.singles_played / self.total_plays, 0.0, 1.0))
            vec[10] = float(np.clip(self.pairs_played / self.total_plays, 0.0, 1.0))
            vec[11] = float(np.clip(self.triples_played / self.total_plays, 0.0, 1.0))
            vec[12] = float(np.clip(self.five_cards_played / self.total_plays, 0.0, 1.0))

        # 8. 2-Hoarding Timing Index [13]
        # Emerges purely from empirical remaining card count when agent chooses to deploy rank 2
        if self.twos_played_remaining_cards:
            avg_rem = np.mean(self.twos_played_remaining_cards)
            vec[13] = float(np.clip(avg_rem / 13.0, 0.0, 1.0))

        # 9. Strategic Pass Frequency [14]
        if self.total_passes > 0:
            vec[14] = float(np.clip(self.strategic_passes / self.total_passes, 0.0, 1.0))

        # 10. Critical Conversion Rate [15]
        if self.times_reached_le3_cards > 0:
            vec[15] = float(np.clip(self.wins_from_le3_cards / self.times_reached_le3_cards, 0.0, 1.0))

        return vec


class OpponentProfileManager:
    """
    Coordinates and maintains historical profiles for all agents at the table.
    Provides bilateral opponent awareness context (Left, Right, Opposite)
    for any observing seat.
    """
    def __init__(self, agent_names: Optional[List[str]] = None):
        if agent_names is None:
            agent_names = list(CANONICAL_AGENTS)
        self.profiles: Dict[str, PlayerProfile] = {
            name: PlayerProfile(name) for name in agent_names
        }
        # Seat mapping: seat_idx (0..3) -> agent_name
        self.seat_to_agent: Dict[int, str] = {
            i: agent_names[i % len(agent_names)] for i in range(4)
        }
        # Positional matchup matrix: (focal_agent, opponent_agent, rel_position) -> Dict[str, int]
        # Tracks specific co-occurrence matchups (e.g. "when Sullivan is on my right, I frequently lose")
        self.positional_matchups: Dict[Tuple[str, str, str], Dict[str, int]] = {}

    def set_seat_mapping(self, seat_to_agent_map: Dict[int, str]):
        """Update seat to agent name mapping (e.g. after seat rotation)."""
        self.seat_to_agent = dict(seat_to_agent_map)

    def get_agent_name_at_seat(self, seat: int) -> str:
        return self.seat_to_agent.get(seat, f"player_{seat}")

    def get_adjacent_profiles(self, focal_seat: int) -> np.ndarray:
        """
        Returns a 48-dimensional float32 vector containing the profiles of
        the three opponents in canonical clockwise order relative to focal_seat:
        - [0:16]: Left Player Profile ((focal_seat + 1) % 4)
        - [16:32]: Opposite Player Profile ((focal_seat + 2) % 4)
        - [32:48]: Right Player Profile ((focal_seat + 3) % 4)
        """
        focal_agent = self.get_agent_name_at_seat(focal_seat)

        left_seat = (focal_seat + 1) % 4
        opp_seat = (focal_seat + 2) % 4
        right_seat = (focal_seat + 3) % 4

        left_agent = self.get_agent_name_at_seat(left_seat)
        opp_agent = self.get_agent_name_at_seat(opp_seat)
        right_agent = self.get_agent_name_at_seat(right_seat)

        pos_left = self.positional_matchups.get((focal_agent, left_agent, "left"))
        pos_opp = self.positional_matchups.get((focal_agent, opp_agent, "opposite"))
        pos_right = self.positional_matchups.get((focal_agent, right_agent, "right"))

        vec_left = self.profiles[left_agent].to_vector(focal_agent, pos_left) if left_agent in self.profiles else np.zeros(16, dtype=np.float32)
        vec_opp = self.profiles[opp_agent].to_vector(focal_agent, pos_opp) if opp_agent in self.profiles else np.zeros(16, dtype=np.float32)
        vec_right = self.profiles[right_agent].to_vector(focal_agent, pos_right) if right_agent in self.profiles else np.zeros(16, dtype=np.float32)

        return np.concatenate([vec_left, vec_opp, vec_right], dtype=np.float32)

    def record_turn(
        self,
        seat: int,
        cards_played: Tuple[int, ...],
        remaining_cards_before: int,
        can_beat: bool,
    ):
        """Record turn statistics for the agent currently sitting at seat."""
        agent_name = self.get_agent_name_at_seat(seat)
        if agent_name in self.profiles:
            self.profiles[agent_name].record_turn(
                cards_played=cards_played,
                remaining_cards_before=remaining_cards_before,
                can_beat=can_beat
            )

    def record_round_results(
        self,
        winner_seat: int,
        remaining_cards_per_seat: List[int],
        penalties_per_seat: List[int],
    ):
        """Record round finish outcomes for all 4 seats and update positional co-occurrence matrix."""
        all_agents = [self.get_agent_name_at_seat(s) for s in range(4)]

        # 1. Update individual profiles
        for s in range(4):
            agent_name = all_agents[s]
            if agent_name in self.profiles:
                won = (s == winner_seat)
                self.profiles[agent_name].record_game_end(
                    won=won,
                    remaining_cards=remaining_cards_per_seat[s],
                    penalty=penalties_per_seat[s],
                    other_players=all_agents
                )

        # 2. Update positional matchup matrix (Focal -> Opponent at Left, Opposite, Right)
        for focal_seat in range(4):
            focal_agent = all_agents[focal_seat]
            focal_won = (focal_seat == winner_seat)

            rel_positions = [
                ((focal_seat + 1) % 4, "left"),
                ((focal_seat + 2) % 4, "opposite"),
                ((focal_seat + 3) % 4, "right"),
            ]
            for opp_seat, rel_pos in rel_positions:
                opp_agent = all_agents[opp_seat]
                opp_won = (opp_seat == winner_seat)
                key = (focal_agent, opp_agent, rel_pos)
                if key not in self.positional_matchups:
                    self.positional_matchups[key] = {
                        "games": 0,
                        "focal_losses": 0,
                        "focal_wins": 0,
                        "opp_wins": 0,
                    }
                stats = self.positional_matchups[key]
                stats["games"] += 1
                if not focal_won:
                    stats["focal_losses"] += 1
                else:
                    stats["focal_wins"] += 1
                if opp_won:
                    stats["opp_wins"] += 1

    def reset(self):
        """Reset all profiles and positional matchup history to initial neutral state."""
        for p in self.profiles.values():
            p.__init__(p.name)
        self.positional_matchups.clear()
