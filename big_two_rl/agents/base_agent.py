"""
Base Agent Interface for Big Two (Capsa Banting) Agents.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Sequence, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all Big Two agents.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(
        self,
        env: Any,
        player_idx: int,
        legal_actions: np.ndarray,
        legal_moves: List[Tuple[int, ...]]
    ) -> int:
        """
        Choose an action_id from legal_actions.

        Args:
            env: The BigTwoEnv or raw_env instance.
            player_idx: The integer player index (0..3).
            legal_actions: NumPy array of integer action IDs that are valid.
            legal_moves: List of valid card tuples corresponding to the actions.

        Returns:
            An integer action_id from legal_actions.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
