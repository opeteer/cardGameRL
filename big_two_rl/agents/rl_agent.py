"""
Reinforcement Learning Agent (MaskablePPO Policy) for Big Two (Capsa Banting).
Can be instantiated for any player seat (Curt, Albert, Sullivan, Bernstein).
"""
import os
from typing import Any, List, Tuple, Optional
import numpy as np

from big_two_rl.agents.base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    Reinforcement Learning Agent powered by MaskablePPO.
    Analyzes full 218-dim observation and selects actions
    via the trained neural network policy with action masking.
    """
    def __init__(
        self,
        name: str = "RL_Agent",
        model_path: Optional[str] = "checkpoints/final_model.zip",
        model: Optional[Any] = None,
        deterministic: bool = False
    ):
        super().__init__(name=name)
        self.model = model
        self.model_path = model_path
        self.deterministic = deterministic

        if self.model is None and model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        # 1. Try native MaskableActorCritic checkpoint (.pt)
        if path.endswith(".pt") or not path.endswith(".zip"):
            try:
                from big_two_rl.models import MaskableActorCritic
                self.model = MaskableActorCritic.load(path, device="cpu")
                print(f"[{self.name}] Model MaskableActorCritic [512,512,256] ELU berhasil dimuat dari: {path}")
                return
            except Exception:
                pass

        # 2. Try SB3 MaskablePPO archive (.zip)
        try:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(path, device="cpu")
            print(f"[{self.name}] Model PPO berhasil dimuat dari: {path}")
        except Exception as e:
            # Fallback to MaskableActorCritic
            try:
                from big_two_rl.models import MaskableActorCritic
                self.model = MaskableActorCritic.load(path, device="cpu")
                print(f"[{self.name}] Model MaskableActorCritic berhasil dimuat dari: {path}")
            except Exception as e2:
                print(f"[{self.name}] Peringatan: Gagal memuat model PPO ({e} / {e2}). Menggunakan fallback acak.")
                self.model = None

    def select_action(
        self,
        env: Any,
        player_idx: int,
        legal_actions: np.ndarray,
        legal_moves: List[Tuple[int, ...]]
    ) -> int:
        if len(legal_actions) == 0:
            return 0

        # Predict using RL policy
        if self.model is not None:
            agent_name = f"player_{player_idx}"
            try:
                obs = env.observe(agent_name)
                mask = env.action_mask(agent_name, as_int8=True)
                bool_mask = (mask == 1)
                action, _ = self.model.predict(
                    obs,
                    action_masks=bool_mask,
                    deterministic=self.deterministic
                )
                action = int(action)
                if bool_mask[action]:
                    return action
            except Exception:
                pass

        # Fallback if prediction fails
        return int(np.random.choice(legal_actions))


# Backward compatibility alias
CurtAgent = RLAgent
