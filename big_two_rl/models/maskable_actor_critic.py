"""
High-Performance Maskable Actor-Critic PyTorch Neural Network for Capsa Banting (Big Two).
Implements:
- 3-Layer Shared Backbone: [512, 512, 256] with ELU activation and LayerNorm.
- Policy Head: Linear(256, 1696) with Invalid Action Masking (-1e8 logits masking).
- Value Head: Linear(256, 1) scalar state value estimator.
- Full compatibility with RLAgent interface and batched CUDA operations.
"""
from typing import Tuple, Optional, Union, Any, Dict
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from big_two_rl.env.observation import OBS_DIM
from big_two_rl.env.actions import NUM_ACTIONS


class MaskableActorCritic(nn.Module):
    """
    Grandmaster-grade Actor-Critic network with invalid action masking,
    ELU activations, and LayerNorm stabilization.
    """
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = NUM_ACTIONS,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared Backbone: MLP 3-layer [512, 512, 256] + ELU + LayerNorm
        self.layer1 = nn.Linear(obs_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.act1 = nn.ELU()

        self.layer2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.act2 = nn.ELU()

        self.layer3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.act3 = nn.ELU()

        # Policy Head (Actor): 256 -> 1696
        self.policy_head = nn.Linear(256, action_dim)

        # Value Head (Critic): 256 -> 1
        self.value_head = nn.Linear(256, 1)

        # Orthogonal weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in [self.layer1, self.layer2, self.layer3]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.ln1(self.layer1(obs)))
        x = self.act2(self.ln2(self.layer2(x)))
        x = self.act3(self.ln3(self.layer3(x)))
        return x

    def forward(
        self,
        obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        logits = self.policy_head(features)
        values = self.value_head(features)

        if action_masks is not None:
            # Set invalid action logits to -1e8 before Softmax
            mask_bool = action_masks.bool()
            logits = torch.where(mask_bool, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))

        return logits, values

    def get_action(
        self,
        obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the masked distribution.
        Returns:
            (actions, values, log_probs, entropy)
        """
        logits, values = self.forward(obs, action_masks=action_masks)
        dist = Categorical(logits=logits)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, values.squeeze(-1), log_probs, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_probs, entropy, and state values for PPO updates.
        """
        logits, values = self.forward(obs, action_masks=action_masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values.squeeze(-1), log_probs, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.value_head(features).squeeze(-1)

    def predict(
        self,
        observation: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, Optional[Any]]:
        """
        RLAgent prediction interface. Accepts single numpy vector or batch.
        """
        device = next(self.parameters()).device
        was_1d = (observation.ndim == 1)

        obs_arr = observation[np.newaxis, :] if was_1d else observation
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)

        mask_tensor = None
        if action_masks is not None:
            mask_arr = action_masks[np.newaxis, :] if was_1d else action_masks
            mask_tensor = torch.as_tensor(mask_arr, dtype=torch.bool, device=device)

        self.eval()
        with torch.no_grad():
            actions, _, _, _ = self.get_action(obs_tensor, action_masks=mask_tensor, deterministic=deterministic)

        if was_1d:
            return int(actions[0].item()), None
        return actions.cpu().numpy(), None

    def save(self, filepath: str):
        """Save model state dict to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "state_dict": self.state_dict()
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "MaskableActorCritic":
        """Load model state dict from disk."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            obs_dim=checkpoint.get("obs_dim", OBS_DIM),
            action_dim=checkpoint.get("action_dim", NUM_ACTIONS)
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
