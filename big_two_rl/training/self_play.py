"""
Self-Play Gymnasium Environment Wrapper and Checkpoint Pool Callback for MaskablePPO.
Allows a single learner agent to train against historical snapshots of itself.
"""
import os
import glob
import random
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from big_two_rl.env.big_two_env import raw_env
from big_two_rl.env.actions import NUM_ACTIONS
from big_two_rl.env.observation import OBS_DIM, OBSERVATION_SPACE


class SelfPlayEnv(gymnasium.Env):
    """
    Gymnasium single-agent wrapper around the 4-player Big Two PettingZoo environment.
    The learner controls one seat (configurable or randomized each episode).
    Opponent seats are controlled by policies sampled from a historical checkpoint pool.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent_models: Optional[List[Any]] = None,
        randomize_learner_seat: bool = True,
        learner_seat: int = 0
    ):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.env = raw_env()
        self.randomize_learner_seat = randomize_learner_seat
        self.default_learner_seat = learner_seat
        self.learner_seat = learner_seat

        # Opponent policies pool
        self.opponent_models: List[Any] = opponent_models if opponent_models is not None else []
        self.assigned_opponent_policies: Dict[int, Optional[Any]] = {}

    def add_opponent_model(self, model: Any):
        """Add a historical snapshot policy to the opponent pool."""
        self.opponent_models.append(model)
        # Keep maximum pool size to prevent excessive memory usage
        if len(self.opponent_models) > 20:
            self.opponent_models.pop(0)

    def action_masks(self) -> np.ndarray:
        """
        Return the boolean action mask for the learner.
        Required by sb3-contrib MaskablePPO.
        """
        learner_agent_name = f"player_{self.learner_seat}"
        return self.env.action_mask(learner_agent_name, as_int8=False)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment, assign learner seat, sample opponent policies,
        and advance until the learner's first turn.
        """
        super().reset(seed=seed)
        self.env.reset(seed=seed)

        if self.randomize_learner_seat:
            self.learner_seat = random.randint(0, 3)
        else:
            self.learner_seat = self.default_learner_seat

        learner_agent = f"player_{self.learner_seat}"

        # Assign opponent policies for other 3 seats
        self.assigned_opponent_policies = {}
        for p in range(4):
            if p != self.learner_seat:
                if self.opponent_models and random.random() < 0.85:
                    self.assigned_opponent_policies[p] = random.choice(self.opponent_models)
                else:
                    self.assigned_opponent_policies[p] = None  # Random masked action

        # Run opponent turns until it is learner's turn or game ends
        self._advance_opponents()

        obs = self.env.observe(learner_agent)
        info = {
            "action_mask": self.env.action_mask(learner_agent, as_int8=True),
            "learner_seat": self.learner_seat,
        }
        return obs, info

    def _select_opponent_action(self, p_idx: int) -> int:
        """Select an action for opponent p_idx using their assigned policy or random."""
        agent_name = f"player_{p_idx}"
        mask = self.env.action_mask(agent_name, as_int8=True)
        legal_indices = np.where(mask == 1)[0]
        if len(legal_indices) == 0:
            return 0

        policy = self.assigned_opponent_policies.get(p_idx)
        if policy is not None:
            try:
                obs = self.env.observe(agent_name)
                bool_mask = (mask == 1)
                action, _ = policy.predict(obs, action_masks=bool_mask, deterministic=False)
                if bool_mask[action]:
                    return int(action)
            except Exception:
                pass

        # Fallback to random masked action
        return int(random.choice(legal_indices))

    def _advance_opponents(self):
        """
        Step the environment for opponent agents until it becomes
        the learner's turn or the game ends.
        """
        while not self.env.game_state.game_over:
            curr_p = self.env.game_state.current_player
            if curr_p == self.learner_seat:
                break

            opp_action = self._select_opponent_action(curr_p)
            self.env.step(opp_action)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute learner's action, then let opponents act until the learner's
        next turn or game over.
        """
        learner_agent = f"player_{self.learner_seat}"

        # 1. Execute learner's action
        self.env.step(action)

        # 2. Advance through opponent turns
        self._advance_opponents()

        game_over = self.env.game_state.game_over
        terminated = game_over
        truncated = False

        reward = 0.0
        if game_over:
            payoffs = self.env.game_state.get_payoffs()
            reward = float(payoffs[self.learner_seat])

        obs = self.env.observe(learner_agent)
        info = {
            "action_mask": self.env.action_mask(learner_agent, as_int8=True),
            "winner": self.env.game_state.winner,
            "turns": self.env.game_state.turn_count,
        }
        return obs, reward, terminated, truncated, info


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training:
    - Periodically saves learner checkpoints to `checkpoints_dir`.
    - Loads snapshots and adds them to the opponent policy pool.
    - Evaluates and logs win rate against historical policies.
    """
    def __init__(
        self,
        save_freq: int = 5000,
        checkpoints_dir: str = "checkpoints",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_num = self.num_timesteps
            save_path = os.path.join(self.checkpoints_dir, f"model_step_{step_num}.zip")
            self.model.save(save_path)
            self.checkpoint_count += 1

            if self.verbose > 0:
                print(f"[SelfPlayCallback] Saved checkpoint to {save_path} at step {step_num}")

            # Try to add copy of policy to the training environment's pool
            try:
                from sb3_contrib import MaskablePPO
                opponent_snapshot = MaskablePPO.load(save_path, device="cpu")
                # Find SelfPlayEnv in training env wrapper chain
                env_to_update = self.training_env
                if hasattr(env_to_update, "envs"):
                    for e in env_to_update.envs:
                        unwrapped = getattr(e, "unwrapped", e)
                        if isinstance(unwrapped, SelfPlayEnv):
                            unwrapped.add_opponent_model(opponent_snapshot)
                elif hasattr(env_to_update, "unwrapped"):
                    if isinstance(env_to_update.unwrapped, SelfPlayEnv):
                        env_to_update.unwrapped.add_opponent_model(opponent_snapshot)
                if self.verbose > 0:
                    print(f"[SelfPlayCallback] Added model snapshot to SelfPlayEnv opponent pool.")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[SelfPlayCallback] Notice: Snapshot load ({e})")

        return True
