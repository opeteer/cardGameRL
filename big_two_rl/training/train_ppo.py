#!/usr/bin/env python3
"""
MaskablePPO Self-Play Training Pipeline for Big Two (Capsa Banting).
Trains an RL agent using sb3-contrib MaskablePPO with Invalid Action Masking
against a historical checkpoint pool.
Supports vectorized environments (DummyVecEnv) for accelerated throughput.
"""
import os
import argparse
import time
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import gymnasium

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from big_two_rl.training.self_play import SelfPlayEnv, SelfPlayCallback
from big_two_rl.env.big_two_env import raw_env


def mask_fn(env: gymnasium.Env) -> np.ndarray:
    """Helper function to extract action mask for ActionMasker wrapper."""
    return env.action_masks()


def make_single_env(seed: int = 42) -> Callable[[], gymnasium.Env]:
    """Factory helper creating an individual SelfPlay environment."""
    def _init():
        base = SelfPlayEnv(randomize_learner_seat=True)
        wrapped = ActionMasker(base, mask_fn)
        return wrapped
    return _init


def evaluate_agent(
    model: MaskablePPO,
    num_games: int = 100,
    seed: int = 1000
) -> Tuple[float, float]:
    """
    Evaluate trained agent (as Agent 0) against 3 random masked walk agents.
    Returns:
        (win_rate, average_payoff)
    """
    env = raw_env()
    wins = 0
    total_payoff = 0.0

    for g in range(num_games):
        env.reset(seed=seed + g)
        while not env.game_state.game_over:
            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            mask = env.action_mask(curr_agent, as_int8=True)
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break

            if p_idx == 0:
                obs = env.observe(curr_agent)
                bool_mask = (mask == 1)
                action, _ = model.predict(obs, action_masks=bool_mask, deterministic=True)
                if not bool_mask[action]:
                    action = np.random.choice(legal)
            else:
                action = np.random.choice(legal)

            env.step(action)

        if env.game_state.winner == 0:
            wins += 1
        payoffs = env.game_state.get_payoffs()
        total_payoff += payoffs[0]

    win_rate = wins / num_games
    avg_payoff = total_payoff / num_games
    return win_rate, avg_payoff


def train_ppo(
    total_timesteps: int = 1000000,
    save_freq: int = 50000,
    num_envs: int = 8,
    eval_games: int = 50,
    checkpoints_dir: str = "checkpoints",
    learning_rate: float = 3e-4,
    device: str = "auto",
    seed: int = 42
):
    print("=" * 65)
    print("Capsa Banting MARL - MaskablePPO Self-Play Training")
    print("=" * 65)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs:   {num_envs}")
    print(f"Save Frequency:  {save_freq:,}")
    print(f"Checkpoints Dir: {checkpoints_dir}")
    print(f"Device:          {device} (CUDA available: {torch.cuda.is_available()})")
    print("=" * 65)

    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Instantiate Vectorized SelfPlay Environments
    if num_envs > 1:
        vec_env = DummyVecEnv([make_single_env(seed + i) for i in range(num_envs)])
    else:
        base_env = SelfPlayEnv(randomize_learner_seat=True)
        vec_env = DummyVecEnv([lambda: ActionMasker(base_env, mask_fn)])

    # 2. Configure MaskablePPO Model
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device
    )

    # 3. Setup Callback
    self_play_callback = SelfPlayCallback(
        save_freq=max(1000, save_freq // num_envs),
        checkpoints_dir=checkpoints_dir,
        verbose=1
    )

    # 4. Training Loop
    start_time = time.time()
    print(f"\nStarting training loop for {total_timesteps:,} timesteps across {num_envs} environments...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=self_play_callback,
        progress_bar=False
    )
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds ({total_timesteps / elapsed:.1f} steps/s)!")

    # 5. Save Final Model
    final_path = os.path.join(checkpoints_dir, "final_model.zip")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    # 6. Evaluate against Random Agents
    if eval_games > 0:
        print("\n" + "-" * 65)
        print(f"Evaluating trained model against 3 Random Agents for {eval_games} games...")
        win_rate, avg_payoff = evaluate_agent(model, num_games=eval_games)
        print(f"Evaluation Results:")
        print(f"Win Rate:       {win_rate * 100:.1f}% (Random baseline is 25.0%)")
        print(f"Average Payoff: {avg_payoff:+.2f} points per game")
        print("-" * 65)

    vec_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Big Two MaskablePPO Agent")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--eval-games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        num_envs=args.num_envs,
        eval_games=args.eval_games,
        checkpoints_dir=args.checkpoints_dir,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed
    )
