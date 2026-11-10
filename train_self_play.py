#!/usr/bin/env python3
"""
High-Performance Maskable PPO Self-Play League Training for Capsa Banting (Big Two).
Implements:
1. 3-Layer Shared Backbone [512, 512, 256] with ELU activation and LayerNorm.
2. Potential-Based Reward Shaping (PBRS) & Grandmaster Tactical Penalties.
3. 234-Dimensional Feature Engineering & Anti-Feeding Hard Constraints.
4. Self-Play League: Active Learner vs 50% Latest, 30% Historical Checkpoints, 20% Baseline Bots.
5. Dynamic Entropy Annealing (0.05 -> 0.001) & Continuous Elo Rating Tracking.
"""
import os
import sys
import copy
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from big_two_rl.env.big_two_env import raw_env
from big_two_rl.env.observation import OBS_DIM
from big_two_rl.env.actions import NUM_ACTIONS
from big_two_rl.models.maskable_actor_critic import MaskableActorCritic


class EloTracker:
    """Tracks dynamic Elo ratings between learner and historical opponents."""
    def __init__(self, initial_elo: float = 1200.0, k_factor: float = 32.0):
        self.k_factor = k_factor
        self.learner_elo = initial_elo
        self.ratings: Dict[str, float] = {"learner": initial_elo, "random": 800.0}

    def record_game(self, winner_seat: int, opp_types: List[str]):
        # Winner seat in [0, 1, 2, 3] where seat 0 is learner
        for seat in range(1, 4):
            opp_key = opp_types[seat - 1]
            opp_elo = self.ratings.get(opp_key, 1200.0)

            # Expected scores
            exp_learner = 1.0 / (1.0 + 10.0 ** ((opp_elo - self.learner_elo) / 400.0))
            exp_opp = 1.0 - exp_learner

            actual_learner = 1.0 if winner_seat == 0 else (0.5 if winner_seat != seat else 0.0)
            actual_opp = 1.0 if winner_seat == seat else (0.5 if winner_seat != 0 else 0.0)

            self.learner_elo += self.k_factor * (actual_learner - exp_learner) / 3.0
            new_opp_elo = opp_elo + self.k_factor * (actual_opp - exp_opp) / 3.0
            self.ratings[opp_key] = new_opp_elo

        self.ratings["learner"] = self.learner_elo


class TrajectoryBuffer:
    """Rollout buffer for the active learner."""
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.masks: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, mask, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.masks.append(mask)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def size(self) -> int:
        return len(self.obs)

    def is_full(self) -> bool:
        return len(self.obs) >= self.capacity

    def get_batches(
        self,
        last_val: float,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        n = len(self.obs)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_val
            else:
                next_val = self.values[t + 1]

            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + self.values[t]

        # Convert to CUDA tensors
        b_obs = torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(np.array(self.actions), dtype=torch.int64, device=device)
        b_masks = torch.as_tensor(np.array(self.masks), dtype=torch.bool, device=device)
        b_values = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device)
        b_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)

        # Normalize advantages
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std()
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

        # Clear buffer
        self.obs.clear()
        self.actions.clear()
        self.masks.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return {
            "obs": b_obs,
            "actions": b_actions,
            "masks": b_masks,
            "values": b_values,
            "log_probs": b_log_probs,
            "returns": b_returns,
            "advantages": b_advantages,
        }


def sample_opponent_action(
    opp_policy: Optional[MaskableActorCritic],
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device
) -> int:
    """Sample action for an opponent seat."""
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0

    if opp_policy is None:
        # 20% baseline bot: random legal move
        return int(np.random.choice(legal))

    obs_t = torch.as_tensor(obs[np.newaxis, :], dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask[np.newaxis, :], dtype=torch.bool, device=device)

    with torch.no_grad():
        dist_logits, _ = opp_policy(obs_t, action_masks=mask_t)
        dist = Categorical(logits=dist_logits)
        action = int(dist.sample().item())

    if not mask[action]:
        action = int(np.random.choice(legal))
    return action


def train_self_play_league(
    total_timesteps: int = 1000000,
    num_envs: int = 32,
    batch_size: int = 2048,
    minibatch_size: int = 256,
    n_epochs: int = 10,
    lr: float = 2.5e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    entropy_start: float = 0.05,
    entropy_end: float = 0.001,
    save_freq: int = 200000,
    checkpoints_dir: str = "checkpoints",
    device_str: str = "cuda",
    seed: int = 42,
):
    print("=" * 75)
    print("GRANDMASTER MASKABLE PPO SELF-PLAY LEAGUE TRAINING (CAPSA BANTING)")
    print("=" * 75)
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Parallel Games:      {num_envs}")
    print(f"Rollout Batch Size:  {batch_size}")
    print(f"Mini-batch Size:     {minibatch_size}")
    print(f"Epochs per Update:   {n_epochs}")
    print(f"Learning Rate:       {lr}")
    print(f"Discount (gamma):    {gamma} (PBRS horizon preservation)")
    print(f"Entropy Annealing:   {entropy_start} -> {entropy_end}")
    print(f"Checkpoints Dir:     {checkpoints_dir}")

    # Configure compute device
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"Compute Device:      {device} ({device_name})")
    print("=" * 75)

    os.makedirs(checkpoints_dir, exist_ok=True)
    league_dir = os.path.join(checkpoints_dir, "league")
    os.makedirs(league_dir, exist_ok=True)

    # 1. Instantiate Active Learner (Grandmaster Model)
    learner = MaskableActorCritic(obs_dim=OBS_DIM, action_dim=NUM_ACTIONS).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr, eps=1e-5)
    elo_tracker = EloTracker()

    # 2. Historical Checkpoint Pool
    historical_pool: List[Tuple[str, MaskableActorCritic]] = []

    # Helper to snapshot learner weights
    def create_snapshot(name: str) -> MaskableActorCritic:
        snap = MaskableActorCritic(obs_dim=OBS_DIM, action_dim=NUM_ACTIONS).to(device)
        snap.load_state_dict(learner.state_dict())
        snap.eval()
        return snap

    latest_snapshot = create_snapshot("latest_0")

    # 3. Initialize Parallel Environments and Opponent Assignments
    envs = [raw_env(use_pbrs_reward=True, enforce_anti_feed=True) for _ in range(num_envs)]
    for e_i, env in enumerate(envs):
        env.reset(seed=seed + e_i * 31)

    # Opponent policy assignments per environment [seat 1, seat 2, seat 3]
    opp_assignments: List[List[Tuple[str, Optional[MaskableActorCritic]]]] = []

    def assign_opponents() -> List[Tuple[str, Optional[MaskableActorCritic]]]:
        # 50% latest snapshot, 30% historical pool, 20% baseline
        seats = []
        for _ in range(3):
            roll = random.random()
            if roll < 0.50 or not historical_pool:
                seats.append(("latest_snap", latest_snapshot))
            elif roll < 0.80 and historical_pool:
                h_name, h_model = random.choice(historical_pool)
                seats.append((h_name, h_model))
            else:
                seats.append(("baseline_bot", None))
        return seats

    for _ in range(num_envs):
        opp_assignments.append(assign_opponents())

    # 4. Rollout Buffer & Metrics Tracking
    buffer = TrajectoryBuffer(capacity=batch_size)
    total_steps = 0
    total_games = 0
    last_saved_step = 0
    start_time = time.time()
    recent_winners = deque(maxlen=200)

    print(f"\nMemulai Self-Play League di {device_name}...")

    # Main Environment Simulation Loop
    while total_steps < total_timesteps:
        # Linear entropy coefficient annealing
        progress = float(total_steps) / float(total_timesteps)
        current_ent_coef = entropy_start - progress * (entropy_start - entropy_end)

        for e_i, env in enumerate(envs):
            if env.game_state.game_over:
                w = env.game_state.winner
                if w is not None:
                    recent_winners.append(w)
                    opp_keys = [opp_assignments[e_i][s][0] for s in range(3)]
                    elo_tracker.record_game(winner_seat=w, opp_types=opp_keys)

                total_games += 1
                opp_assignments[e_i] = assign_opponents()
                env.reset(seed=seed + total_games * 23 + e_i)

            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            obs = env.observe(curr_agent)
            mask = env.action_mask(curr_agent, as_int8=False)

            if p_idx == 0:
                # Active Learner Turn (Seat 0)
                obs_t = torch.as_tensor(obs[np.newaxis, :], dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(mask[np.newaxis, :], dtype=torch.bool, device=device)

                learner.eval()
                with torch.no_grad():
                    action_t, value_t, log_prob_t, _ = learner.get_action(
                        obs_t, action_masks=mask_t, deterministic=False
                    )

                action = int(action_t.item())
                val = float(value_t.item())
                lp = float(log_prob_t.item())

                env.step(action)
                total_steps += 1

                # Reward assigned to learner for this transition
                r = float(env.rewards[curr_agent])
                done = bool(env.game_state.game_over)

                buffer.add(
                    obs=obs,
                    action=action,
                    mask=mask,
                    reward=r,
                    value=val,
                    log_prob=lp,
                    done=done
                )

                if buffer.is_full():
                    break
            else:
                # Opponent Turn (Seats 1, 2, 3)
                opp_model = opp_assignments[e_i][p_idx - 1][1]
                opp_act = sample_opponent_action(opp_model, obs, mask, device)
                env.step(opp_act)

        # 5. PPO Training Update when Rollout Buffer is Full
        if buffer.is_full():
            learner.eval()
            with torch.no_grad():
                dummy_obs = torch.as_tensor(obs[np.newaxis, :], dtype=torch.float32, device=device)
                last_value = float(learner.predict_values(dummy_obs).item())

            batches = buffer.get_batches(
                last_val=last_value,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device
            )

            learner.train()
            b_obs = batches["obs"]
            b_actions = batches["actions"]
            b_masks = batches["masks"]
            b_old_log_probs = batches["log_probs"]
            b_returns = batches["returns"]
            b_advantages = batches["advantages"]

            n_samples = len(b_obs)
            indices = np.arange(n_samples)

            total_pi_loss = 0.0
            total_vf_loss = 0.0
            total_ent = 0.0
            updates = 0

            for _ in range(n_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, n_samples, minibatch_size):
                    end_idx = min(start_idx + minibatch_size, n_samples)
                    mb_idx = indices[start_idx:end_idx]

                    mb_o = b_obs[mb_idx]
                    mb_a = b_actions[mb_idx]
                    mb_m = b_masks[mb_idx]
                    mb_olp = b_old_log_probs[mb_idx]
                    mb_ret = b_returns[mb_idx]
                    mb_adv = b_advantages[mb_idx]

                    values, log_probs, entropy = learner.evaluate_actions(
                        mb_o, mb_a, action_masks=mb_m
                    )

                    # PPO Clipped Objective
                    ratio = torch.exp(log_probs - mb_olp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    vf_loss = 0.5 * (values - mb_ret).pow(2).mean()

                    # Total Loss with Dynamic Entropy
                    ent_loss = -current_ent_coef * entropy.mean()
                    loss = pi_loss + 0.5 * vf_loss + ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(learner.parameters(), 0.5)
                    optimizer.step()

                    total_pi_loss += pi_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_ent += entropy.mean().item()
                    updates += 1

            # Update latest snapshot
            latest_snapshot = create_snapshot(f"snap_{total_steps}")

            # Logging Metrics
            elapsed = time.time() - start_time
            sps = total_steps / max(1.0, elapsed)
            learner_wins = recent_winners.count(0)
            total_rec = max(1, len(recent_winners))
            learner_wr = (learner_wins / total_rec) * 100.0

            print(
                f"[{elapsed:6.1f}s | {total_steps:7,}/{total_timesteps:,} steps | {sps:4.0f} step/s | {total_games:5,} games] "
                f"Learner WR: {learner_wr:4.1f}% | Elo: {elo_tracker.learner_elo:4.0f} | "
                f"Pi Loss: {total_pi_loss/max(1,updates):+6.4f} | Ent: {total_ent/max(1,updates):5.3f}"
            )

        # 6. Periodic Checkpointing & Historical Pool Addition
        if total_steps - last_saved_step >= save_freq:
            last_saved_step = total_steps
            snap_path = os.path.join(league_dir, f"snapshot_{total_steps // 1000}k.pt")
            learner.save(snap_path)
            # Add to historical pool (max 15 models to conserve VRAM)
            hist_model = create_snapshot(f"step_{total_steps // 1000}k")
            historical_pool.append((f"step_{total_steps // 1000}k", hist_model))
            if len(historical_pool) > 15:
                historical_pool.pop(0)

            print(f">>> Snapshot tersimpan di {snap_path} (Historical Pool Size: {len(historical_pool)})")

    total_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"Self-Play League Selesai dalam {total_time:.2f} detik ({total_steps / total_time:.0f} step/s)!")
    print(f"Total Permainan: {total_games:,} game | Final Learner Elo: {elo_tracker.learner_elo:.0f}")
    print("=" * 75)

    # Save final grandmaster model
    final_model_path = os.path.join(checkpoints_dir, "grandmaster_model.pt")
    learner.save(final_model_path)
    print(f"\nModel Grandmaster PyTorch [512, 512, 256] ELU tersimpan di: {final_model_path}")

    # Synchronize all checkpoint aliases for tournament compatibility
    print("\nMenyinkronkan model turnamen dengan arsitektur Grandmaster baru:")
    for name in ["curt", "albert", "sullivan", "bernstein", "final"]:
        target_pt = os.path.join(checkpoints_dir, f"{name}_model.pt")
        learner.save(target_pt)
        print(f"  -> {target_pt}")

    return learner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grandmaster Maskable PPO Self-Play League Training")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total learner timesteps (default: 1,000,000)")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel games (default: 32)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Rollout batch size (default: 2048)")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per PPO update (default: 10)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor (default: 0.995)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--entropy-start", type=float, default=0.05, help="Initial entropy coef (default: 0.05)")
    parser.add_argument("--entropy-end", type=float, default=0.001, help="Final entropy coef (default: 0.001)")
    parser.add_argument("--save-freq", type=int, default=200000, help="Snapshot save frequency (default: 200,000)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train_self_play_league(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        save_freq=args.save_freq,
        checkpoints_dir=args.checkpoints_dir,
        device_str=args.device,
        seed=args.seed
    )
