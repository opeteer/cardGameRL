#!/usr/bin/env python3
"""
Simultaneous Multi-Agent PPO Training Pipeline for Capsa Banting (Big Two).
Trains 4 neural network policies (Curt, Albert, Sullivan, Bernstein)
playing the game together using MaskableActorCritic [512, 512, 256] ELU LayerNorm,
Potential-Based Reward Shaping (PBRS), Anti-Feeding Constraints, and Dynamic Elo Tracking.
"""
import os
import sys
import time
import random
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from big_two_rl.env.big_two_env import raw_env
from big_two_rl.env.actions import NUM_ACTIONS
from big_two_rl.env.observation import OBS_DIM
from big_two_rl.models.maskable_actor_critic import MaskableActorCritic

AGENT_NAMES = ["Curt", "Albert", "Sullivan", "Bernstein"]


class MultiAgentEloTracker:
    """Tracks dynamic Elo ratings between all 4 playing agents."""
    def __init__(self, initial_elo: float = 1200.0, k_factor: float = 32.0):
        self.k_factor = k_factor
        self.ratings: List[float] = [initial_elo] * 4

    def record_game(self, winner_seat: int):
        # Update pairwise against other 3 players
        delta_elo = [0.0] * 4
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                exp_i = 1.0 / (1.0 + 10.0 ** ((self.ratings[j] - self.ratings[i]) / 400.0))
                if winner_seat == i:
                    actual_i = 1.0
                elif winner_seat == j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5  # Neither won
                delta_elo[i] += (self.k_factor / 3.0) * (actual_i - exp_i)

        for i in range(4):
            self.ratings[i] += delta_elo[i]


class MultiAgentTrajectoryBuffer:
    """
    Trajectory tracker for an individual player across multiple parallel environments.
    Computes Generalized Advantage Estimation (GAE) per episode trajectory before batching.
    """
    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        # env_id -> list of transition dicts for active episode
        self.active_trajectories: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # Completed transitions ready for batch extraction
        self.ready_transitions: List[Dict[str, Any]] = []

    def add_step(
        self,
        env_id: int,
        obs: np.ndarray,
        action: int,
        mask: np.ndarray,
        value: float,
        log_prob: float
    ):
        self.active_trajectories[env_id].append({
            "obs": obs,
            "action": action,
            "mask": mask,
            "value": value,
            "log_prob": log_prob,
            "reward": 0.0,
            "done": False,
        })

    def add_step_reward(self, env_id: int, reward: float):
        if self.active_trajectories[env_id]:
            self.active_trajectories[env_id][-1]["reward"] += reward

    def finish_trajectory(
        self,
        env_id: int,
        terminal_reward: float,
        gamma: float = 0.995,
        gae_lambda: float = 0.95
    ):
        traj = self.active_trajectories.pop(env_id, [])
        if not traj:
            return

        traj[-1]["reward"] += terminal_reward
        traj[-1]["done"] = True

        # Compute GAE within this player's episodic trajectory
        k = len(traj)
        advantages = np.zeros(k, dtype=np.float32)
        returns = np.zeros(k, dtype=np.float32)
        last_gae = 0.0
        next_val = 0.0

        for i in reversed(range(k)):
            r = traj[i]["reward"]
            v = traj[i]["value"]
            done = traj[i]["done"]
            delta = r + gamma * next_val * (1.0 - float(done)) - v
            last_gae = delta + gamma * gae_lambda * (1.0 - float(done)) * last_gae
            advantages[i] = last_gae
            returns[i] = last_gae + v
            next_val = v

        for i in range(k):
            traj[i]["advantage"] = advantages[i]
            traj[i]["return"] = returns[i]

        self.ready_transitions.extend(traj)

    def size(self) -> int:
        return len(self.ready_transitions)

    def extract_ready_batch(self, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.ready_transitions) < 128:
            return None

        transitions = self.ready_transitions
        self.ready_transitions = []

        obs_arr = np.array([t["obs"] for t in transitions], dtype=np.float32)
        act_arr = np.array([t["action"] for t in transitions], dtype=np.int64)
        mask_arr = np.array([t["mask"] for t in transitions], dtype=bool)
        lp_arr = np.array([t["log_prob"] for t in transitions], dtype=np.float32)
        ret_arr = np.array([t["return"] for t in transitions], dtype=np.float32)
        adv_arr = np.array([t["advantage"] for t in transitions], dtype=np.float32)

        # Normalize advantages
        adv_mean = adv_arr.mean()
        adv_std = adv_arr.std()
        adv_norm = (adv_arr - adv_mean) / (adv_std + 1e-8)

        return {
            "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(act_arr, dtype=torch.int64, device=device),
            "masks": torch.as_tensor(mask_arr, dtype=torch.bool, device=device),
            "log_probs": torch.as_tensor(lp_arr, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(ret_arr, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(adv_norm, dtype=torch.float32, device=device),
        }


def ppo_update_agent(
    policy: MaskableActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_epochs: int = 4,
    minibatch_size: int = 256,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Perform PPO policy & value function update for one agent over minibatches.
    """
    b_obs = batch["obs"]
    b_actions = batch["actions"]
    b_masks = batch["masks"]
    b_old_log_probs = batch["log_probs"]
    b_returns = batch["returns"]
    b_advantages = batch["advantages"]

    n_samples = len(b_obs)
    indices = np.arange(n_samples)

    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    num_updates = 0

    policy.train()
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, n_samples, minibatch_size):
            end_idx = min(start_idx + minibatch_size, n_samples)
            mb_idx = indices[start_idx:end_idx]

            mb_obs = b_obs[mb_idx]
            mb_actions = b_actions[mb_idx]
            mb_masks = b_masks[mb_idx]
            mb_old_log_probs = b_old_log_probs[mb_idx]
            mb_returns = b_returns[mb_idx]
            mb_advantages = b_advantages[mb_idx]

            values, log_probs, entropy = policy.evaluate_actions(
                mb_obs, mb_actions, action_masks=mb_masks
            )

            # PPO Clipped Surrogate Objective
            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value Function Loss
            vf_loss = 0.5 * (values - mb_returns).pow(2).mean()

            # Entropy Regularization
            ent_loss = -ent_coef * entropy.mean()

            loss = pi_loss + vf_coef * vf_loss + ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pi_loss += pi_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent += entropy.mean().item()
            num_updates += 1

    if num_updates > 0:
        return total_pi_loss / num_updates, total_vf_loss / num_updates, total_ent / num_updates
    return 0.0, 0.0, 0.0


def evaluate_agents(
    policies: List[MaskableActorCritic],
    num_games: int = 100,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[float], List[float]]:
    """
    Evaluate the 4 trained policies playing against each other in round-robin evaluation.
    """
    env = raw_env(use_pbrs_reward=False, enforce_anti_feed=True)
    wins = [0] * 4
    total_payoffs = [0.0] * 4

    for p in range(4):
        policies[p].eval()

    for g in range(num_games):
        env.reset(seed=10000 + g)
        while not env.game_state.game_over:
            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            obs = env.observe(curr_agent)
            mask = env.action_mask(curr_agent, as_int8=False)

            obs_t = torch.as_tensor(obs[np.newaxis, :], dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask[np.newaxis, :], dtype=torch.bool, device=device)

            with torch.no_grad():
                act_t, _, _, _ = policies[p_idx].get_action(
                    obs_t, action_masks=mask_t, deterministic=True
                )
                action = int(act_t.item())

            if not mask[action]:
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else 0

            env.step(action)

        w = env.game_state.winner
        if w is not None:
            wins[w] += 1
        payoffs = env.game_state.get_payoffs()
        for p in range(4):
            total_payoffs[p] += payoffs[p]

    win_rates = [w / num_games for w in wins]
    avg_payoffs = [p / num_games for p in total_payoffs]
    return win_rates, avg_payoffs


def train_multi_agent(
    total_timesteps: int = 1000000,
    num_envs: int = 32,
    rollout_steps: int = 2048,
    minibatch_size: int = 256,
    n_epochs: int = 4,
    lr: float = 2.5e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    entropy_start: float = 0.05,
    entropy_end: float = 0.001,
    save_freq: int = 200000,
    checkpoints_dir: str = "checkpoints",
    device_str: str = "auto",
    eval_games: int = 100,
    seed: int = 42,
    warm_start: bool = True,
):
    print("=" * 75)
    print("PELATIHAN 4 AGEN SECARA SIMULTAN (MULTI-AGENT PPO BIG TWO)")
    print("=" * 75)
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Parallel Games:      {num_envs}")
    print(f"Rollout Steps/Agent: {rollout_steps}")
    print(f"Mini-batch Size:     {minibatch_size}")
    print(f"Epochs per Update:   {n_epochs}")
    print(f"Learning Rate:       {lr}")
    print(f"Discount (gamma):    {gamma} (PBRS horizon preservation)")
    print(f"Entropy Annealing:   {entropy_start} -> {entropy_end}")
    print(f"Checkpoints Dir:     {checkpoints_dir}")

    # Configure compute device
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Compute Device:      {device} ({device_name})")
    print(f"4 Agen:              Curt (0), Albert (1), Sullivan (2), Bernstein (3)")
    print("=" * 75)

    os.makedirs(checkpoints_dir, exist_ok=True)
    multi_agent_dir = os.path.join(checkpoints_dir, "multi_agent")
    os.makedirs(multi_agent_dir, exist_ok=True)

    # 1. Instantiate 4 separate MaskableActorCritic policies
    policies: List[MaskableActorCritic] = []
    optimizers: List[torch.optim.Optimizer] = []
    buffers: List[MultiAgentTrajectoryBuffer] = []
    elo_tracker = MultiAgentEloTracker()

    for p in range(4):
        name = AGENT_NAMES[p]
        policy = MaskableActorCritic(obs_dim=OBS_DIM, action_dim=NUM_ACTIONS).to(device)

        # Warm start from existing checkpoint if available
        loaded = False
        if warm_start:
            p_path = os.path.join(checkpoints_dir, f"{name.lower()}_model.pt")
            if os.path.exists(p_path):
                try:
                    ckpt = torch.load(p_path, map_location=device)
                    policy.load_state_dict_flexible(ckpt["state_dict"])
                    print(f"  -> Agen {name} diinisialisasi (transfer bobot) dari: {p_path}")
                    loaded = True
                except Exception as e:
                    print(f"  -> Gagal memuat {p_path}: {e}")

            if not loaded:
                gm_path = os.path.join(checkpoints_dir, "grandmaster_model.pt")
                if os.path.exists(gm_path):
                    try:
                        ckpt = torch.load(gm_path, map_location=device)
                        policy.load_state_dict_flexible(ckpt["state_dict"])
                        print(f"  -> Agen {name} diinisialisasi (transfer bobot) dari base: {gm_path}")
                        loaded = True
                    except Exception:
                        pass

        if not loaded:
            print(f"  -> Agen {name} diinisialisasi dari bobot acak (orthogonal init).")

        optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        policies.append(policy)
        optimizers.append(optimizer)
        buffers.append(MultiAgentTrajectoryBuffer(p))

    # 2. Initialize Parallel Environments
    envs = [raw_env(use_pbrs_reward=True, enforce_anti_feed=True) for _ in range(num_envs)]
    for e_idx, env in enumerate(envs):
        env.reset(seed=seed + e_idx * 31)

    # Metrics Tracking
    total_steps = 0
    total_games = 0
    last_saved_step = 0
    start_time = time.time()
    recent_winners = deque(maxlen=300)
    recent_payoffs = [deque(maxlen=300) for _ in range(4)]

    print(f"\nMemulai simulasi permainan bersama 4 agen di {device_name}...")

    # Main Environment Simulation Loop
    while total_steps < total_timesteps:
        # Linear dynamic entropy annealing
        progress = float(total_steps) / float(total_timesteps)
        current_ent_coef = entropy_start - progress * (entropy_start - entropy_end)

        # 1. Group active envs by current agent for batched inference
        groups = defaultdict(list)
        for e_i, env in enumerate(envs):
            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            obs = env.observe(curr_agent)
            mask = env.action_mask(curr_agent, as_int8=False)
            groups[p_idx].append((e_i, curr_agent, obs, mask))

        # 2. Batched inference per player group
        for p_idx, items in groups.items():
            if not items:
                continue

            e_indices = [x[0] for x in items]
            obs_batch = np.array([x[2] for x in items], dtype=np.float32)
            mask_batch = np.array([x[3] for x in items], dtype=bool)

            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)

            policies[p_idx].eval()
            with torch.no_grad():
                act_t, val_t, lp_t, _ = policies[p_idx].get_action(
                    obs_t, action_masks=mask_t, deterministic=False
                )

            actions = act_t.cpu().numpy()
            values = val_t.cpu().numpy()
            log_probs = lp_t.cpu().numpy()

            for i, e_i in enumerate(e_indices):
                act = int(actions[i])
                val = float(values[i])
                lp = float(log_probs[i])
                env = envs[e_i]
                curr_agent = items[i][1]

                # Action mask safety guard
                if not mask_batch[i][act]:
                    legal = np.where(mask_batch[i])[0]
                    act = int(np.random.choice(legal)) if len(legal) > 0 else 0

                # Record transition before step
                buffers[p_idx].add_step(
                    env_id=e_i,
                    obs=obs_batch[i],
                    action=act,
                    mask=mask_batch[i],
                    value=val,
                    log_prob=lp
                )

                # Step environment
                env.step(act)
                total_steps += 1

                # Check game completion
                if env.game_state.game_over:
                    total_games += 1
                    winner = env.game_state.winner
                    if winner is not None:
                        recent_winners.append(winner)
                        elo_tracker.record_game(winner)

                    # Finish episodic trajectories for all 4 agents
                    for p in range(4):
                        agent_key = f"player_{p}"
                        term_r = float(env.rewards[agent_key])
                        recent_payoffs[p].append(term_r)
                        buffers[p].finish_trajectory(
                            env_id=e_i,
                            terminal_reward=term_r,
                            gamma=gamma,
                            gae_lambda=gae_lambda
                        )

                    # Reset finished game environment
                    env.reset(seed=seed + total_games * 29 + e_i)
                else:
                    # Intermediate step: assign step reward (PBRS / tactical penalty)
                    step_r = float(env.rewards[curr_agent])
                    buffers[p_idx].add_step_reward(env_id=e_i, reward=step_r)

                if total_steps >= total_timesteps:
                    break

            if total_steps >= total_timesteps:
                break

        # 3. PPO Update when all 4 agents have accumulated enough transitions
        min_buf = min(buffers[p].size() for p in range(4))
        if min_buf >= rollout_steps:
            loss_stats = []
            for p in range(4):
                batch = buffers[p].extract_ready_batch(device=device)
                if batch is not None:
                    pi_l, vf_l, ent = ppo_update_agent(
                        policy=policies[p],
                        optimizer=optimizers[p],
                        batch=batch,
                        device=device,
                        n_epochs=n_epochs,
                        minibatch_size=minibatch_size,
                        clip_range=clip_range,
                        ent_coef=current_ent_coef,
                        vf_coef=0.5,
                        max_grad_norm=0.5
                    )
                    loss_stats.append((pi_l, vf_l, ent))

            # Logging metrics
            elapsed = time.time() - start_time
            sps = total_steps / max(1.0, elapsed)
            total_rec = max(1, len(recent_winners))
            wr = [recent_winners.count(p) / total_rec * 100.0 for p in range(4)]
            r_elo = elo_tracker.ratings

            print(
                f"[{elapsed:6.1f}s | {total_steps:7,}/{total_timesteps:,} steps | {sps:4.0f} step/s | {total_games:5,} games] "
                f"WR: Curt {wr[0]:4.1f}% | Alb {wr[1]:4.1f}% | Sul {wr[2]:4.1f}% | Ber {wr[3]:4.1f}% | "
                f"Elo: {r_elo[0]:4.0f}/{r_elo[1]:4.0f}/{r_elo[2]:4.0f}/{r_elo[3]:4.0f}"
            )

        # 4. Periodic Checkpoint Saving
        if total_steps - last_saved_step >= save_freq:
            last_saved_step = total_steps
            snap_tag = f"step_{total_steps // 1000}k"
            print(f"\n>>> Menyimpan Checkpoint Multi-Agent ({snap_tag})...")
            for p in range(4):
                agent_name = AGENT_NAMES[p].lower()
                snap_path = os.path.join(multi_agent_dir, f"{agent_name}_{snap_tag}.pt")
                policies[p].save(snap_path)
                print(f"  -> {AGENT_NAMES[p]}: {snap_path}")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"Pelatihan Multi-Agent Selesai dalam {elapsed_total:.2f} detik ({total_steps / elapsed_total:.0f} step/s)!")
    print(f"Total Pertandingan Dimainkan: {total_games:,}")
    print(f"Final Elo Ratings: Curt={elo_tracker.ratings[0]:.0f}, Albert={elo_tracker.ratings[1]:.0f}, "
          f"Sullivan={elo_tracker.ratings[2]:.0f}, Bernstein={elo_tracker.ratings[3]:.0f}")
    print("=" * 75)

    # 5. Save Final Models for All 4 Agents
    print("\nMenyimpan model final PyTorch [512, 512, 256] ELU untuk ke-4 agen:")
    final_paths = {}
    for p in range(4):
        agent_name = AGENT_NAMES[p].lower()
        out_pt = os.path.join(checkpoints_dir, f"{agent_name}_model.pt")
        policies[p].save(out_pt)
        final_paths[AGENT_NAMES[p]] = out_pt
        print(f"  - {AGENT_NAMES[p]}: {out_pt}")

    # Determine highest Elo agent as champion
    best_p = int(np.argmax(elo_tracker.ratings))
    best_name = AGENT_NAMES[best_p]
    gm_path = os.path.join(checkpoints_dir, "grandmaster_model.pt")
    final_m_path = os.path.join(checkpoints_dir, "final_model.pt")
    policies[best_p].save(gm_path)
    policies[best_p].save(final_m_path)
    print(f"\nModel Juara (Champion: {best_name}, Elo: {elo_tracker.ratings[best_p]:.0f}) disinkronkan ke:")
    print(f"  -> {gm_path}")
    print(f"  -> {final_m_path}")

    # 6. Final Round-Robin Tournament Evaluation
    if eval_games > 0:
        print("\n" + "-" * 75)
        print(f"EVALUASI TURNAMEN ROUND-ROBIN KE-4 AGEN ({eval_games} Pertandingan Mandiri)...")
        eval_wr, eval_pay = evaluate_agents(policies, num_games=eval_games, device=device)
        print(f"{'Peringkat':<10} | {'Nama Pemain':<15} | {'Win Rate':<12} | {'Rata-rata Skor/Game':<20}")
        print("-" * 65)

        rank_order = sorted(range(4), key=lambda i: (eval_wr[i], eval_pay[i]), reverse=True)
        for rank, p in enumerate(rank_order, 1):
            crown = "👑 " if rank == 1 else "   "
            print(f"{crown}#{rank:<5} | {AGENT_NAMES[p]:<15} | {eval_wr[p]*100:6.1f}%     | {eval_pay[p]:+6.2f} pt")
        print("-" * 75)

    return policies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent PPO Training for Big Two (Capsa Banting)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total environment timesteps (default: 1,000,000)")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments (default: 32)")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps per player before PPO update")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for PPO update")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per PPO update (default: 4)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor (default: 0.995)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--entropy-start", type=float, default=0.05, help="Initial entropy coef (default: 0.05)")
    parser.add_argument("--entropy-end", type=float, default=0.001, help="Final entropy coef (default: 0.001)")
    parser.add_argument("--save-freq", type=int, default=200000, help="Checkpoint frequency (default: 200,000)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--eval-games", type=int, default=100, help="Evaluation games count (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-warm-start", action="store_true", help="Do not load pre-existing checkpoint weights")
    args = parser.parse_args()

    train_multi_agent(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
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
        eval_games=args.eval_games,
        seed=args.seed,
        warm_start=not args.no_warm_start
    )
