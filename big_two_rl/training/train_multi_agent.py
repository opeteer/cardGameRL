#!/usr/bin/env python3
"""
Simultaneous Multi-Agent PPO Training Pipeline for Big Two (Capsa Banting).
Trains 4 separate neural network policies (Curt, Albert, Sullivan, Bernstein)
concurrently on NVIDIA GPU using Maskable Actor-Critic Policy and Invalid Action Masking.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO

from big_two_rl.env.big_two_env import raw_env
from big_two_rl.env.actions import NUM_ACTIONS
from big_two_rl.env.observation import OBS_DIM
from big_two_rl.engine.card import card_to_str


AGENT_NAMES = ["Curt", "Albert", "Sullivan", "Bernstein"]


class TrajectoryBuffer:
    """
    Trajectory tracker for an individual player across multiple parallel environments.
    Computes Generalized Advantage Estimation (GAE) per episode trajectory before flattening.
    """
    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        # env_id -> list of transition dicts
        self.active_trajectories: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.ready_transitions: List[Dict[str, Any]] = []

    def add_step(
        self,
        env_id: int,
        obs: np.ndarray,
        action: int,
        value: float,
        log_prob: float,
        mask: np.ndarray
    ):
        self.active_trajectories[env_id].append({
            "obs": obs,
            "action": action,
            "value": value,
            "log_prob": log_prob,
            "mask": mask,
            "reward": 0.0,
            "done": False,
        })

    def add_reward(self, env_id: int, reward: float):
        if self.active_trajectories[env_id]:
            self.active_trajectories[env_id][-1]["reward"] += reward

    def finish_trajectory(
        self,
        env_id: int,
        terminal_reward: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        traj = self.active_trajectories.pop(env_id, [])
        if not traj:
            return

        traj[-1]["reward"] += terminal_reward
        traj[-1]["done"] = True

        # Compute GAE within this trajectory
        k = len(traj)
        advantages = [0.0] * k
        returns = [0.0] * k
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

    def extract_ready_batch(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.ready_transitions:
            return None

        transitions = self.ready_transitions
        self.ready_transitions = []

        return {
            "obs": np.array([t["obs"] for t in transitions], dtype=np.float32),
            "actions": np.array([t["action"] for t in transitions], dtype=np.int64),
            "masks": np.array([t["mask"] for t in transitions], dtype=bool),
            "log_probs": np.array([t["log_prob"] for t in transitions], dtype=np.float32),
            "values": np.array([t["value"] for t in transitions], dtype=np.float32),
            "advantages": np.array([t["advantage"] for t in transitions], dtype=np.float32),
            "returns": np.array([t["return"] for t in transitions], dtype=np.float32),
        }


def ppo_update_agent(
    policy: MaskableActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    device: torch.device,
    n_epochs: int = 4,
    batch_size: int = 256,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Perform PPO policy & value function update for one agent over minibatches.
    """
    obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    masks_t = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
    old_log_probs_t = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)

    # Normalize advantages
    adv_mean = advantages_t.mean()
    adv_std = advantages_t.std()
    advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)

    total_samples = len(obs_t)
    indices = np.arange(total_samples)

    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    num_updates = 0

    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            mb_idx = indices[start_idx:end_idx]

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_masks = masks_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]

            # Forward pass through policy
            dist = policy.get_distribution(mb_obs, action_masks=mb_masks)
            log_prob = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            values = policy.predict_values(mb_obs).flatten()

            # PPO Clipped Surrogate Loss
            ratio = torch.exp(log_prob - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            value_loss = 0.5 * (values - mb_returns).pow(2).mean()

            # Total Loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pi_loss += policy_loss.item()
            total_vf_loss += value_loss.item()
            total_ent += entropy.item()
            num_updates += 1

    if num_updates > 0:
        return total_pi_loss / num_updates, total_vf_loss / num_updates, total_ent / num_updates
    return 0.0, 0.0, 0.0


def evaluate_agents(
    policies: List[MaskableActorCriticPolicy],
    num_games: int = 50,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[float], List[float]]:
    """
    Evaluate the 4 trained policies playing against each other in round-robin evaluation.
    """
    env = raw_env()
    wins = [0] * 4
    total_payoffs = [0.0] * 4

    for g in range(num_games):
        env.reset(seed=5000 + g)
        while not env.game_state.game_over:
            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            obs = env.observe(curr_agent)
            mask = env.action_mask(curr_agent, as_int8=False)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                dist = policies[p_idx].get_distribution(obs_t, action_masks=mask_t)
                # Deterministic action for evaluation
                action = int(torch.argmax(dist.distribution.logits).item())
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
    checkpoints_dir: str = "checkpoints",
    learning_rate: float = 3e-4,
    device_str: str = "cuda",
    warm_start_path: Optional[str] = "checkpoints/final_model.zip",
    eval_games: int = 50,
    save_freq: int = 100000,
    seed: int = 42
):
    print("=" * 70)
    print("Multi-Agent Simultaneous PPO Training Pipeline (Big Two / Capsa Banting)")
    print("=" * 70)
    print(f"Target Timesteps:  {total_timesteps:,}")
    print(f"Parallel Envs:     {num_envs}")
    print(f"Rollout Steps:     {rollout_steps}")
    print(f"Save Frequency:    {save_freq:,}")
    print(f"Checkpoints Dir:   {checkpoints_dir}")

    # Configure compute device
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"Compute Device:    {device} ({device_name})")
    print(f"Agents (4):        Curt (0), Albert (1), Sullivan (2), Bernstein (3)")
    print("=" * 70)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoints_dir, "multi_agent"), exist_ok=True)

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Discrete(NUM_ACTIONS)

    # 1. Instantiate 4 separate MaskableActorCriticPolicy models
    policies: List[MaskableActorCriticPolicy] = []
    optimizers: List[torch.optim.Optimizer] = []
    buffers: List[TrajectoryBuffer] = []

    # Optional warm start template
    warm_template = None
    if warm_start_path and os.path.exists(warm_start_path):
        print(f"\n[Warm Start] Loading pre-trained base model weights from: {warm_start_path}")
        warm_template = MaskablePPO.load(warm_start_path, device="cpu")

    for p in range(4):
        policy = MaskableActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: learning_rate,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU
        ).to(device)

        if warm_template is not None:
            policy.load_state_dict(warm_template.policy.state_dict())
            print(f"  -> Agent {AGENT_NAMES[p]} ({p}) diinisialisasi dari pre-trained checkpoint.")

        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        policies.append(policy)
        optimizers.append(optimizer)
        buffers.append(TrajectoryBuffer(p))

    # 2. Initialize Vectorized Parallel Environments
    envs = [raw_env() for _ in range(num_envs)]
    for e_idx, env in enumerate(envs):
        env.reset(seed=seed + e_idx * 17)

    # Metrics Tracking
    total_steps = 0
    total_games = 0
    last_saved_step = 0
    start_time = time.time()
    last_log_time = start_time
    recent_winners = deque(maxlen=200)
    recent_payoffs = [deque(maxlen=200) for _ in range(4)]

    print(f"\nMemulai simulasi permainan paralel & pelatihan 4 agen di {device_name}...")

    # Main Multi-Agent Environment Step Loop
    while total_steps < total_timesteps:
        # Step through environments in batched player groups
        groups = defaultdict(list)
        for e_idx, env in enumerate(envs):
            if env.game_state.game_over:
                # Record game completion & reset
                w = env.game_state.winner
                if w is not None:
                    recent_winners.append(w)
                payoffs = env.game_state.get_payoffs()
                for p in range(4):
                    recent_payoffs[p].append(payoffs[p])
                    buffers[p].finish_trajectory(e_idx, terminal_reward=float(payoffs[p]))
                total_games += 1
                env.reset(seed=seed + total_games * 19 + e_idx)

            curr_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[curr_agent]
            obs = env.observe(curr_agent)
            mask = env.action_mask(curr_agent, as_int8=False)
            groups[p_idx].append((e_idx, obs, mask))

        # Batched inference per player group on GPU
        for p_idx, items in groups.items():
            if not items:
                continue

            e_indices = [x[0] for x in items]
            obs_batch = np.array([x[1] for x in items], dtype=np.float32)
            mask_batch = np.array([x[2] for x in items], dtype=bool)

            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)

            with torch.no_grad():
                dist = policies[p_idx].get_distribution(obs_t, action_masks=mask_t)
                actions = dist.sample()
                values = policies[p_idx].predict_values(obs_t).flatten()
                log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            # Execute actions in environments and log transitions
            for i, e_i in enumerate(e_indices):
                act = int(actions_np[i])
                buffers[p_idx].add_step(
                    env_id=e_i,
                    obs=obs_batch[i],
                    action=act,
                    value=float(values_np[i]),
                    log_prob=float(log_probs_np[i]),
                    mask=mask_batch[i]
                )

                # Prior cards count to compute shedding reward
                cards_before = len(envs[e_i].game_state.hands[p_idx])
                envs[e_i].step(act)
                cards_after = len(envs[e_i].game_state.hands[p_idx])

                # Dense card shedding bonus (+0.1 per card discarded)
                cards_played = cards_before - cards_after
                if cards_played > 0:
                    buffers[p_idx].add_reward(e_i, 0.1 * cards_played)

                total_steps += 1

        # Check if rollout buffer has enough ready transitions for training update
        min_buffer_size = min(buffers[p].size() for p in range(4))
        if min_buffer_size >= rollout_steps:
            # PPO Updates for all 4 agents
            loss_stats = []
            for p in range(4):
                batch = buffers[p].extract_ready_batch()
                if batch is not None and len(batch["obs"]) >= 128:
                    pi_loss, vf_loss, ent = ppo_update_agent(
                        policy=policies[p],
                        optimizer=optimizers[p],
                        batch=batch,
                        device=device,
                        n_epochs=4,
                        batch_size=256,
                        clip_range=0.2,
                        ent_coef=0.01,
                        vf_coef=0.5
                    )
                    loss_stats.append((pi_loss, vf_loss, ent))

            # Logging metrics
            curr_time = time.time()
            elapsed = curr_time - start_time
            sps = total_steps / max(1.0, elapsed)

            win_counts = [recent_winners.count(p) for p in range(4)]
            total_recent = max(1, len(recent_winners))
            win_rates = [w / total_recent for w in win_counts]
            avg_recent_payoffs = [
                np.mean(recent_payoffs[p]) if len(recent_payoffs[p]) > 0 else 0.0
                for p in range(4)
            ]

            print(
                f"[{elapsed:6.1f}s | {total_steps:7,}/{total_timesteps:,} steps | {sps:4.0f} step/s | {total_games:5,} games] "
                f"Win: Curt {win_rates[0]*100:4.1f}% | Albert {win_rates[1]*100:4.1f}% | "
                f"Sullivan {win_rates[2]*100:4.1f}% | Bernstein {win_rates[3]*100:4.1f}%"
            )

        # Periodic checkpoint save
        if total_steps - last_saved_step >= save_freq:
            last_saved_step = total_steps
            print(f"\n>>> Menyimpan Checkpoint Multi-Agent pada {total_steps:,} langkah...")
            save_multi_agent_models(
                policies=policies,
                save_dir=os.path.join(checkpoints_dir, "multi_agent"),
                step_suffix=f"step_{total_steps // 1000}k",
                template_path=warm_start_path
            )

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Multi-Agent Training Selesai dalam {elapsed_total:.2f} detik ({total_steps / elapsed_total:.0f} step/s)!")
    print(f"Total Games Dimainkan: {total_games:,}")
    print("=" * 70)

    # Save final distinct models for all 4 agents
    print("\nMenyimpan model final independen untuk ke-4 agen:")
    final_paths = save_multi_agent_models(
        policies=policies,
        save_dir=checkpoints_dir,
        step_suffix="model",
        template_path=warm_start_path
    )
    for name, pth in final_paths.items():
        print(f"  - {name}: {pth}")

    # Run Round-Robin Evaluation
    if eval_games > 0:
        print("\n" + "-" * 70)
        print(f"Evaluasi Round-Robin 4 Agen Mandiri ({eval_games} Game)...")
        eval_win_rates, eval_payoffs = evaluate_agents(
            policies=policies,
            num_games=eval_games,
            device=device
        )
        print("Hasil Evaluasi:")
        for p in range(4):
            print(
                f"  [{AGENT_NAMES[p]}] Win Rate: {eval_win_rates[p]*100:5.1f}% | "
                f"Avg Payoff: {eval_payoffs[p]:+6.2f} pt/game"
            )
        print("-" * 70)

    return policies


def save_multi_agent_models(
    policies: List[MaskableActorCriticPolicy],
    save_dir: str,
    step_suffix: str = "model",
    template_path: Optional[str] = "checkpoints/final_model.zip"
) -> Dict[str, str]:
    """
    Save 4 distinct agent models as SB3-compatible MaskablePPO zip files.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_map = {
        "Curt": f"curt_{step_suffix}.zip",
        "Albert": f"albert_{step_suffix}.zip",
        "Sullivan": f"sullivan_{step_suffix}.zip",
        "Bernstein": f"bernstein_{step_suffix}.zip",
    }
    saved_paths = {}

    # Load template to ensure exact SB3 metadata and structure
    if template_path and os.path.exists(template_path):
        template = MaskablePPO.load(template_path, device="cpu")
    else:
        # Fallback dummy env
        from big_two_rl.training.train_ppo import make_single_env
        template = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=make_single_env()(),
            device="cpu"
        )

    for p in range(4):
        name = AGENT_NAMES[p]
        filename = file_map[name]
        out_path = os.path.join(save_dir, filename)
        # Copy policy state dict to template (on CPU)
        template.policy.load_state_dict(policies[p].cpu().state_dict())
        template.save(out_path)
        saved_paths[name] = out_path
        # Move back to cuda if policy was on cuda
        if torch.cuda.is_available():
            policies[p].to("cuda")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simultaneous Multi-Agent PPO Training for Big Two")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total environment timesteps (default: 1,000,000)")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments (default: 32)")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps per player before PPO update")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda, cpu)")
    parser.add_argument("--warm-start", type=str, default="checkpoints/final_model.zip", help="Warm start model path")
    parser.add_argument("--eval-games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument("--save-freq", type=int, default=100000, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_multi_agent(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        checkpoints_dir=args.checkpoints_dir,
        learning_rate=args.lr,
        device_str=args.device,
        warm_start_path=args.warm_start if os.path.exists(args.warm_start) else None,
        eval_games=args.eval_games,
        save_freq=args.save_freq,
        seed=args.seed
    )
