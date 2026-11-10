#!/usr/bin/env python3
"""
Root entrypoint for Simultaneous Multi-Agent Training of 4 agents playing together.
"""
from big_two_rl.training.train_multi_agent import train_multi_agent
import sys
import argparse

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
