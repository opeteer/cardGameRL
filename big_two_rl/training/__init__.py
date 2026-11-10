"""
Training package for Big Two (Capsa Banting) using MaskablePPO and Self-Play.
"""
from big_two_rl.training.self_play import SelfPlayEnv, SelfPlayCallback

__all__ = ["SelfPlayEnv", "SelfPlayCallback"]
