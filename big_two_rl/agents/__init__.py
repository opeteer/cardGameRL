"""
Agent package for Big Two (Capsa Banting).
Semua 4 agen (Curt, Albert, Sullivan, Bernstein) menggunakan sistem PPO RL yang sepenuhnya identik.
Tidak ada pembedaan karakteristik atau aturan khusus antar agen.
"""
import os
from typing import List, Optional

from big_two_rl.agents.base_agent import BaseAgent
from big_two_rl.agents.rl_agent import RLAgent, CurtAgent
from big_two_rl.agents.aggressive_agent import AlbertAgent
from big_two_rl.agents.conservative_agent import SullivanAgent
from big_two_rl.agents.combo_agent import BernsteinAgent


def create_tournament_agents(
    model_path: Optional[str] = None,
    shared_model: bool = True
) -> List[BaseAgent]:
    """
    Instansiasi keempat agen menggunakan sistem PPO RL yang persis sama:
    Pemain 0: Curt [PPO RL]      -> checkpoints/curt_model.pt
    Pemain 1: Albert [PPO RL]    -> checkpoints/albert_model.pt
    Pemain 2: Sullivan [PPO RL]  -> checkpoints/sullivan_model.pt
    Pemain 3: Bernstein [PPO RL] -> checkpoints/bernstein_model.pt

    Keempat model merupakan duplikasi identik dari arsitektur model Grandmaster PPO.
    """
    names = ["Curt", "Albert", "Sullivan", "Bernstein"]
    dedicated_paths = {
        name: f"checkpoints/{name.lower()}_model.pt"
        for name in names
    }
    all_dedicated_exist = all(os.path.exists(p) for p in dedicated_paths.values())

    if all_dedicated_exist and (not shared_model or model_path is None):
        agents = []
        for name in names:
            p_path = dedicated_paths[name]
            agents.append(RLAgent(name=name, model_path=p_path, deterministic=False))
        print(">>> Berhasil memuat 4 model Grandmaster PPO untuk masing-masing agen (curt_model.pt, albert_model.pt, sullivan_model.pt, bernstein_model.pt).")
        return agents

    if model_path is None:
        if os.path.exists("checkpoints/grandmaster_model.pt"):
            model_path = "checkpoints/grandmaster_model.pt"
        elif os.path.exists("checkpoints/final_model.pt"):
            model_path = "checkpoints/final_model.pt"

    loaded_model = None
    if shared_model and model_path and os.path.exists(model_path):
        if model_path.endswith(".pt"):
            try:
                from big_two_rl.models import MaskableActorCritic
                loaded_model = MaskableActorCritic.load(model_path, device="cpu")
                print(f">>> Berhasil memuat model Grandmaster PPO [512,512,256] ELU dari: {model_path}")
            except Exception as e:
                print(f">>> Gagal memuat model PyTorch ({e})")
        else:
            try:
                from sb3_contrib import MaskablePPO
                loaded_model = MaskablePPO.load(model_path, device="cpu")
                print(f">>> Berhasil memuat model PPO bersama yang identik untuk keempat agen dari: {model_path}")
            except Exception as e:
                print(f">>> Gagal memuat model PPO bersama ({e})")

    return [
        RLAgent(
            name=names[i],
            model_path=model_path,
            model=loaded_model,
            deterministic=False
        )
        for i in range(4)
    ]


__all__ = [
    "BaseAgent",
    "RLAgent",
    "CurtAgent",
    "AlbertAgent",
    "SullivanAgent",
    "BernsteinAgent",
    "create_tournament_agents",
]
