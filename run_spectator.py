#!/usr/bin/env python3
"""
Entrypoint script for visualizing Big Two (Capsa Banting) Tournament.
Features custom player names: Curt, Albert, Sullivan, Bernstein.
Displays live on-screen Scoreboard, slow pacing, crowns the champion,
and supports dynamic seat reshuffling (pengacakan tempat duduk) every N rounds.
"""
import os
import sys
import time
import argparse
import numpy as np
from typing import Optional, List, Tuple

from big_two_rl.engine.card import card_to_str, cards_to_str
from big_two_rl.engine.evaluator import evaluate_hand
from big_two_rl.env.actions import action_to_cards
from big_two_rl.env.big_two_env import raw_env
from big_two_rl.gui.spectator import BigTwoSpectator, DEFAULT_PLAYER_NAMES
from big_two_rl.agents import create_tournament_agents


def compute_fair_tournament_scores(hands, winner_idx):
    """
    Sistem Skor Khusus Turnamen yang Adil (Fair Tournament Scoring):
    - Penentuan Peringkat Berdasarkan Sisa Kartu:
      * Posisi 1 (Pemenang / 0 kartu tersisa) : +10 Poin Placement
      * Posisi 2 (Sisa kartu tersedikit ke-2)  : +5 Poin Placement
      * Posisi 3 (Sisa kartu tersedikit ke-3)  : +2 Poin Placement
      * Posisi 4 (Sisa kartu terbanyak)        : 0 Poin Placement
    - Bonus Efisiensi Kartu (Cards Cleared):
      * Setiap pemain mendapat +1 Poin per kartu yang berhasil dibuang: (13 - len(hand))
    
    Total Poin Game = Poin Posisi + (13 - Sisa Kartu)
    (Rentang poin tiap game: 0 s/d 23 poin, tidak ada penalti negatif yang merusak turnamen)
    """
    card_counts = [len(h) for h in hands]
    # Urutkan berdasarkan sisa kartu (pemenang dengan 0 kartu pasti nomor 1)
    ranking = sorted(range(4), key=lambda i: (0 if i == winner_idx else 1, card_counts[i]))
    
    placement_table = [10, 5, 2, 0]
    round_points = [0, 0, 0, 0]
    for rank_pos, p in enumerate(ranking):
        placement_pts = placement_table[rank_pos]
        cards_shed = 13 - card_counts[p]
        round_points[p] = placement_pts + cards_shed
        
    return round_points, card_counts


def print_scoreboard(player_names, total_wins, total_scores, current_game, total_games):
    """Print ASCII scoreboard to stdout."""
    print("\n" + "=" * 65)
    print(f"🏆 PAPAN SKOR TURNAMEN (Game {current_game}/{total_games})")
    print("=" * 65)
    print(f"{'Peringkat':<10} | {'Nama Pemain':<15} | {'Menang':<8} | {'Poin Turnamen':<15}")
    print("-" * 65)

    # Sort players by total_scores descending, then total_wins descending
    ranking = sorted(range(4), key=lambda i: (total_scores[i], total_wins[i]), reverse=True)
    for rank, p in enumerate(ranking, 1):
        crown = "👑 " if rank == 1 and total_scores[p] > 0 else "   "
        print(f"{crown}#{rank:<5} | {player_names[p] + ' [RL]':<18} | {total_wins[p]:<8} | {total_scores[p]} pt")
    print("=" * 65 + "\n")


def run_spectator_simulation(
    num_games: int = 10,
    fps: int = 2,
    step_delay: float = 5.0,
    model_path: str = "checkpoints/final_model.zip",
    headless: bool = False,
    screenshot_dir: str = None,
    player_names: list = None,
    seed: Optional[int] = None,
    shuffle_freq: int = 5
):
    canonical_names = list(player_names) if player_names else list(DEFAULT_PLAYER_NAMES)

    print("=" * 65)
    print("TURNAMEN CAPSA BANTING (BIG TWO) - 4 AGENT BATTLE (ALL PPO RL)")
    print("=" * 65)
    for i, name in enumerate(canonical_names):
        print(f"Agen {i} ({name:<9}): PPO Reinforcement Learning Policy")
    print("-" * 65)
    print(f"Jumlah Pertandingan : {num_games} Game")
    print(f"Jeda Setiap Langkah : {step_delay:.2f} detik")
    print(f"Acak Tempat Duduk   : Setiap {shuffle_freq} Ronde" if shuffle_freq > 0 else "Acak Tempat Duduk   : Nonaktif (Kursi Tetap)")
    print(f"Mode Headless       : {headless}")
    print("=" * 65)

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Instantiate all 4 tournament agents using identical PPO RL system
    all_agents = create_tournament_agents(model_path=model_path, shared_model=False)
    print(">>> Seluruh 4 agen menggunakan sistem PPO RL yang sepenuhnya identik!\n")

    spectator = BigTwoSpectator(title="Turnamen Capsa Banting - Seat Rotation Edition")
    spectator.fps = max(1, fps)

    canonical_wins = [0, 0, 0, 0]
    canonical_scores = [0, 0, 0, 0]
    current_seats = [0, 1, 2, 3]  # Map: seat index (0..3) -> canonical agent index (0..3)
    rng = np.random.default_rng(seed if seed is not None else 42)

    for game_idx in range(num_games):
        current_game_num = game_idx + 1

        # Cek pengacakan tempat duduk setiap shuffle_freq ronde
        is_shuffle_round = (shuffle_freq > 0 and game_idx > 0 and game_idx % shuffle_freq == 0)
        shuffle_banner = None
        if is_shuffle_round:
            prev_seats = list(current_seats)
            for _ in range(20):
                new_perm = list(rng.permutation(4))
                if new_perm != prev_seats:
                    current_seats = new_perm
                    break
            else:
                current_seats = [(s + 1) % 4 for s in prev_seats]

            shuffle_banner = f"🔄 KURSI DIACAK: South={canonical_names[current_seats[0]]} | West={canonical_names[current_seats[1]]} | North={canonical_names[current_seats[2]]} | East={canonical_names[current_seats[3]]}"
            print("\n" + "🔄 " + "=" * 61 + " 🔄")
            print(f"🔄 PENGACAKAN TEMPAT DUDUK (SEAT RESHUFFLE - TIAP {shuffle_freq} RONDE) 🔄")
            print("🔄 " + "=" * 61 + " 🔄")
            print(f"  • Kursi 0 (South): {canonical_names[current_seats[0]]}")
            print(f"  • Kursi 1 (West) : {canonical_names[current_seats[1]]}")
            print(f"  • Kursi 2 (North): {canonical_names[current_seats[2]]}")
            print(f"  • Kursi 3 (East) : {canonical_names[current_seats[3]]}")
            print("-" * 65)

        # Mapping nama dan agen pada masing-masing kursi ronde ini
        seat_names = [canonical_names[current_seats[s]] for s in range(4)]
        seat_agents = [all_agents[current_seats[s]] for s in range(4)]

        env = raw_env()
        game_seed = (seed + game_idx) if seed is not None else None
        env.reset(seed=game_seed)

        print(f"\n{'#' * 65}")
        print(f"### MEMULAI GAME {current_game_num}/{num_games} ###")
        print(f"Susunan Meja: South={seat_names[0]} | West={seat_names[1]} | North={seat_names[2]} | East={seat_names[3]}")
        print(f"{'#' * 65}")
        gs = env.game_state
        if hasattr(gs, 'last_seed') and gs.last_seed is not None:
            print(f"🔒 Hardware Seed : 0x{gs.last_seed:016x} (SHA-256 Distilled)")
            if getattr(gs, 'entropy_info', None):
                info = gs.entropy_info
                print(f"  • RDRAND       : 0x{info['rdrand']:016x} (Hardware DRNG)")
                print(f"  • RDSEED       : 0x{info['rdseed']:016x} (Thermal Quantum Noise)")
                print(f"  • CPU Jitter   : 0x{info['cpu_jitter']:016x} (RDTSCP Execution Delta)")
                print(f"  • Mem Latency  : 0x{info['memory_latency']:016x} (16MB DRAM Bus Stride)")
                print(f"  • Cache Miss   : 0x{info['cache_miss']:016x} (clflush Hit/Miss Delta)")
        start_p = env.game_state.current_player
        print(f"Kartu 3♦ dipegang oleh: {seat_names[start_p]} (Kursi {start_p}) [Wajib membuka putaran]")
        print("-" * 65)

        last_action_text = f"{seat_names[start_p]} wajib membuka dengan 3♦"
        spectator.render_state(
            env.game_state,
            player_names=seat_names,
            wins=canonical_wins,
            scores=canonical_scores,
            current_game=current_game_num,
            total_games=num_games,
            last_action_text=last_action_text,
            leaderboard_names=canonical_names,
            leaderboard_wins=canonical_wins,
            leaderboard_scores=canonical_scores,
            seat_shuffle_banner=shuffle_banner
        )

        running = True
        while running and not env.game_state.game_over:
            # 1. Check window events
            if not spectator.handle_events():
                running = False
                break

            # 2. Handle pause
            if spectator.paused:
                spectator.render_state(
                    env.game_state,
                    player_names=seat_names,
                    wins=canonical_wins,
                    scores=canonical_scores,
                    current_game=current_game_num,
                    total_games=num_games,
                    last_action_text=last_action_text,
                    leaderboard_names=canonical_names,
                    leaderboard_wins=canonical_wins,
                    leaderboard_scores=canonical_scores,
                    seat_shuffle_banner=shuffle_banner
                )
                continue

            current_agent = env.agent_selection
            p_idx = env.agent_name_to_idx[current_agent]
            mask = env.action_mask(current_agent, as_int8=True)
            legal_actions = np.where(mask == 1)[0]

            if len(legal_actions) == 0:
                print(f"Error: Tidak ada aksi legal untuk {seat_names[p_idx]}")
                break

            # 3. Decision Making: Agent di kursi p_idx menggunakan model PPO masing-masing
            legal_moves = env.game_state.get_legal_actions(p_idx)
            agent = seat_agents[p_idx]
            action = agent.select_action(env, p_idx, legal_actions, legal_moves)
            tag = f"[{seat_names[p_idx]} - RL]    "

            # 4. Resolve cards and format narrative
            player_hand = env.game_state.hands[p_idx]
            cards_move = action_to_cards(action, hand=player_hand, legal_moves=legal_moves)

            turn_num = env.game_state.turn_count + 1
            if len(cards_move) == 0:
                last_action_text = f"{seat_names[p_idx]}: PASS"
                print(f"Turn {turn_num:2d} | {seat_names[p_idx]:9s} {tag} : PASS")
            else:
                combo = evaluate_hand(cards_move)
                combo_name = combo.combo_type.name if combo else "Cards"
                c_str = cards_to_str(cards_move)
                last_action_text = f"{seat_names[p_idx]} memainkan {combo_name}: {c_str}"
                print(f"Turn {turn_num:2d} | {seat_names[p_idx]:9s} {tag} : memainkan {combo_name:12s} {c_str}")

            # 5. Step Environment
            env.step(action)
            spectator.render_state(
                env.game_state,
                player_names=seat_names,
                wins=canonical_wins,
                scores=canonical_scores,
                current_game=current_game_num,
                total_games=num_games,
                last_action_text=last_action_text,
                leaderboard_names=canonical_names,
                leaderboard_wins=canonical_wins,
                leaderboard_scores=canonical_scores,
                seat_shuffle_banner=shuffle_banner
            )

            # 6. Paced delay while processing interactive events
            delay_start = time.time()
            effective_delay = max(0.05, step_delay / (spectator.fps / 2.0))
            while time.time() - delay_start < effective_delay:
                if not spectator.handle_events():
                    running = False
                    break
                spectator.render_state(
                    env.game_state,
                    player_names=seat_names,
                    wins=canonical_wins,
                    scores=canonical_scores,
                    current_game=current_game_num,
                    total_games=num_games,
                    last_action_text=last_action_text,
                    leaderboard_names=canonical_names,
                    leaderboard_wins=canonical_wins,
                    leaderboard_scores=canonical_scores,
                    seat_shuffle_banner=shuffle_banner
                )

        # Game ended
        if env.game_state.game_over:
            w = env.game_state.winner
            winner_canonical_idx = current_seats[w]
            canonical_wins[winner_canonical_idx] += 1
            round_points, card_counts = compute_fair_tournament_scores(env.game_state.hands, w)

            # Akumulasikan poin turnamen ke masing-masing agen pemilik kursi
            for s in range(4):
                agent_idx = current_seats[s]
                canonical_scores[agent_idx] += round_points[s]

            is_last_game = (current_game_num == num_games)
            print("-" * 65)
            print(f">>> GAME {current_game_num} SELESAI: {seat_names[w]} MENANG dalam {env.game_state.turn_count} giliran!")
            print(f">>> Poin Turnamen Game Ini : {seat_names[0]}: +{round_points[0]} pt | {seat_names[1]}: +{round_points[1]} pt | {seat_names[2]}: +{round_points[2]} pt | {seat_names[3]}: +{round_points[3]} pt")
            print(f">>> Kartu Berhasil Dibuang : {seat_names[0]}: {13 - card_counts[0]}/13 | {seat_names[1]}: {13 - card_counts[1]}/13 | {seat_names[2]}: {13 - card_counts[2]}/13 | {seat_names[3]}: {13 - card_counts[3]}/13")
            print("-" * 65)

            # Print updated scoreboard (Always sorted by agent's tournament points)
            print_scoreboard(canonical_names, canonical_wins, canonical_scores, current_game_num, num_games)

            if screenshot_dir:
                shot_name = f"tournament_game_{current_game_num}.png"
                spectator.save_screenshot(os.path.join(screenshot_dir, shot_name))

            # Display round end dialog
            if running:
                end_timer = time.time()
                pause_time = 3.5 if is_last_game else (0.6 if num_games >= 50 else 1.2)
                while time.time() - end_timer < pause_time:
                    if not spectator.handle_events():
                        running = False
                        break
                    spectator.render_state(
                        env.game_state,
                        player_names=seat_names,
                        wins=canonical_wins,
                        scores=canonical_scores,
                        round_scores=round_points,
                        current_game=current_game_num,
                        total_games=num_games,
                        last_action_text=f"Game {current_game_num} dimenangkan oleh {seat_names[w]}!",
                        is_tournament_over=is_last_game,
                        leaderboard_names=canonical_names,
                        leaderboard_wins=canonical_wins,
                        leaderboard_scores=canonical_scores,
                        seat_shuffle_banner=shuffle_banner
                    )

        if not running:
            break

    # Determine Champion based on canonical_scores, then canonical_wins
    champ_idx = max(range(4), key=lambda i: (canonical_scores[i], canonical_wins[i]))
    print("\n" + "=" * 65)
    print(f"HASIL AKHIR TURNAMEN {num_games} GAME (SISTEM SKOR ADIL + SEAT ROTATION)")
    print("=" * 65)
    print_scoreboard(canonical_names, canonical_wins, canonical_scores, num_games, num_games)
    print("*" * 65)
    print(f"🏆 JUARA UTAMA TURNAMEN ADALAH: {canonical_names[champ_idx].upper()}! 🏆")
    print(f"Meraih {canonical_scores[champ_idx]} Poin Turnamen dengan {canonical_wins[champ_idx]} Kemenangan Game dari {num_games} Game!")
    print("*" * 65 + "\n")

    if screenshot_dir:
        spectator.save_screenshot(os.path.join(screenshot_dir, "tournament_final_champion.png"))

    spectator.close()
    return champ_idx, canonical_wins, canonical_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turnamen Capsa Banting 4 Pemain")
    parser.add_argument("--games", type=int, default=10, help="Jumlah game turnamen (default: 10)")
    parser.add_argument("--delay", type=float, default=5.0, help="Jeda setiap langkah dalam detik (default: 5.0)")
    parser.add_argument("--fps", type=int, default=2, help="Frame rate Pygame (default: 2)")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.zip", help="Path model RL")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa jendela GUI")
    parser.add_argument("--screenshots", type=str, default=None, help="Direktori penyimpanan screenshot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None untuk pengacakan alami)")
    parser.add_argument("--shuffle-freq", type=int, default=5, help="Frekuensi pengacakan tempat duduk dalam jumlah ronde (default: 5)")
    args = parser.parse_args()

    run_spectator_simulation(
        num_games=args.games,
        fps=args.fps,
        step_delay=args.delay,
        model_path=args.model if os.path.exists(args.model) else None,
        headless=args.headless,
        screenshot_dir=args.screenshots,
        seed=args.seed,
        shuffle_freq=args.shuffle_freq
    )
