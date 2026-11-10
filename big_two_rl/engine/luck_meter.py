"""
Mekanisme Evaluasi dan Akumulasi Poin Hoki (Luck Meter) untuk Capsa Banting (Big Two).

Menghitung bobot 'keberuntungan' (Hand Luck Score) dari 13 kartu awal yang dibagikan
kepada tiap pemain di awal ronde berdasarkan:
1. Kekuatan Nilai Kartu Absolut (Rank 3 s/d 2).
2. Kartu Kekuatan / Kartu Sakti:
   - Kartu '2' (Kartu Tertinggi): Bonus eksponensial (+15 per satu, +35 per dua, +65 per tiga, +100 per Dragon 2s).
   - Kartu 2♠ (Dewa Angin / Kartu Mutlak Tertinggi): Bonus keagungan +5 pt.
   - Kartu As ('A'): +8 pt per As.
   - Kartu 3♦ (Inisiatif Pembuka Giliran Pertama): +5 pt hak memimpin putaran.
3. Struktur & Koordinasi Tangan (Partition Steps):
   - Efisiensi langkah minimum untuk menghabiskan tangan (N_step): semakin sedikit langkah, semakin hoki.
   - Kombinasi bawaan (Quads +30, Triples +12, Pairs +4).
4. Penalti Sampah Unpaired Rendah (Rank 3..8 tanpa pasangan/kombo).
"""
from typing import Sequence, Dict, Any, Tuple
from collections import Counter

from big_two_rl.engine.card import get_rank, get_suit, card_to_str
from big_two_rl.engine.evaluator import compute_hand_partition_metrics


def calculate_initial_hand_luck(hand: Sequence[int]) -> int:
    """
    Hitung skor hoki dari 13 kartu tangan awal pemain.
    Skor normal berkisar antara 70 s/d 280+ poin:
    - Rata-rata tangan biasa : ~130 poin
    - Tangan ampas / sial    : < 100 poin
    - Tangan dewa / super hoki: > 170 poin
    """
    if not hand:
        return 0

    hand_tuple = tuple(sorted(hand))
    ranks = [get_rank(c) for c in hand]
    rank_counts = Counter(ranks)

    # 1. Base rank strength (sum of ranks, rank in 0..12)
    base_rank_score = sum(ranks)  # average ~78 poin

    # 2. Power cards: 2s (rank 12)
    twos = rank_counts.get(12, 0)
    twos_table = {0: 0, 1: 15, 2: 35, 3: 65, 4: 100}
    twos_bonus = twos_table.get(twos, 100)
    if 51 in hand:  # 2 of Spades (Absolute Supreme Card)
        twos_bonus += 5

    # 3. Power cards: Aces (rank 11)
    aces = rank_counts.get(11, 0)
    aces_bonus = aces * 8

    # 4. Initiative Card: 3 of Diamonds (card 0)
    init_bonus = 5 if 0 in hand else 0

    # 5. Hand Coordination & Partition Steps
    steps, unpaired_low = compute_hand_partition_metrics(hand_tuple)
    # Langkah ideal: 4-5 langkah -> bonus tinggi; 8+ langkah -> tidak ada bonus
    step_bonus = max(0, (8 - steps) * 8)
    garbage_penalty = unpaired_low * 3

    # 6. Multiples Bonus (Quads, Triples, Pairs)
    multiples_bonus = 0
    for r, cnt in rank_counts.items():
        if cnt == 4:
            multiples_bonus += 30
        elif cnt == 3:
            multiples_bonus += 12
        elif cnt == 2:
            multiples_bonus += 4

    total_luck = (
        base_rank_score
        + twos_bonus
        + aces_bonus
        + init_bonus
        + step_bonus
        + multiples_bonus
        - garbage_penalty
    )
    return max(10, int(total_luck))


def get_hand_luck_breakdown(hand: Sequence[int]) -> Dict[str, Any]:
    """
    Dapatkan rincian statistik keberuntungan kartu tangan awal.
    """
    hand_tuple = tuple(sorted(hand))
    ranks = [get_rank(c) for c in hand]
    rank_counts = Counter(ranks)

    twos = rank_counts.get(12, 0)
    aces = rank_counts.get(11, 0)
    has_2s = (51 in hand)
    has_3d = (0 in hand)
    steps, unpaired_low = compute_hand_partition_metrics(hand_tuple)

    # Identifikasi sorotan kartu
    highlights = []
    if has_2s:
        highlights.append("2♠ Dewa")
    elif twos > 0:
        highlights.append(f"{twos}x Kartu 2")
    if aces >= 2:
        highlights.append(f"{aces}x As")
    elif aces == 1:
        highlights.append("As")
    if has_3d:
        highlights.append("3♦ Lead")
    if steps <= 4:
        highlights.append(f"{steps} Langkah (Ultra Cepat)")
    elif steps == 5:
        highlights.append(f"{steps} Langkah")

    # Identifikasi kelipatan
    quads = [r for r, c in rank_counts.items() if c == 4]
    triples = [r for r, c in rank_counts.items() if c == 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    if quads:
        highlights.append("Quads Bom")
    if triples:
        highlights.append("Tris")
    if len(pairs) >= 3:
        highlights.append(f"{len(pairs)} Pair")

    total_score = calculate_initial_hand_luck(hand)
    return {
        "score": total_score,
        "twos": twos,
        "aces": aces,
        "has_2s": has_2s,
        "has_3d": has_3d,
        "steps": steps,
        "unpaired_low": unpaired_low,
        "quads": len(quads),
        "triples": len(triples),
        "pairs": len(pairs),
        "highlights": ", ".join(highlights) if highlights else "Tangan Standar"
    }
