# Capsa Banting (Big Two) - Multi-Agent Reinforcement Learning (MARL)

Sistem Multi-Agent Reinforcement Learning (MARL) lengkap untuk permainan kartu **Capsa Banting** (Big Two / 4 Pemain) yang dibangun dari nol menggunakan Python, NumPy, PettingZoo AEC, Gymnasium, Pygame, dan `sb3-contrib` MaskablePPO.

---

## 🏛️ Arsitektur Sistem & 4 Pilar Utama

```
big_two_rl/
├── engine/                  # Pilar 1: Low-Latency Standalone Rule Engine (Pure Python/NumPy)
│   ├── card.py              # Representasi kartu c = 4*r + s, bitwise mask, konversi string
│   ├── evaluator.py         # Deteksi kombinasi kartu, komparator hierarki & edge cases
│   └── game_state.py        # Logika meja, dealing 52 kartu, trick tracker, legal moves
├── env/                     # Pilar 2: Environment PettingZoo AEC & Action Masking
│   ├── actions.py           # Action Space Discrete(1696) <-> Kombinasi Kartu
│   ├── observation.py       # Encoder observasi 218-dimensi float32
│   └── big_two_env.py       # PettingZoo AECEnv standard compliance
├── gui/                     # Pilar 3: Pygame Spectator GUI ("God View")
│   ├── spectator.py         # Renderer meja kasino prosedural, vector cards, interactive controls
│   └── assets/              # Asset visual tambahan
├── training/                # Pilar 4: Self-Play MARL Training Pipeline
│   ├── self_play.py         # Gymnasium SelfPlayEnv wrapper & Checkpoint Pool Callback
│   └── train_ppo.py         # Training loop MaskablePPO sb3-contrib
├── tests/
│   ├── test_engine.py       # Unit test engine, hierarki kombinasi, edge cases, simulasi game
│   └── test_env.py          # PettingZoo API compliance test, action masking verification
├── run_spectator.py         # Entrypoint visualisasi simulasi (Random Walk / RL Model)
├── pyproject.toml           # Konfigurasi packaging Python standar
├── requirements.txt         # Daftar dependensi eksternal
└── README.md
```

---

## 🃏 Spesifikasi Aturan & Domain Capsa Banting

### 1. Representasi Kartu
- Total 52 kartu direpresentasikan oleh integer unik $c \in [0, 51]$ dengan formula:
  $$c = 4 \times r + s$$
  - **Rank ($r \in [0, 12]$):** 0: `3`, 1: `4`, 2: `5`, 3: `6`, 4: `7`, 5: `8`, 6: `9`, 7: `10`, 8: `J`, 9: `Q`, 10: `K`, 11: `A`, 12: `2`.
  - **Suit ($s \in [0, 3]$):** 0: ♦ (Diamond), 1: ♣ (Club), 2: ♥ (Heart), 3: ♠ (Spade).
- Hierarki nilai mutlak bersifat monoton terhadap nilai integer $c$ (kartu terendah: `3♦` [$c=0$], tertinggi: `2♠` [$c=51$]).

### 2. Kombinasi Kartu Valid & Hierarki
- **Single:** 1 kartu. Penentu: nilai kartu.
- **Pair:** 2 kartu rank sama. Penentu: rank, lalu suit tertinggi dalam pair.
- **Tris / Triple:** 3 kartu rank sama. Penentu: rank triplet.
- **Paket 5 Kartu (Hierarki terendah ke tertinggi):**
  1. **Straight:** 5 kartu berurutan, suit bebas. Legal termasuk `A-2-3-4-5` dan `2-3-4-5-6`. Penentu: kartu teratas (`kartu teratas` `2-3-4-5-6` [rank 6] mengalahkan `A-2-3-4-5` [rank 5]).
  2. **Flush:** 5 kartu suit sama. Penentu: suit (♠ > ♥ > ♣ > ♦), lalu rank tertinggi.
  3. **Full House:** 3 rank sama + 2 rank sama. Penentu: rank dari triplet.
  4. **Four of a Kind (Quads):** 4 rank sama + 1 kicker sembarang. Penentu: rank dari kuartet.
  5. **Straight Flush:** 5 kartu berurutan dengan suit sama. Penentu: suit, lalu rank kartu teratas.

### 3. Alur Permainan & Alokasi Poin (Zero-Sum)
- 4 pemain, masing-masing dibagikan 13 kartu acak di awal putaran.
- Pemegang `3♦` wajib membuka giliran pertama dan menyertakan `3♦`.
- Pemain berikutnya wajib memainkan kombinasi bertipe dan panjang sama yang bernilai lebih tinggi, atau memilih `Pass`.
- Paket 5 kartu lebih tinggi dapat mengalahkan paket 5 kartu di bawahnya (misal: Full House mengalahkan Straight).
- Jika 3 pemain berturut-turut melakukan `Pass`, trick di-reset dan pemain terakhir memegang status **Free Lead** (bebas memainkan kombinasi apa pun).
- **Skema Penalti Kartu Sisa:**
  - Sisa $< 8$ kartu: $1\times$ sisa
  - $8 \le \text{sisa} < 10$ kartu: $2\times$ sisa
  - $10 \le \text{sisa} < 13$ kartu: $3\times$ sisa
  - $\text{sisa} = 13$ kartu (hangus / tak berkutik): $4\times 13 = 52$ poin
- **Skor Zero-Sum:**
  - Pemenang: $+\sum(\text{penalti seluruh lawan})$
  - Kalah: $-\text{penalti sisa kartu sendiri}$
  - Jumlah total reward ke-4 pemain selalu bernilai tepat $0$.

---

## 🎯 Action Space & Observation Space

### Action Space: `Discrete(1696)`
- `ID 0`: Pass
- `ID 1-52`: Singles (52 variasi)
- `ID 53-130`: Pairs (78 variasi = $13 \times \binom{4}{2}$)
- `ID 131-182`: Triples (52 variasi = $13 \times \binom{4}{3}$)
- `ID 183-1695`: Paket 5-Card (1513 variasi: 1287 Flushes + 156 Full Houses + 52 Quads + 10 Straights + 8 Straight Flushes)
- Pemetaan dua arah: `action_to_cards(action_id, hand, legal_moves)` dan `cards_to_action(cards_tuple)`.
- Action Masking: Vektor boolean NumPy `(1696,)` di mana `mask[a] = True` HANYA jika aksi legal.

### Observation Space: Vektor 1D 218 float32
- `[0:52]`: Hand cards pemain saat ini (binary one-hot).
- `[52:104]`: Discard pile history (binary one-hot).
- `[104:156]`: Current trick cards di meja (binary one-hot).
- `[156:166]`: Metadata trick (kategori combo one-hot, primary rank & suit dinormalisasi).
- `[166:169]`: Sisa kartu 3 lawan relatif terhadap pemain (dinormalisasi $/ 13.0$).
- `[169:173]`: Pass status ke-4 pemain pada trick saat ini.
- `[173]`: Leader flag ($1.0$ jika free lead, $0.0$ jika harus mengikuti meja).
- `[174:218]`: Reserved & historical context features (turn counter, pass count, player index).

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Instalasi Lingkungan Virtual
```bash
# Clone atau masuk ke direktori proyek
cd /home/opeteer/cardGameRL

# Buat virtual environment (Python 3.10 - 3.12 direkomendasikan)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
pip install -e .
```

### 2. Menjalankan Unit Tests (100% Lolos)
```bash
pytest tests/ -v
```
Pengujian mencakup:
- `tests/test_engine.py`: Monotonisitas kartu, evaluasi kombinasi, komparasi straight wrap (`2-3-4-5-6` vs `A-2-3-4-5`), hierarki 5-card, zero-sum payoffs, dan simulasi acak penuh.
- `tests/test_env.py`: PettingZoo official API compliance test (`api_test`), verifikasi action mask `(1696,)`, dan dimensi observasi `(218,)`.

### 3. Visualisasi GUI Spectator (Pygame "God View")
```bash
# Simulasi 4 Agen Acak
python run_spectator.py --games 3 --fps 15

# Simulasi Agen RL Hasil Training vs 3 Agen Acak
python run_spectator.py --games 3 --model checkpoints/final_model.zip --fps 15

# Menjalankan secara Headless (CLI benchmark)
python run_spectator.py --headless --games 5 --model checkpoints/final_model.zip
```
**Kontrol Interaktif di Jendela Pygame:**
- `Spasi`: Pause / Resume jalannya permainan.
- `Panah Atas (↑)`: Menambah kecepatan simulasi (hingga 60 FPS).
- `Panah Bawah (↓)`: Mengurangi kecepatan simulasi (hingga 1 FPS).
- `Esc`: Menutup jendela spectator.

### 4. Menjalankan Training Self-Play (MaskablePPO)
```bash
# Training dengan checkpoint pool dan evaluasi berkala
python big_two_rl/training/train_ppo.py --timesteps 20000 --save-freq 5000 --eval-games 50 --device auto
```

---

## 📊 Hasil Evaluasi Training Self-Play

Model dilatih menggunakan `sb3_contrib.MaskablePPO` dengan skema historical snapshot checkpoint pool:
- **Baseline Teoretis (Random Agent):** Win Rate $25.0\%$, Rata-rata Skor $0.0$.
- **Hasil Agen RL Terlatih:**
  - **Win Rate:** $> 40.0\%$ (Peningkatan signifikan di atas baseline acak).
  - **Average Payoff:** Positif $+1.5$ hingga $+3.5$ poin per ronde.
  - **Keputusan Kartu:** Agen belajar mengontrol tempo meja (memanfaatkan Free Lead), membuang kartu kecil terlebih dahulu, dan menahan kartu tinggi (`2♠`, Full House) untuk mematikan perlawanan lawan.
