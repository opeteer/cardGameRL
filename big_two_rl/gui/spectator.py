"""
Spectator GUI for Big Two (Capsa Banting) using Pygame.
"God View" visualization showing all 4 players' hands, current trick in table center,
active turn highlighting, pass status, live Tournament Scoreboard, and interactive playback controls.
"""
import os
import sys
import pygame
import numpy as np
from typing import Optional, List, Tuple, Sequence, Dict, Any

from big_two_rl.engine.card import (
    Card,
    get_rank,
    get_suit,
    is_red,
    RANK_CHARS,
    SUIT_SYMBOLS,
    SUIT_CHARS,
)
from big_two_rl.engine.evaluator import HandType, HandCombo
from big_two_rl.engine.game_state import GameState

# Color Palette
COLOR_BG = (15, 45, 25)           # Dark casino table rim
COLOR_FELT = (28, 92, 45)         # Green casino felt table
COLOR_FELT_INNER = (32, 105, 52)  # Inner table glow
COLOR_BORDER = (180, 140, 60)     # Brass/gold table accent
COLOR_WHITE = (250, 250, 250)
COLOR_BLACK = (20, 20, 20)
COLOR_RED = (215, 35, 35)
COLOR_GOLD = (255, 215, 0)
COLOR_CYAN = (0, 230, 230)
COLOR_GRAY = (120, 120, 120)
COLOR_PASS = (220, 50, 50)
COLOR_CARD_BG = (252, 252, 250)
COLOR_CARD_BORDER = (160, 160, 160)

DEFAULT_PLAYER_NAMES = ["Curt", "Albert", "Sullivan", "Bernstein"]


class BigTwoSpectator:
    """
    God-view Pygame renderer for Big Two games.
    Renders 1024x768 casino table, 4 hands, current trick, live Scoreboard, and interactive controls.
    """
    def __init__(self, width: int = 1024, height: int = 768, title: str = "Capsa Banting - God View Spectator"):
        self.width = width
        self.height = height
        self.title = title

        # Check if headless mode requested
        self.headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        pygame.init()
        pygame.font.init()

        try:
            self.screen = pygame.display.set_mode((self.width, self.height))
            if not self.headless:
                pygame.display.set_caption(self.title)
        except Exception:
            self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.fps = 4
        self.paused = False

        # Fonts
        self.font_title = pygame.font.SysFont("dejavusans,arial,helvetica", 24, bold=True)
        self.font_large = pygame.font.SysFont("dejavusans,arial,helvetica", 26, bold=True)
        self.font_med = pygame.font.SysFont("dejavusans,arial,helvetica", 16, bold=True)
        self.font_small = pygame.font.SysFont("dejavusans,arial,helvetica", 13, bold=True)
        self.font_tiny = pygame.font.SysFont("dejavusans,arial,helvetica", 12)

        # Card dimensions
        self.card_w = 46
        self.card_h = 68

        # Agent Profile Pictures (Avatars & Framed Portraits)
        self.portraits_dir = os.path.join(os.path.dirname(__file__), "assets", "portraits")
        self.avatars: Dict[str, pygame.Surface] = {}
        self.framed_portraits: Dict[str, pygame.Surface] = {}
        for name in ["curt", "albert", "sullivan", "bernstein"]:
            cap = name.capitalize()
            av_p = os.path.join(self.portraits_dir, f"{name}_avatar.png")
            pt_p = os.path.join(self.portraits_dir, f"{name}.png")
            if os.path.exists(av_p):
                try:
                    surf = pygame.image.load(av_p)
                    self.avatars[cap] = surf.convert_alpha() if pygame.display.get_surface() else surf
                except Exception:
                    pass
            if os.path.exists(pt_p):
                try:
                    surf = pygame.image.load(pt_p)
                    self.framed_portraits[cap] = surf.convert_alpha() if pygame.display.get_surface() else surf
                except Exception:
                    pass

    def handle_events(self) -> bool:
        """
        Process Pygame keyboard and window events.
        Returns False if the user closed the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 2)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 2)
        return True

    def draw_suit_icon(self, surface: pygame.Surface, suit: int, cx: int, cy: int, size: int):
        """Draw procedural vector suit icon."""
        color = COLOR_RED if suit in (0, 2) else COLOR_BLACK

        if suit == 0:  # Diamond
            r = size
            pts = [(cx, cy - r), (cx + int(r * 0.75), cy), (cx, cy + r), (cx - int(r * 0.75), cy)]
            pygame.draw.polygon(surface, color, pts)

        elif suit == 1:  # Club
            r = int(size * 0.45)
            # Top, Left, Right circles
            pygame.draw.circle(surface, color, (cx, cy - int(r * 0.8)), r)
            pygame.draw.circle(surface, color, (cx - int(r * 0.8), cy + int(r * 0.3)), r)
            pygame.draw.circle(surface, color, (cx + int(r * 0.8), cy + int(r * 0.3)), r)
            # Stem
            stem_pts = [(cx, cy), (cx - int(r * 0.4), cy + int(size * 0.9)), (cx + int(r * 0.4), cy + int(size * 0.9))]
            pygame.draw.polygon(surface, color, stem_pts)

        elif suit == 2:  # Heart
            r = int(size * 0.5)
            pygame.draw.circle(surface, color, (cx - int(r * 0.7), cy - int(r * 0.3)), r)
            pygame.draw.circle(surface, color, (cx + int(r * 0.7), cy - int(r * 0.3)), r)
            tri_pts = [
                (cx - int(r * 1.5), cy - int(r * 0.1)),
                (cx + int(r * 1.5), cy - int(r * 0.1)),
                (cx, cy + int(size * 0.95))
            ]
            pygame.draw.polygon(surface, color, tri_pts)

        elif suit == 3:  # Spade
            r = int(size * 0.5)
            pygame.draw.circle(surface, color, (cx - int(r * 0.7), cy + int(r * 0.1)), r)
            pygame.draw.circle(surface, color, (cx + int(r * 0.7), cy + int(r * 0.1)), r)
            tri_pts = [
                (cx - int(r * 1.45), cy + int(r * 0.1)),
                (cx + int(r * 1.45), cy + int(r * 0.1)),
                (cx, cy - int(size * 0.95))
            ]
            pygame.draw.polygon(surface, color, tri_pts)
            stem_pts = [(cx, cy), (cx - int(r * 0.5), cy + int(size * 0.95)), (cx + int(r * 0.5), cy + int(size * 0.95))]
            pygame.draw.polygon(surface, color, stem_pts)

    def draw_card(self, x: int, y: int, card_val: int, highlight: bool = False, scale: float = 1.0):
        """Draw a single procedural card with rank and suit."""
        w = int(self.card_w * scale)
        h = int(self.card_h * scale)

        # Card shadow
        pygame.draw.rect(self.screen, (10, 25, 15), (x + 2, y + 2, w, h), border_radius=5)
        # Card face
        pygame.draw.rect(self.screen, COLOR_CARD_BG, (x, y, w, h), border_radius=5)
        border_col = COLOR_GOLD if highlight else COLOR_CARD_BORDER
        border_w = 2 if highlight else 1
        pygame.draw.rect(self.screen, border_col, (x, y, w, h), width=border_w, border_radius=5)

        r = get_rank(card_val)
        s = get_suit(card_val)
        color = COLOR_RED if s in (0, 2) else COLOR_BLACK
        rank_str = RANK_CHARS[r]

        # Top-left rank label
        lbl = self.font_small.render(rank_str, True, color)
        self.screen.blit(lbl, (x + 3, y + 2))

        # Top-left small suit icon
        self.draw_suit_icon(self.screen, s, x + w - 8, y + 9, size=5)

        # Center larger suit icon
        self.draw_suit_icon(self.screen, s, x + w // 2, y + h // 2, size=int(10 * scale))

    def draw_scoreboard(
        self,
        player_names: List[str],
        wins: List[int],
        scores: List[int],
        current_game: int,
        total_games: int
    ):
        """Draw Tournament Leaderboard HUD in top-left corner."""
        sb_x, sb_y, sb_w, sb_h = 35, 30, 260, 150
        # Background card with border
        pygame.draw.rect(self.screen, (18, 32, 22), (sb_x, sb_y, sb_w, sb_h), border_radius=10)
        pygame.draw.rect(self.screen, COLOR_BORDER, (sb_x, sb_y, sb_w, sb_h), width=2, border_radius=10)

        # Header
        hdr_txt = f"PAPAN SKOR ({current_game}/{total_games})"
        hdr = self.font_small.render(hdr_txt, True, COLOR_GOLD)
        self.screen.blit(hdr, (sb_x + 10, sb_y + 8))

        # Divider line
        pygame.draw.line(self.screen, (70, 70, 70), (sb_x + 8, sb_y + 28), (sb_x + sb_w - 8, sb_y + 28))

        # Rows for Curt, Albert, Sullivan, Bernstein
        max_w = max(wins) if max(wins) > 0 else -1
        for i in range(4):
            y_pos = sb_y + 34 + i * 28
            is_leader = (wins[i] == max_w and max_w > 0)
            col = COLOR_GOLD if is_leader else COLOR_WHITE
            prefix = "👑 " if is_leader else "• "
            agent_name = player_names[i]
            name_str = f"{prefix}{agent_name}"
            score_str = f"{wins[i]} W | {scores[i]} pt"

            # Draw small avatar icon if available
            av = self.avatars.get(agent_name)
            if av:
                av_scaled = pygame.transform.smoothscale(av, (22, 22))
                self.screen.blit(av_scaled, (sb_x + 8, y_pos + 1))
                text_x = sb_x + 34
            else:
                text_x = sb_x + 8

            lbl_name = self.font_tiny.render(name_str, True, col)
            lbl_score = self.font_tiny.render(score_str, True, col)
            self.screen.blit(lbl_name, (text_x, y_pos + 4))
            self.screen.blit(lbl_score, (sb_x + sb_w - lbl_score.get_width() - 8, y_pos + 4))

    def draw_hand(self, player_idx: int, hand: Sequence[int], is_active: bool, has_passed: bool, player_names: List[str]):
        """Render a player's hand at their designated table position."""
        n = len(hand)
        card_spacing = 24 if n > 8 else 30

        # Position mappings:
        # 0: South (bottom)
        # 1: West (left)
        # 2: North (top)
        # 3: East (right)
        labels = [
            f"South ({player_names[0]})",
            f"West ({player_names[1]})",
            f"North ({player_names[2]})",
            f"East ({player_names[3]})"
        ]

        if player_idx == 0:  # South
            total_w = max(0, (n - 1) * card_spacing + self.card_w)
            start_x = (self.width - total_w) // 2
            start_y = self.height - self.card_h - 32
            badge_pos = (self.width // 2, self.height - 18)
        elif player_idx == 2:  # North
            total_w = max(0, (n - 1) * card_spacing + self.card_w)
            start_x = (self.width - total_w) // 2
            start_y = 35
            badge_pos = (self.width // 2, 18)
        elif player_idx == 1:  # West
            total_w = max(0, (n - 1) * card_spacing + self.card_w)
            start_x = 35
            start_y = (self.height - self.card_h) // 2 + 30
            badge_pos = (start_x + total_w // 2, start_y - 18)
        else:  # East (3)
            total_w = max(0, (n - 1) * card_spacing + self.card_w)
            start_x = self.width - total_w - 35
            start_y = (self.height - self.card_h) // 2 + 30
            badge_pos = (start_x + total_w // 2, start_y - 18)

        # Player badge
        bg_col = COLOR_GOLD if is_active else (35, 35, 35)
        txt_col = COLOR_BLACK if is_active else COLOR_WHITE
        info_text = f"{labels[player_idx]} [{n} kartu]"
        lbl = self.font_med.render(info_text, True, txt_col)
        badge_rect = lbl.get_rect(center=badge_pos)
        pad_rect = badge_rect.inflate(16, 6)

        # Draw player avatar circle beside badge
        p_name = player_names[player_idx]
        av = self.avatars.get(p_name)
        if av:
            av_size = 32
            av_scaled = pygame.transform.smoothscale(av, (av_size, av_size))
            av_x = max(8, pad_rect.left - av_size - 6)
            av_y = pad_rect.centery - av_size // 2
            self.screen.blit(av_scaled, (av_x, av_y))
            if is_active:
                pygame.draw.circle(self.screen, COLOR_CYAN, (av_x + av_size // 2, av_y + av_size // 2), av_size // 2 + 2, width=2)

        pygame.draw.rect(self.screen, bg_col, pad_rect, border_radius=10)
        if is_active:
            pygame.draw.rect(self.screen, COLOR_CYAN, pad_rect, width=2, border_radius=10)
        self.screen.blit(lbl, badge_rect)

        # Pass badge if player passed
        if has_passed:
            pass_lbl = self.font_small.render("PASS", True, COLOR_WHITE)
            pass_rect = pass_lbl.get_rect(center=(badge_pos[0], badge_pos[1] + 20))
            pygame.draw.rect(self.screen, COLOR_PASS, pass_rect.inflate(10, 4), border_radius=5)
            self.screen.blit(pass_lbl, pass_rect)

        # Draw cards
        for i, c in enumerate(hand):
            cx = start_x + i * card_spacing
            cy = start_y
            self.draw_card(cx, cy, c, highlight=is_active)

    def draw_current_trick(
        self,
        trick_cards: Sequence[int],
        combo: Optional[HandCombo],
        lead_player: int,
        player_names: List[str],
        last_action_text: Optional[str] = None
    ):
        """Render the cards currently in play at the center of the table."""
        cx = self.width // 2
        cy = self.height // 2

        # Table center plate
        plate_w = 370
        plate_h = 175
        plate_rect = pygame.Rect(cx - plate_w // 2, cy - plate_h // 2 - 10, plate_w, plate_h)
        pygame.draw.rect(self.screen, COLOR_FELT_INNER, plate_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLOR_BORDER, plate_rect, width=2, border_radius=15)

        if not trick_cards:
            lbl = self.font_med.render("FREE LEAD", True, COLOR_GOLD)
            sub = self.font_small.render(f"{player_names[lead_player]} bebas membuka kartu", True, COLOR_WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(cx, cy - 20)))
            self.screen.blit(sub, sub.get_rect(center=(cx, cy + 8)))
        else:
            # Combo description banner
            combo_name = combo.combo_type.name.replace("_", " ") if combo else "TRICK"
            header = self.font_small.render(f"LEAD: {player_names[lead_player]} | {combo_name}", True, COLOR_GOLD)
            self.screen.blit(header, header.get_rect(center=(cx, cy - plate_h // 2 + 10)))

            # Draw trick cards
            n = len(trick_cards)
            spacing = 42
            total_w = (n - 1) * spacing + int(self.card_w * 1.1)
            sx = cx - total_w // 2
            sy = cy - int(self.card_h * 1.1) // 2 - 2

            for i, c in enumerate(trick_cards):
                self.draw_card(sx + i * spacing, sy, c, highlight=True, scale=1.1)

        # Last Action Notification Ribbon
        if last_action_text:
            act_lbl = self.font_small.render(last_action_text, True, COLOR_WHITE)
            act_rect = act_lbl.get_rect(center=(cx, cy + plate_h // 2 - 24))
            pad_rect = act_rect.inflate(18, 6)
            pygame.draw.rect(self.screen, (20, 20, 20), pad_rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_CYAN, pad_rect, width=1, border_radius=8)
            self.screen.blit(act_lbl, act_rect)

    def save_screenshot(self, filepath: str):
        """Save the current screen render to an image file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pygame.image.save(self.screen, filepath)

    def render_state(
        self,
        game_state: GameState,
        player_names: Optional[List[str]] = None,
        wins: Optional[List[int]] = None,
        scores: Optional[List[int]] = None,
        round_scores: Optional[List[int]] = None,
        current_game: int = 1,
        total_games: int = 10,
        last_action_text: Optional[str] = None,
        is_tournament_over: bool = False,
        leaderboard_names: Optional[List[str]] = None,
        leaderboard_wins: Optional[List[int]] = None,
        leaderboard_scores: Optional[List[int]] = None,
        seat_shuffle_banner: Optional[str] = None
    ):
        """Render full game frame from game_state with live scoreboard."""
        names = player_names or DEFAULT_PLAYER_NAMES
        w_list = wins or [0, 0, 0, 0]
        s_list = scores or [0, 0, 0, 0]

        lead_names = leaderboard_names or names
        lead_w = leaderboard_wins if leaderboard_wins is not None else w_list
        lead_s = leaderboard_scores if leaderboard_scores is not None else s_list

        # 1. Background Casino Table
        self.screen.fill(COLOR_BG)
        # Rounded felt table
        table_rect = pygame.Rect(20, 20, self.width - 40, self.height - 40)
        pygame.draw.rect(self.screen, COLOR_FELT, table_rect, border_radius=40)
        pygame.draw.rect(self.screen, COLOR_BORDER, table_rect, width=4, border_radius=40)

        # 2. Status Bar at Top Right with Hardware Seed Telemetry
        hw_tag = f"HW Seed: 0x{game_state.last_seed & 0xFFFFFFFF:08X} (TRNG)" if hasattr(game_state, 'last_seed') and game_state.last_seed is not None else ""
        status_txt = f"{hw_tag} | Giliran: {game_state.turn_count} | Kecepatan: {self.fps} FPS"
        if self.paused:
            status_txt += " [PAUSED]"
        lbl_status = self.font_tiny.render(status_txt, True, COLOR_WHITE)
        self.screen.blit(lbl_status, (self.width - lbl_status.get_width() - 25, 8))

        # Instructions Bar at Bottom Left
        ctrl_txt = "Spasi: Pause/Resume | Panah Atas/Bawah: Kecepatan | Esc: Keluar"
        lbl_ctrl = self.font_tiny.render(ctrl_txt, True, (200, 200, 200))
        self.screen.blit(lbl_ctrl, (25, self.height - 18))

        # Optional Seat Shuffle Notification Banner
        if seat_shuffle_banner:
            shuff_surface = self.font_tiny.render(seat_shuffle_banner, True, COLOR_CYAN)
            shuff_rect = shuff_surface.get_rect(center=(self.width // 2, 8))
            bg_rect = shuff_rect.inflate(14, 4)
            pygame.draw.rect(self.screen, (15, 30, 45), bg_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_CYAN, bg_rect, width=1, border_radius=6)
            self.screen.blit(shuff_surface, shuff_rect)

        # 3. Draw Scoreboard in Top-Left (Always tracks individual agents)
        self.draw_scoreboard(lead_names, lead_w, lead_s, current_game, total_games)

        # 4. Draw 4 Hands
        for p in range(4):
            is_active = (game_state.current_player == p and not game_state.game_over)
            has_passed = game_state.passes[p]
            self.draw_hand(p, game_state.hands[p], is_active, has_passed, names)

        # 5. Draw Center Trick with Last Action Ribbon
        self.draw_current_trick(
            game_state.current_trick_cards,
            game_state.current_trick_combo,
            game_state.trick_lead_player,
            names,
            last_action_text=last_action_text
        )

        # 6. Game Over / Tournament Over Banner
        if game_state.game_over and game_state.winner is not None:
            w = game_state.winner
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            dialog_w, dialog_h = 620, 270
            dx = (self.width - dialog_w) // 2
            dy = (self.height - dialog_h) // 2
            pygame.draw.rect(self.screen, (22, 40, 28), (dx, dy, dialog_w, dialog_h), border_radius=16)
            pygame.draw.rect(self.screen, COLOR_GOLD, (dx, dy, dialog_w, dialog_h), width=3, border_radius=16)

            # Determine winner agent name for portrait display
            if is_tournament_over:
                champ_idx = max(range(4), key=lambda i: (lead_s[i], lead_w[i]))
                winner_name = lead_names[champ_idx]
            else:
                champ_idx = None
                winner_name = names[w]

            # Render framed portrait on the left side of dialog if available
            framed = self.framed_portraits.get(winner_name)
            if framed:
                port_w, port_h = 140, 182
                port_x = dx + 26
                port_y = dy + (dialog_h - port_h) // 2
                framed_scaled = pygame.transform.smoothscale(framed, (port_w, port_h))
                self.screen.blit(framed_scaled, (port_x, port_y))
                pygame.draw.rect(self.screen, COLOR_GOLD, (port_x, port_y, port_w, port_h), width=2, border_radius=4)
                content_cx = dx + 176 + (dialog_w - 186) // 2
            else:
                content_cx = self.width // 2

            if is_tournament_over:
                title_txt = "🏆 JUARA TURNAMEN 🏆"
                win_text = self.font_large.render(title_txt, True, COLOR_GOLD)
                self.screen.blit(win_text, win_text.get_rect(center=(content_cx, dy + 36)))

                name_lbl = self.font_title.render(f"{winner_name.upper()}!", True, COLOR_WHITE)
                self.screen.blit(name_lbl, name_lbl.get_rect(center=(content_cx, dy + 68)))

                sub_txt = f"Total Poin: {lead_s[champ_idx]} pt ({lead_w[champ_idx]} Kemenangan)"
                sub_lbl = self.font_med.render(sub_txt, True, COLOR_CYAN)
                self.screen.blit(sub_lbl, sub_lbl.get_rect(center=(content_cx, dy + 102)))
            else:
                title_txt = f"PEMENANG GAME {current_game}"
                win_text = self.font_large.render(title_txt, True, COLOR_GOLD)
                self.screen.blit(win_text, win_text.get_rect(center=(content_cx, dy + 36)))

                name_lbl = self.font_title.render(f"{winner_name}!", True, COLOR_WHITE)
                self.screen.blit(name_lbl, name_lbl.get_rect(center=(content_cx, dy + 68)))

                if round_scores is not None:
                    payoff_str = f"Poin Game: {names[0]}: +{round_scores[0]} | {names[1]}: +{round_scores[1]} | {names[2]}: +{round_scores[2]} | {names[3]}: +{round_scores[3]}"
                else:
                    payoffs = game_state.get_payoffs()
                    payoff_str = f"Poin Ronde: {names[0]}: {payoffs[0]:+d} | {names[1]}: {payoffs[1]:+d} | {names[2]}: {payoffs[2]:+d} | {names[3]}: {payoffs[3]:+d}"
                p_lbl = self.font_small.render(payoff_str, True, COLOR_CYAN)
                self.screen.blit(p_lbl, p_lbl.get_rect(center=(content_cx, dy + 106)))

            total_str = f"Total Skor: {lead_names[0]}: {lead_s[0]} pt | {lead_names[1]}: {lead_s[1]} pt | {lead_names[2]}: {lead_s[2]} pt | {lead_names[3]}: {lead_s[3]} pt"
            t_lbl = self.font_small.render(total_str, True, COLOR_WHITE)
            self.screen.blit(t_lbl, t_lbl.get_rect(center=(content_cx, dy + 144)))

            hint_txt = "Turnamen selesai!" if is_tournament_over else "Menyiapkan game berikutnya..."
            hint = self.font_tiny.render(hint_txt, True, (200, 200, 200))
            self.screen.blit(hint, hint.get_rect(center=(content_cx, dy + 225)))

        if not self.headless:
            pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
