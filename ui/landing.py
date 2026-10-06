"""
Landing screen module for NeuroDrive.
Minimalist, high-tech interface welcoming users with telemetry aesthetic and mode selectors.
"""

from typing import Callable, Optional
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)
from ui.controls import UIButton
from ui.theme import FontManager, draw_panel


class LandingScreen:
    """Renders the landing view and handles entry action dispatch."""

    def __init__(
        self,
        width: int,
        height: int,
        on_start_training: Callable[[], None],
        on_load_best: Callable[[], None],
        on_view_network: Callable[[], None],
        on_view_about: Callable[[], None],
    ):
        self.width: int = width
        self.height: int = height
        self.on_start_training: Callable[[], None] = on_start_training
        self.on_load_best: Callable[[], None] = on_load_best
        self.on_view_network: Callable[[], None] = on_view_network
        self.on_view_about: Callable[[], None] = on_view_about

        self.buttons: list[UIButton] = []
        self._init_layout()

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._init_layout()

    def _init_layout(self) -> None:
        center_x = self.width // 2

        btn_w, btn_h = 240, 44
        btn_y = self.height // 2 + 120

        self.buttons = [
            UIButton(
                pygame.Rect(center_x - btn_w - 15, btn_y, btn_w, btn_h),
                "START TRAINING",
                callback=self.on_start_training,
                is_accent=True,
                icon="▶",
            ),
            UIButton(
                pygame.Rect(center_x + 15, btn_y, btn_w, btn_h),
                "LOAD BEST AGENT",
                callback=self.on_load_best,
                icon="⚡",
            ),
            UIButton(
                pygame.Rect(center_x - btn_w - 15, btn_y + 54, btn_w, btn_h - 4),
                "NEURAL NETWORK",
                callback=self.on_view_network,
                icon="⛬",
            ),
            UIButton(
                pygame.Rect(center_x + 15, btn_y + 54, btn_w, btn_h - 4),
                "ABOUT ARCHITECTURE",
                callback=self.on_view_about,
                icon="ℹ",
            ),
        ]

    def handle_event(self, event: pygame.event.Event) -> None:
        for btn in self.buttons:
            btn.handle_event(event)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(COLOR_BG)

        center_x = self.width // 2

        # 1. Subtle background circuit accent lines
        grid_col = (18, 23, 30)
        for x in range(0, self.width, 60):
            pygame.draw.line(surface, grid_col, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 60):
            pygame.draw.line(surface, grid_col, (0, y), (self.width, y), 1)

        # 2. Main Title Banner
        font_hero = FontManager.get(44, bold=True)
        font_sub = FontManager.get(14, bold=True)
        font_tag = FontManager.get(13, bold=False)

        hero_surf = font_hero.render("NEURODRIVE", True, COLOR_TEXT_PRIMARY)
        surface.blit(hero_surf, (center_x - hero_surf.get_width() // 2, 90))

        sub_surf = font_sub.render("AUTONOMOUS RACING INTELLIGENCE", True, COLOR_ACCENT)
        surface.blit(sub_surf, (center_x - sub_surf.get_width() // 2, 148))

        tag_surf = font_tag.render('"Teaching an AI to drive — one generation at a time."', True, COLOR_TEXT_SECONDARY)
        surface.blit(tag_surf, (center_x - tag_surf.get_width() // 2, 178))

        # 3. Three Feature Cards
        card_w = 260
        card_h = 110
        card_y = 230
        card_gap = 24
        total_cards_w = card_w * 3 + card_gap * 2
        start_x = center_x - total_cards_w // 2

        features = [
            ("NEUROEVOLUTION", "NEAT-powered autonomous learning", "Genomes evolve topology & weights through selective breeding without backpropagation."),
            ("AI PERCEPTION", "7-Ray Raycast Vision", "Normalized geometric perception rays calculate real-time track boundary proximity."),
            ("LIVE EVOLUTION", "Real-Time Telemetry", "Watch population navigate circuit simultaneously with accelerated simulation speeds."),
        ]

        font_card_head = FontManager.get(13, bold=True)
        font_card_sub = FontManager.get(11, bold=True)
        font_card_body = FontManager.get(10, bold=False)

        for i, (f_title, f_sub, f_desc) in enumerate(features):
            c_x = start_x + i * (card_w + card_gap)
            c_rect = pygame.Rect(c_x, card_y, card_w, card_h)

            draw_panel(surface, c_rect, bg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_radius=6)

            # Top accent stripe
            pygame.draw.line(surface, COLOR_ACCENT, (c_x + 12, card_y + 1), (c_x + 40, card_y + 1), 2)

            t_s = font_card_head.render(f_title, True, COLOR_TEXT_PRIMARY)
            s_s = font_card_sub.render(f_sub, True, COLOR_ACCENT)

            surface.blit(t_s, (c_x + 14, card_y + 14))
            surface.blit(s_s, (c_x + 14, card_y + 34))

            # Multi-line description
            words = f_desc.split(" ")
            line1, line2 = "", ""
            for w in words:
                if len(line1) + len(w) < 36:
                    line1 += w + " "
                else:
                    line2 += w + " "

            d1 = font_card_body.render(line1.strip(), True, COLOR_TEXT_MUTED)
            d2 = font_card_body.render(line2.strip(), True, COLOR_TEXT_MUTED)
            surface.blit(d1, (c_x + 14, card_y + 60))
            surface.blit(d2, (c_x + 14, card_y + 76))

        # 4. Render Action Buttons
        for btn in self.buttons:
            btn.draw(surface)

        # 5. Footer info
        font_foot = FontManager.get(10, bold=False)
        foot_str = "NeuroDrive v1.0.0 — Pure NEAT Neuroevolution Architecture | Deterministic 2D Physics"
        foot_surf = font_foot.render(foot_str, True, (60, 70, 85))
        surface.blit(foot_surf, (center_x - foot_surf.get_width() // 2, self.height - 28))
