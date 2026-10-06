"""
Theme, typography, palette, and geometric drawing helpers for NeuroDrive UI.
Establishes the "Autonomous Driving Research Lab + F1 Telemetry" aesthetic.
"""

from typing import Dict, List, Optional, Tuple
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_ACCENT_DIM,
    COLOR_ACCENT_GLOW,
    COLOR_BG,
    COLOR_DANGER,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
)


class FontManager:
    """Manages cached Pygame fonts for crisp typography across resolutions."""

    _fonts: Dict[Tuple[str, int, bool], pygame.font.Font] = {}

    @classmethod
    def get(cls, size: int = 14, bold: bool = False, font_name: Optional[str] = None) -> pygame.font.Font:
        key = (font_name or "consolas", size, bold)
        if key not in cls._fonts:
            try:
                # Try modern system monospace/clean fonts
                candidates = ["consolas", "segoeui", "arial", "dejavusans"]
                chosen = font_name or "consolas"
                font = pygame.font.SysFont(chosen, size, bold=bold)
                cls._fonts[key] = font
            except Exception:
                cls._fonts[key] = pygame.font.Font(None, size)
        return cls._fonts[key]


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    bg_color: Tuple[int, int, int] = COLOR_PANEL_BG,
    border_color: Tuple[int, int, int] = COLOR_PANEL_BORDER,
    border_width: int = 1,
    border_radius: int = 4,
    glow_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draws a clean technical panel card with subtle border and optional glow."""
    # Base panel surface
    pygame.draw.rect(surface, bg_color, rect, border_radius=border_radius)

    # Optional subtle glow outline
    if glow_color:
        glow_rect = rect.inflate(2, 2)
        pygame.draw.rect(surface, glow_color, glow_rect, width=1, border_radius=border_radius + 1)

    # Standard clean border
    if border_width > 0:
        pygame.draw.rect(surface, border_color, rect, width=border_width, border_radius=border_radius)


def draw_header_banner(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    subtitle: Optional[str] = None,
    system_status: str = "SYSTEM ONLINE",
    status_color: Tuple[int, int, int] = COLOR_SUCCESS,
    generation_badge: Optional[int] = None,
    mode_text: str = "TRAINING MODE",
) -> None:
    """Draws the top telemetry command bar."""
    # Top bar background
    pygame.draw.rect(surface, (16, 21, 28), rect)
    pygame.draw.line(surface, COLOR_PANEL_BORDER, (rect.left, rect.bottom - 1), (rect.right, rect.bottom - 1), 1)

    font_brand = FontManager.get(18, bold=True)
    font_sub = FontManager.get(11, bold=False)
    font_meta = FontManager.get(12, bold=True)

    # Left: Brand & Subtitle
    brand_surf = font_brand.render(title, True, COLOR_TEXT_PRIMARY)
    surface.blit(brand_surf, (rect.x + 18, rect.y + 10))

    if subtitle:
        sub_surf = font_sub.render(subtitle, True, COLOR_ACCENT)
        surface.blit(sub_surf, (rect.x + 20 + brand_surf.get_width() + 10, rect.y + 14))

    # Right: Telemetry status indicators
    curr_x = rect.right - 20

    # Mode badge
    mode_surf = font_meta.render(mode_text, True, COLOR_TEXT_SECONDARY)
    curr_x -= mode_surf.get_width()
    surface.blit(mode_surf, (curr_x, rect.y + 13))

    curr_x -= 30

    # Generation badge
    if generation_badge is not None:
        gen_str = f"GEN {generation_badge}"
        gen_surf = font_meta.render(gen_str, True, COLOR_ACCENT)
        curr_x -= gen_surf.get_width()
        surface.blit(gen_surf, (curr_x, rect.y + 13))
        curr_x -= 30

    # System Status Dot + Text
    status_surf = font_meta.render(system_status, True, status_color)
    curr_x -= status_surf.get_width()
    surface.blit(status_surf, (curr_x, rect.y + 13))

    # Green pulse dot
    dot_x = curr_x - 12
    dot_y = rect.y + 20
    pygame.draw.circle(surface, status_color, (dot_x, dot_y), 4)


def draw_metric_row(
    surface: pygame.Surface,
    x: int,
    y: int,
    width: int,
    label: str,
    value_str: str,
    value_color: Tuple[int, int, int] = COLOR_TEXT_PRIMARY,
    is_large: bool = False,
) -> int:
    """Renders a labeled telemetry metric row."""
    font_lbl = FontManager.get(11, bold=False)
    font_val = FontManager.get(16 if is_large else 13, bold=True)

    lbl_surf = font_lbl.render(label.upper(), True, COLOR_TEXT_MUTED)
    val_surf = font_val.render(value_str, True, value_color)

    surface.blit(lbl_surf, (x, y))
    val_x = x + width - val_surf.get_width()
    surface.blit(val_surf, (val_x, y - (2 if is_large else 0)))

    row_h = 24 if is_large else 20
    return y + row_h
