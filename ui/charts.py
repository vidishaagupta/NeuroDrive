"""
Real-time embedded analytics sparklines and trend charts for NeuroDrive UI.
Renders generation metrics (Best Fitness, Avg Fitness, Progress, Survival) directly to Pygame.
"""

from typing import List, Optional, Tuple
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
)
from ui.theme import FontManager, draw_panel


def render_sparkline_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    data: List[float],
    line_color: Tuple[int, int, int] = COLOR_ACCENT,
    unit: str = "",
) -> None:
    """
    Renders an embedded mini trend chart with technical grid and range labels.
    """
    draw_panel(surface, rect, bg_color=(18, 22, 28), border_color=COLOR_PANEL_BORDER, border_radius=3)

    font_title = FontManager.get(10, bold=True)
    font_val = FontManager.get(10, bold=False)

    # Title & latest value
    latest_str = f"{data[-1]:.1f}{unit}" if data else f"N/A{unit}"
    t_surf = font_title.render(title.upper(), True, COLOR_TEXT_MUTED)
    v_surf = font_val.render(latest_str, True, line_color)

    surface.blit(t_surf, (rect.x + 8, rect.y + 6))
    surface.blit(v_surf, (rect.right - 8 - v_surf.get_width(), rect.y + 6))

    chart_box = pygame.Rect(rect.x + 8, rect.y + 22, rect.width - 16, rect.height - 28)

    # Draw faint grid
    grid_lines = 3
    for i in range(1, grid_lines):
        gy = chart_box.y + int(chart_box.height * (i / grid_lines))
        pygame.draw.line(surface, (26, 32, 40), (chart_box.left, gy), (chart_box.right, gy), 1)

    if not data or len(data) < 2:
        empty_surf = font_val.render("Awaiting Gen 1 data...", True, COLOR_TEXT_MUTED)
        surface.blit(
            empty_surf,
            (chart_box.centerx - empty_surf.get_width() // 2, chart_box.centery - empty_surf.get_height() // 2),
        )
        return

    # Determine scale
    min_val = min(data)
    max_val = max(data)
    val_range = max(1e-5, max_val - min_val)

    # Plot points
    n = len(data)
    pts = []
    for i, val in enumerate(data):
        px = chart_box.left + int((i / (n - 1)) * chart_box.width)
        # Inverted Y for screen coordinates
        norm_y = (val - min_val) / val_range
        py = chart_box.bottom - int(norm_y * chart_box.height)
        pts.append((px, py))

    # Draw line
    if len(pts) >= 2:
        pygame.draw.lines(surface, line_color, False, pts, 2)

    # Highlight latest point
    last_pt = pts[-1]
    pygame.draw.circle(surface, line_color, last_pt, 3)
    pygame.draw.circle(surface, (255, 255, 255), last_pt, 1)
