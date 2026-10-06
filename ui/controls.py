"""
Interactive UI elements: Buttons, Speed Selectors, and Feature Toggles.
Implements micro-interactions, responsive states, and telemetry styling.
"""

from typing import Callable, List, Optional, Tuple
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_ACCENT_DIM,
    COLOR_DANGER,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    SPEED_MULTIPLIERS,
)
from ui.theme import FontManager, draw_panel


class UIButton:
    """Styled interactive button with hover and active states."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        is_accent: bool = False,
        is_danger: bool = False,
        icon: str = "",
    ):
        self.rect: pygame.Rect = rect
        self.text: str = text
        self.callback: Optional[Callable[[], None]] = callback
        self.is_accent: bool = is_accent
        self.is_danger: bool = is_danger
        self.icon: str = icon

        self.is_hovered: bool = False
        self.is_pressed: bool = False
        self.is_active: bool = False
        self.is_disabled: bool = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Processes mouse motion and click events. Returns True if clicked."""
        if self.is_disabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_pressed and self.rect.collidepoint(event.pos):
                self.is_pressed = False
                if self.callback:
                    self.callback()
                return True
            self.is_pressed = False

        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Renders button surface with micro-interaction states."""
        # Determine background and border colors
        if self.is_disabled:
            bg_col = (18, 22, 28)
            border_col = (35, 40, 48)
            text_col = COLOR_TEXT_MUTED
        elif self.is_pressed:
            bg_col = (10, 50, 65) if self.is_accent else (30, 36, 44)
            border_col = COLOR_ACCENT
            text_col = COLOR_ACCENT
        elif self.is_active:
            bg_col = (16, 65, 80)
            border_col = COLOR_ACCENT
            text_col = COLOR_TEXT_PRIMARY
        elif self.is_hovered:
            bg_col = (25, 45, 58) if self.is_accent else (32, 40, 50)
            border_col = COLOR_ACCENT if self.is_accent else (70, 80, 95)
            text_col = COLOR_TEXT_PRIMARY
        else:
            bg_col = (18, 30, 38) if self.is_accent else (24, 29, 36)
            border_col = (0, 140, 160) if self.is_accent else COLOR_PANEL_BORDER
            text_col = COLOR_ACCENT if self.is_accent else COLOR_TEXT_PRIMARY

        if self.is_danger:
            if self.is_hovered:
                border_col = COLOR_DANGER
                text_col = COLOR_DANGER

        # Draw panel
        draw_panel(surface, self.rect, bg_color=bg_col, border_color=border_col, border_radius=4)

        # Draw label & icon
        font = FontManager.get(12, bold=True)
        display_str = f"{self.icon} {self.text}".strip()
        text_surf = font.render(display_str, True, text_col)

        tx = self.rect.centerx - text_surf.get_width() // 2
        ty = self.rect.centery - text_surf.get_height() // 2
        surface.blit(text_surf, (tx, ty))


class UIToggle:
    """A clean ON/OFF toggle switch for features."""

    def __init__(self, rect: pygame.Rect, label: str, initial_state: bool = True, on_change: Optional[Callable[[bool], None]] = None):
        self.rect: pygame.Rect = rect
        self.label: str = label
        self.state: bool = initial_state
        self.on_change: Optional[Callable[[bool], None]] = on_change
        self.is_hovered: bool = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.state = not self.state
                if self.on_change:
                    self.on_change(self.state)
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        font_lbl = FontManager.get(12, bold=False)
        font_state = FontManager.get(11, bold=True)

        # Draw label
        lbl_surf = font_lbl.render(self.label, True, COLOR_TEXT_PRIMARY if self.is_hovered else COLOR_TEXT_SECONDARY)
        surface.blit(lbl_surf, (self.rect.x, self.rect.centery - lbl_surf.get_height() // 2))

        # Draw switch pill
        switch_w = 44
        switch_h = 20
        switch_x = self.rect.right - switch_w
        switch_y = self.rect.centery - switch_h // 2
        switch_rect = pygame.Rect(switch_x, switch_y, switch_w, switch_h)

        bg_col = (10, 50, 60) if self.state else (22, 27, 34)
        border_col = COLOR_ACCENT if self.state else COLOR_PANEL_BORDER

        draw_panel(surface, switch_rect, bg_color=bg_col, border_color=border_col, border_radius=10)

        # Draw status text inside pill
        state_txt = "ON" if self.state else "OFF"
        state_col = COLOR_ACCENT if self.state else COLOR_TEXT_MUTED
        txt_surf = font_state.render(state_txt, True, state_col)
        surface.blit(txt_surf, (switch_rect.centerx - txt_surf.get_width() // 2, switch_rect.centery - txt_surf.get_height() // 2))


class UISpeedSelector:
    """Multi-speed button group: 0.5x, 1x, 2x, 5x, 10x."""

    def __init__(self, rect: pygame.Rect, on_change: Optional[Callable[[float], None]] = None):
        self.rect: pygame.Rect = rect
        self.speeds: List[float] = SPEED_MULTIPLIERS
        self.current_speed: float = 1.0
        self.on_change: Optional[Callable[[float], None]] = on_change
        self.buttons: List[UIButton] = []
        self._init_buttons()

    def _init_buttons(self) -> None:
        btn_count = len(self.speeds)
        gap = 4
        btn_w = (self.rect.width - gap * (btn_count - 1)) // btn_count
        h = self.rect.height

        self.buttons = []
        for i, s in enumerate(self.speeds):
            bx = self.rect.x + i * (btn_w + gap)
            btn_rect = pygame.Rect(bx, self.rect.y, btn_w, h)
            text = f"{s}x" if s != int(s) else f"{int(s)}x"

            # Capture speed via closure
            def make_cb(speed_val: float):
                return lambda: self.set_speed(speed_val)

            btn = UIButton(btn_rect, text, callback=make_cb(s))
            if s == self.current_speed:
                btn.is_active = True
            self.buttons.append(btn)

    def set_speed(self, speed_val: float) -> None:
        self.current_speed = speed_val
        for btn, s in zip(self.buttons, self.speeds):
            btn.is_active = (s == speed_val)
        if self.on_change:
            self.on_change(speed_val)

    def handle_event(self, event: pygame.event.Event) -> bool:
        handled = False
        for btn in self.buttons:
            if btn.handle_event(event):
                handled = True
        return handled

    def draw(self, surface: pygame.Surface) -> None:
        for btn in self.buttons:
            btn.draw(surface)
