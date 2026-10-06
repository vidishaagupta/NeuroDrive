"""
Master Dashboard UI for NeuroDrive.
Coordinates Left Control Panel, Center Racing Viewport, Right Telemetry,
and Top Telemetry Command Bar.
"""

from typing import Any, Callable, Dict, List, Optional
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
    COLOR_WARNING,
)
from game.track import Track
from ui.charts import render_sparkline_chart
from ui.controls import UIButton, UISpeedSelector, UIToggle
from ui.telemetry import CrashAlert, TelemetryPanel
from ui.theme import FontManager, draw_header_banner, draw_panel


class DashboardUI:
    """Orchestrates layout, widgets, and user events for the main simulation viewport."""

    def __init__(
        self,
        width: int,
        height: int,
        on_toggle_pause: Callable[[], None],
        on_reset_training: Callable[[], None],
        on_switch_mode: Callable[[str], None],
        on_speed_change: Callable[[float], None],
        on_view_network: Callable[[], None],
        on_export_analytics: Callable[[], None],
        on_back_to_menu: Callable[[], None],
    ):
        self.width: int = width
        self.height: int = height
        self.on_toggle_pause = on_toggle_pause
        self.on_reset_training = on_reset_training
        self.on_switch_mode = on_switch_mode
        self.on_speed_change = on_speed_change
        self.on_view_network = on_view_network
        self.on_export_analytics = on_export_analytics
        self.on_back_to_menu = on_back_to_menu

        # Viewport toggles
        self.show_vision: bool = True
        self.show_checkpoints: bool = True
        self.show_trajectory: bool = False

        self.crash_alert: CrashAlert = CrashAlert()

        self._init_layout()

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._init_layout()

    def _init_layout(self) -> None:
        # 1. Top Header: Rect(0, 0, width, 52)
        self.header_rect = pygame.Rect(0, 0, self.width, 52)

        # 2. Left Control Panel: width ~216
        left_w = 216
        pad = 12
        panel_y = self.header_rect.bottom + pad
        panel_h = self.height - panel_y - pad

        self.left_rect = pygame.Rect(pad, panel_y, left_w, panel_h)

        # 3. Right Telemetry Panel: width ~260
        right_w = 260
        self.right_rect = pygame.Rect(self.width - right_w - pad, panel_y, right_w, panel_h)
        self.telemetry_panel = TelemetryPanel(self.right_rect)

        # 4. Center Racing Viewport
        vp_x = self.left_rect.right + pad
        vp_w = self.right_rect.left - pad - vp_x
        self.viewport_rect = pygame.Rect(vp_x, panel_y, vp_w, panel_h)

        # Left Panel Interactive Controls
        bx = self.left_rect.x + 12
        bw = self.left_rect.width - 24
        curr_y = self.left_rect.y + 36

        btn_h = 28
        btn_gap = 6
        self.btn_pause = UIButton(
            pygame.Rect(bx, curr_y, bw, btn_h),
            "PAUSE",
            callback=self.on_toggle_pause,
            icon="Ⅱ",
            is_accent=True,
        )
        curr_y += btn_h + btn_gap

        self.btn_reset = UIButton(
            pygame.Rect(bx, curr_y, bw, btn_h),
            "RESET",
            callback=self.on_reset_training,
            icon="↻",
            is_danger=True,
        )
        curr_y += btn_h + btn_gap

        self.btn_network = UIButton(
            pygame.Rect(bx, curr_y, bw, btn_h),
            "NEURAL NETWORK",
            callback=self.on_view_network,
            icon="⛬",
        )
        curr_y += btn_h + btn_gap

        self.btn_export = UIButton(
            pygame.Rect(bx, curr_y, bw, btn_h),
            "EXPORT CHARTS",
            callback=self.on_export_analytics,
            icon="📊",
        )
        curr_y += btn_h + btn_gap

        self.btn_menu = UIButton(
            pygame.Rect(bx, curr_y, bw, btn_h),
            "MAIN MENU",
            callback=self.on_back_to_menu,
            icon="←",
        )
        curr_y += btn_h + 16

        # Speed selector
        self.speed_selector_rect = pygame.Rect(bx, curr_y, bw, 24)
        self.speed_selector = UISpeedSelector(self.speed_selector_rect, on_change=self.on_speed_change)
        curr_y += 24 + 16

        # Toggles
        self.toggle_vision = UIToggle(
            pygame.Rect(bx, curr_y, bw, 22),
            "AI VISION",
            initial_state=self.show_vision,
            on_change=self._set_vision,
        )
        curr_y += 26

        self.toggle_checkpoints = UIToggle(
            pygame.Rect(bx, curr_y, bw, 22),
            "CHECKPOINTS",
            initial_state=self.show_checkpoints,
            on_change=self._set_checkpoints,
        )
        curr_y += 26

        self.toggle_trajectory = UIToggle(
            pygame.Rect(bx, curr_y, bw, 22),
            "TRAJECTORY",
            initial_state=self.show_trajectory,
            on_change=self._set_trajectory,
        )
        curr_y += 30

        # Sparklines area at bottom of left panel
        avail_spark_space = max(80, self.left_rect.bottom - curr_y - 12)
        spark_h = min(72, (avail_spark_space - 8) // 2)
        self.spark_fit_rect = pygame.Rect(bx, curr_y, bw, spark_h)
        self.spark_prog_rect = pygame.Rect(bx, curr_y + spark_h + 8, bw, spark_h)

    def _set_vision(self, val: bool) -> None:
        self.show_vision = val

    def _set_checkpoints(self, val: bool) -> None:
        self.show_checkpoints = val

    def _set_trajectory(self, val: bool) -> None:
        self.show_trajectory = val

    def handle_event(self, event: pygame.event.Event) -> None:
        self.btn_pause.handle_event(event)
        self.btn_reset.handle_event(event)
        self.btn_network.handle_event(event)
        self.btn_export.handle_event(event)
        self.btn_menu.handle_event(event)
        self.speed_selector.handle_event(event)
        self.toggle_vision.handle_event(event)
        self.toggle_checkpoints.handle_event(event)
        self.toggle_trajectory.handle_event(event)

    def update(self) -> None:
        self.crash_alert.update()

    def draw(
        self,
        surface: pygame.Surface,
        track: Track,
        is_paused: bool,
        mode_str: str,
        active_car_data: Dict[str, Any],
        training_data: Dict[str, Any],
        brain_data: Dict[str, Any],
        sensor_readings: List[float],
        fitness_history: List[float],
        progress_history: List[float],
    ) -> None:
        surface.fill(COLOR_BG)

        # 1. Top Header Banner
        gen_badge = training_data.get("generation", 0)
        sys_status = "● PAUSED" if is_paused else "● SIMULATING"
        status_col = COLOR_WARNING if is_paused else COLOR_SUCCESS

        draw_header_banner(
            surface,
            self.header_rect,
            title="NEURODRIVE",
            subtitle="AUTONOMOUS RACING INTELLIGENCE",
            system_status=sys_status,
            status_color=status_col,
            generation_badge=gen_badge,
            mode_text=mode_str,
        )

        # 2. Left Control Panel
        draw_panel(surface, self.left_rect)
        font_sec = FontManager.get(11, bold=True)

        # AI Control Header
        pygame.draw.line(surface, COLOR_ACCENT, (self.left_rect.x + 12, self.left_rect.y + 14), (self.left_rect.x + 12, self.left_rect.y + 24), 2)
        lbl_ctrl = font_sec.render("AI CONTROL", True, COLOR_ACCENT)
        surface.blit(lbl_ctrl, (self.left_rect.x + 20, self.left_rect.y + 13))

        # Update Pause button label dynamically
        self.btn_pause.text = "RESUME" if is_paused else "PAUSE"
        self.btn_pause.icon = "▶" if is_paused else "Ⅱ"

        self.btn_pause.draw(surface)
        self.btn_reset.draw(surface)
        self.btn_network.draw(surface)
        self.btn_export.draw(surface)
        self.btn_menu.draw(surface)

        # Speed title
        lbl_speed = font_sec.render("SIMULATION SPEED", True, COLOR_TEXT_SECONDARY)
        surface.blit(lbl_speed, (self.left_rect.x + 12, self.speed_selector_rect.y - 16))
        self.speed_selector.draw(surface)

        # Toggles title
        lbl_toggles = font_sec.render("VIEWPORT OVERLAYS", True, COLOR_TEXT_SECONDARY)
        surface.blit(lbl_toggles, (self.left_rect.x + 12, self.toggle_vision.rect.y - 16))
        self.toggle_vision.draw(surface)
        self.toggle_checkpoints.draw(surface)
        self.toggle_trajectory.draw(surface)

        # Embedded Sparklines
        if self.spark_prog_rect.bottom <= self.left_rect.bottom - 4:
            render_sparkline_chart(surface, self.spark_fit_rect, "Best Fitness", fitness_history, COLOR_SUCCESS)
            render_sparkline_chart(surface, self.spark_prog_rect, "Max Progress", progress_history, COLOR_ACCENT, "%")

        # 3. Center Racing Viewport
        draw_panel(surface, self.viewport_rect, bg_color=(12, 16, 22), border_color=COLOR_PANEL_BORDER)

        # Viewport Header Overlay Tags
        font_vp = FontManager.get(10, bold=True)
        font_vp_sub = FontManager.get(9, bold=False)

        vp_tag = font_vp.render("AUTONOMOUS CIRCUIT VIEWPORT", True, COLOR_TEXT_PRIMARY)
        surface.blit(vp_tag, (self.viewport_rect.x + 14, self.viewport_rect.y + 10))

        if mode_str == "INFERENCE MODE":
            mode_sub = font_vp_sub.render("● AUTONOMOUS RUN  |  CHAMPION GENOME", True, COLOR_ACCENT)
        else:
            mode_sub = font_vp_sub.render("● 35 AI AGENTS  |  PARALLEL NEAT EVALUATION", True, (80, 160, 185))
        surface.blit(mode_sub, (self.viewport_rect.x + 14 + vp_tag.get_width() + 14, self.viewport_rect.y + 11))

        if self.show_vision:
            vis_tag = font_vp.render("● AI VISION ACTIVE (7 RAYS)", True, COLOR_ACCENT)
        else:
            vis_tag = font_vp.render("○ AI VISION OFF", True, COLOR_TEXT_MUTED)
        surface.blit(vis_tag, (self.viewport_rect.right - 14 - vis_tag.get_width(), self.viewport_rect.y + 10))

        # 4. Crash Alert Overlay
        self.crash_alert.draw(surface, self.viewport_rect.centerx, self.viewport_rect.y + 36)

        # 5. Right Telemetry Panel
        self.telemetry_panel.draw(
            surface,
            active_car_data=active_car_data,
            training_data=training_data,
            brain_data=brain_data,
            sensor_readings=sensor_readings,
            mode_is_inference=(mode_str == "INFERENCE MODE"),
        )
