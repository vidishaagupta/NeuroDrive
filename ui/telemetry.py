"""
Live Telemetry, Training Status, AI Brain, and Sensor Perception HUD components.
Displays real-time vehicle kinematics, neural network metrics, and sensor values
with high-precision F1 research telemetry aesthetics.
"""

from typing import Any, Dict, List, Optional, Tuple
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
    COLOR_WARNING,
)
from ui.theme import FontManager, draw_metric_row, draw_panel


class CrashAlert:
    """Manages non-blocking collision feedback notifications."""

    def __init__(self):
        self.message: str = ""
        self.fitness: float = 0.0
        self.progress_pct: float = 0.0
        self.frames_left: int = 0
        self.max_frames: int = 65

    def trigger(self, fitness: float, progress_pct: float) -> None:
        self.message = "COLLISION DETECTED"
        self.fitness = fitness
        self.progress_pct = progress_pct
        self.frames_left = self.max_frames

    def update(self) -> None:
        if self.frames_left > 0:
            self.frames_left -= 1

    def draw(self, surface: pygame.Surface, center_x: int, top_y: int) -> None:
        if self.frames_left <= 0:
            return

        w, h = 250, 42
        rect = pygame.Rect(center_x - w // 2, top_y, w, h)

        draw_panel(
            surface,
            rect,
            bg_color=(32, 16, 20),
            border_color=COLOR_DANGER,
            border_radius=3,
        )

        font_head = FontManager.get(10, bold=True)
        font_sub = FontManager.get(10, bold=False)

        head_surf = font_head.render("⚠  COLLISION DETECTED", True, COLOR_DANGER)
        surface.blit(head_surf, (rect.centerx - head_surf.get_width() // 2, rect.y + 6))

        sub_str = f"Fitness: {self.fitness:.1f}  |  Progress: {self.progress_pct:.1f}%"
        sub_surf = font_sub.render(sub_str, True, COLOR_TEXT_SECONDARY)
        surface.blit(sub_surf, (rect.centerx - sub_surf.get_width() // 2, rect.y + 22))


class TelemetryPanel:
    """Renders the right-hand telemetry and status display with research lab clarity."""

    def __init__(self, rect: pygame.Rect):
        self.rect: pygame.Rect = rect

    def draw(
        self,
        surface: pygame.Surface,
        active_car_data: Dict[str, Any],
        training_data: Dict[str, Any],
        brain_data: Dict[str, Any],
        sensor_readings: List[float],
        mode_is_inference: bool = False,
    ) -> None:
        """Renders all right panel telemetry modules."""
        draw_panel(surface, self.rect)

        curr_y = self.rect.y + 10
        pad_x = 12
        content_w = self.rect.width - pad_x * 2
        font_sec = FontManager.get(11, bold=True)
        font_sub = FontManager.get(9, bold=False)

        def draw_sec_header(title: str, tag: Optional[str] = None) -> int:
            nonlocal curr_y
            # Accent left tick
            pygame.draw.line(surface, COLOR_ACCENT, (self.rect.x + pad_x, curr_y + 2), (self.rect.x + pad_x, curr_y + 12), 2)
            t_s = font_sec.render(title, True, COLOR_ACCENT)
            surface.blit(t_s, (self.rect.x + pad_x + 8, curr_y))

            if tag:
                tg_s = font_sub.render(tag, True, COLOR_TEXT_MUTED)
                surface.blit(tg_s, (self.rect.x + pad_x + content_w - tg_s.get_width(), curr_y + 2))

            curr_y += 16
            pygame.draw.line(
                surface,
                COLOR_PANEL_BORDER,
                (self.rect.x + pad_x, curr_y),
                (self.rect.x + pad_x + content_w, curr_y),
                1,
            )
            curr_y += 6
            return curr_y

        # ----------------------------------------------------
        # 1. LIVE TELEMETRY / BEST AGENT TELEMETRY
        # ----------------------------------------------------
        sec_title = "BEST AGENT TELEMETRY" if mode_is_inference else "LIVE TELEMETRY"
        tag_text = "AUTONOMOUS DRIVER" if mode_is_inference else "ACTIVE LEADER"
        draw_sec_header(sec_title, tag_text)

        speed_kmh = active_car_data.get("speed", 0.0) * 19.5
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "SPEED", f"{speed_kmh:.0f} km/h", COLOR_TEXT_PRIMARY, is_large=True)

        prog_pct = active_car_data.get("progress_pct", 0.0)
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "TRACK PROGRESS", f"{prog_pct:.1f}%", COLOR_ACCENT)

        # Progress bar
        bar_h = 4
        bar_y = curr_y - 2
        pygame.draw.rect(surface, (28, 34, 42), pygame.Rect(self.rect.x + pad_x, bar_y, content_w, bar_h), border_radius=1)
        fill_w = int(max(0.0, min(100.0, prog_pct)) * 0.01 * content_w)
        if fill_w > 0:
            pygame.draw.rect(surface, COLOR_ACCENT, pygame.Rect(self.rect.x + pad_x, bar_y, fill_w, bar_h), border_radius=1)
        curr_y += 8

        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "CHECKPOINT", f"{active_car_data.get('checkpoint_str', '0/0')}")
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "FITNESS", f"{active_car_data.get('fitness', 0.0):.2f}", COLOR_SUCCESS)
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "SURVIVAL", f"{active_car_data.get('survival_time', 0.0):.1f} s")
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "LAPS COMPLETED", f"{active_car_data.get('laps', 0)}")

        curr_y += 6

        # ----------------------------------------------------
        # 2. TRAINING STATUS (Only in Training mode)
        # ----------------------------------------------------
        if not mode_is_inference:
            draw_sec_header("TRAINING STATUS", f"POP: {training_data.get('population_size', 35)}")

            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "GENERATION", f"{training_data.get('generation', 0)}", COLOR_TEXT_PRIMARY)
            alive_count = training_data.get("alive_count", 0)
            pop_size = training_data.get("population_size", 35)
            alive_col = COLOR_SUCCESS if alive_count > 0 else COLOR_DANGER
            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "ALIVE AGENTS", f"{alive_count} / {pop_size}", alive_col)
            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "BEST FITNESS", f"{training_data.get('best_fitness', 0.0):.2f}", COLOR_SUCCESS)
            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "AVERAGE FITNESS", f"{training_data.get('avg_fitness', 0.0):.2f}")
            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "MAX PROGRESS", f"{training_data.get('max_progress', 0.0):.1f}%", COLOR_ACCENT)

            status_str = training_data.get("status_text", "SIMULATING")
            status_color = COLOR_WARNING if "PAUSED" in status_str else COLOR_SUCCESS
            curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "STATUS", status_str, status_color)

            curr_y += 6

        # ----------------------------------------------------
        # 3. AI BRAIN (NEAT TOPOLOGY FLOW)
        # ----------------------------------------------------
        draw_sec_header("AI BRAIN (NEAT TOPOLOGY)")

        # Clear Architecture Pipeline Breadcrumb
        font_flow = FontManager.get(9, bold=True)
        flow_surf = font_flow.render("7 SENSORS  ►  NEAT NET  ►  3 ACTIONS", True, (100, 180, 205))
        surface.blit(flow_surf, (self.rect.x + pad_x, curr_y))
        curr_y += 16

        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "INPUT NODES", f"{brain_data.get('inputs', 7)}")
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "HIDDEN NODES", f"{brain_data.get('hidden_count', 0)}", COLOR_ACCENT)
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "CONNECTIONS", f"{brain_data.get('connection_count', 0)}")
        curr_y = draw_metric_row(surface, self.rect.x + pad_x, curr_y, content_w, "OUTPUT NODES", f"{brain_data.get('outputs', 3)}")

        curr_y += 4

        # Action chips: STEER LEFT, STRAIGHT, STEER RIGHT
        action_names = ["STEER LEFT", "STRAIGHT", "STEER RIGHT"]
        curr_action = brain_data.get("last_action", 1)
        chip_gap = 4
        chip_w = (content_w - chip_gap * 2) // 3
        chip_h = 22

        font_chip = FontManager.get(9, bold=True)
        for i, act_lbl in enumerate(action_names):
            chip_x = self.rect.x + pad_x + i * (chip_w + chip_gap)
            chip_rect = pygame.Rect(chip_x, curr_y, chip_w, chip_h)
            is_active = (i == curr_action)

            bg_c = (14, 52, 68) if is_active else (20, 25, 32)
            bd_c = COLOR_ACCENT if is_active else COLOR_PANEL_BORDER
            tx_c = COLOR_ACCENT if is_active else COLOR_TEXT_MUTED

            draw_panel(surface, chip_rect, bg_color=bg_c, border_color=bd_c, border_radius=2)
            txt_s = font_chip.render(act_lbl, True, tx_c)
            surface.blit(txt_s, (chip_rect.centerx - txt_s.get_width() // 2, chip_rect.centery - txt_s.get_height() // 2))

        curr_y += chip_h + 10

        # ----------------------------------------------------
        # 4. SENSOR PERCEPTION (7 Rays)
        # ----------------------------------------------------
        if curr_y + 80 < self.rect.bottom:
            draw_sec_header("AI SENSOR PERCEPTION (7 RAYS)")

            sensor_labels = ["-75°", "-45°", "-20°", "0°", "+20°", "+45°", "+75°"]
            font_s_lbl = FontManager.get(9, bold=False)
            font_s_val = FontManager.get(9, bold=True)
            bar_max_w = content_w - 68

            for i in range(min(len(sensor_readings), len(sensor_labels))):
                val = sensor_readings[i]
                lbl = sensor_labels[i]

                # Label
                l_surf = font_s_lbl.render(lbl, True, COLOR_TEXT_MUTED)
                surface.blit(l_surf, (self.rect.x + pad_x, curr_y))

                # Bar background
                bx = self.rect.x + pad_x + 32
                by = curr_y + 2
                bw = int(val * bar_max_w)
                bh = 6

                pygame.draw.rect(surface, (24, 28, 36), pygame.Rect(bx, by, bar_max_w, bh), border_radius=1)

                # Colored fill
                col = (0, 210, 240) if val > 0.55 else (COLOR_WARNING if val > 0.25 else COLOR_DANGER)
                if bw > 0:
                    pygame.draw.rect(surface, col, pygame.Rect(bx, by, bw, bh), border_radius=1)

                # Value text
                v_surf = font_s_val.render(f"{val:.2f}", True, COLOR_TEXT_SECONDARY)
                surface.blit(v_surf, (bx + bar_max_w + 6, curr_y - 1))

                curr_y += 12
