"""
Car entity class representing an autonomous racing vehicle in NeuroDrive.
Includes modern vector graphics, sensor management, checkpoint navigation,
and telemetry tracking.
"""

import math
from typing import List, Optional, Tuple
import pygame
from config import (
    CAR_LENGTH,
    CAR_WIDTH,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_PANEL_BG,
    COLOR_SUCCESS,
    FITNESS_CHECKPOINT_REWARD,
    FITNESS_CRASH_PENALTY,
    FITNESS_IDLE_TIMEOUT_FRAMES,
    FITNESS_LAP_BONUS,
    FITNESS_SPEED_MULT,
    FITNESS_STEP_SURVIVAL,
)
from game.checkpoint import CheckpointManager
from game.collision import check_vehicle_wall_collision, get_car_corners
from game.physics import VehiclePhysics
from game.sensors import SensorArray


class Car:
    """Autonomous racing car agent."""

    def __init__(
        self,
        x: float,
        y: float,
        angle_deg: float = 0.0,
        genome_id: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 220, 240),
        is_best: bool = False,
    ):
        self.genome_id: Optional[int] = genome_id
        self.physics: VehiclePhysics = VehiclePhysics(x, y, angle_deg)
        self.sensors: SensorArray = SensorArray()

        self.spawn_pos: pygame.math.Vector2 = pygame.math.Vector2(x, y)
        self.spawn_angle: float = angle_deg

        self.is_alive: bool = True
        self.crashed: bool = False
        self.fitness: float = 0.0

        self.current_checkpoint_idx: int = 1
        self.total_checkpoints_passed: int = 0
        self.laps_completed: int = 0
        self.frames_since_checkpoint: int = 0
        self.survival_frames: int = 0
        self.distance_traveled: float = 0.0
        self.last_action: int = 1  # 0: Left, 1: Straight, 2: Right

        self.color: Tuple[int, int, int] = color
        self.is_best: bool = is_best

        self.trajectory: List[Tuple[float, float]] = []
        self.max_trajectory_length: int = 80

    def get_sensor_inputs(
        self, wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]]
    ) -> List[float]:
        """Reads 7 normalized distance sensors."""
        return self.sensors.update(self.physics.position, self.physics.angle, wall_segments)

    def apply_action(self, action: int) -> None:
        """Applies neural network decision to vehicle dynamics."""
        if not self.is_alive:
            return
        self.last_action = action
        # Continuous throttle driving forward, with directional steering
        self.physics.apply_action(action, throttle=1.0)

    def step(
        self,
        wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """
        Executes one physics frame: movement, collision detection,
        checkpoint evaluation, and fitness calculation.
        """
        if not self.is_alive:
            return

        prev_pos = pygame.math.Vector2(self.physics.position)
        self.physics.update()
        curr_pos = self.physics.position

        step_dist = (curr_pos - prev_pos).length()
        self.distance_traveled += step_dist
        self.survival_frames += 1

        # Record trajectory breadcrumbs
        if len(self.trajectory) == 0 or (curr_pos - pygame.math.Vector2(self.trajectory[-1])).length() > 6:
            self.trajectory.append((curr_pos.x, curr_pos.y))
            if len(self.trajectory) > self.max_trajectory_length:
                self.trajectory.pop(0)

        # Baseline step survival reward + speed bonus
        self.fitness += FITNESS_STEP_SURVIVAL + (self.physics.speed * FITNESS_SPEED_MULT)

        # 1. Checkpoint evaluation
        target_cp = checkpoint_manager.get_checkpoint(self.current_checkpoint_idx)
        if target_cp.check_crossing(prev_pos, curr_pos):
            # Checkpoint achieved
            self.total_checkpoints_passed += 1
            self.current_checkpoint_idx = (self.current_checkpoint_idx + 1) % checkpoint_manager.total_count
            self.frames_since_checkpoint = 0
            self.fitness += FITNESS_CHECKPOINT_REWARD

            if self.current_checkpoint_idx == 0:
                self.laps_completed += 1
                self.fitness += FITNESS_LAP_BONUS
        else:
            self.frames_since_checkpoint += 1
            # Proximity progress reward: reward getting closer to the checkpoint center
            dist_to_cp = (curr_pos - target_cp.center).length()
            progress_nudge = max(0.0, (200.0 - dist_to_cp) / 200.0) * 0.1
            self.fitness += progress_nudge

            # Stagnation / reverse timeout check
            if self.frames_since_checkpoint > FITNESS_IDLE_TIMEOUT_FRAMES:
                self.is_alive = False
                self.crashed = False  # Timed out, didn't hit wall
                return

        # 2. Collision evaluation against track boundaries
        corners = get_car_corners(self.physics.position, self.physics.angle)
        if check_vehicle_wall_collision(corners, wall_segments):
            self.is_alive = False
            self.crashed = True
            self.fitness += FITNESS_CRASH_PENALTY
            # Ensure fitness doesn't fall below minimum
            if self.fitness < 0.0:
                self.fitness = 0.0

    def draw(
        self,
        surface: pygame.Surface,
        show_sensors: bool = False,
        show_trajectory: bool = False,
    ) -> None:
        """Renders the racing car, trajectory, and perception sensors."""
        # 1. Render trajectory trail if enabled
        if show_trajectory and len(self.trajectory) > 2:
            trail_color = (*self.color[:3], 90) if len(self.color) == 3 else self.color
            pygame.draw.lines(surface, (0, 180, 220), False, self.trajectory, 1)

        # 2. Render perception sensors if enabled and car is alive
        if show_sensors and self.is_alive:
            self.sensors.draw(surface)

        # 3. Render car chassis
        pos = self.physics.position
        angle_rad = math.radians(self.physics.angle)

        # Compute unit orientation vectors
        fwd = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad))
        right = pygame.math.Vector2(-math.sin(angle_rad), math.cos(angle_rad))

        l_half = CAR_LENGTH * 0.5
        w_half = CAR_WIDTH * 0.5

        # Subtle shadow underneath
        shadow_offset = pygame.math.Vector2(2, 3)
        shadow_corners = [
            (pos + shadow_offset + fwd * l_half - right * w_half),
            (pos + shadow_offset + fwd * l_half + right * w_half),
            (pos + shadow_offset - fwd * l_half + right * w_half),
            (pos + shadow_offset - fwd * l_half - right * w_half),
        ]
        pygame.draw.polygon(
            surface,
            (5, 7, 10),
            [(int(p.x), int(p.y)) for p in shadow_corners],
        )

        # Main body polygon
        corners = [
            pos + fwd * l_half - right * w_half,
            pos + fwd * l_half + right * w_half,
            pos - fwd * l_half + right * w_half,
            pos - fwd * l_half - right * w_half,
        ]
        body_pts = [(int(p.x), int(p.y)) for p in corners]

        # Select color palette based on state and genome differentiation
        if not self.is_alive:
            # Crashed / dead: low-visibility muted dark graphite
            car_body_color = (25, 29, 36)
            trim_color = (42, 48, 58)
            cockpit_color = (14, 16, 20)
        elif self.is_best:
            # Active champion / leader: prominent electric cyan with crisp white trim
            car_body_color = COLOR_ACCENT
            trim_color = (255, 255, 255)
            cockpit_color = (12, 24, 32)
        else:
            # Subtle team liveries for population agents to reduce clutter
            team_idx = (self.genome_id or 0) % 4
            team_palettes = [
                ((42, 56, 72), (70, 92, 116)),    # Titanium Slate
                ((32, 48, 68), (58, 82, 110)),    # Midnight Navy
                ((40, 48, 58), (66, 76, 90)),     # Stealth Carbon
                ((28, 54, 62), (50, 88, 100)),    # Deep Teal
            ]
            car_body_color, trim_color = team_palettes[team_idx]
            cockpit_color = (14, 18, 24)

        # Subtle underglow outline for best agent only
        if self.is_best and self.is_alive:
            glow_corners = [
                pos + fwd * (l_half + 3.0) - right * (w_half + 2.5),
                pos + fwd * (l_half + 3.0) + right * (w_half + 2.5),
                pos - fwd * (l_half + 3.0) + right * (w_half + 2.5),
                pos - fwd * (l_half + 3.0) - right * (w_half + 2.5),
            ]
            pygame.draw.polygon(
                surface,
                (0, 160, 190),
                [(int(p.x), int(p.y)) for p in glow_corners],
                1,
            )

        # Draw main chassis
        pygame.draw.polygon(surface, car_body_color, body_pts)
        pygame.draw.polygon(surface, trim_color, body_pts, 1)

        # Windshield / Cockpit
        cockpit_fwd = pos + fwd * (l_half * 0.22)
        cockpit_back = pos - fwd * (l_half * 0.28)
        cockpit_w = w_half * 0.58
        cockpit_pts = [
            cockpit_fwd - right * cockpit_w,
            cockpit_fwd + right * cockpit_w,
            cockpit_back + right * (cockpit_w * 0.8),
            cockpit_back - right * (cockpit_w * 0.8),
        ]
        pygame.draw.polygon(
            surface,
            cockpit_color,
            [(int(p.x), int(p.y)) for p in cockpit_pts],
        )

        # Front Headlights
        if self.is_alive:
            hl_l = pos + fwd * (l_half * 0.92) - right * (w_half * 0.65)
            hl_r = pos + fwd * (l_half * 0.92) + right * (w_half * 0.65)
            light_col = (255, 255, 255) if self.is_best else (180, 215, 235)
            pygame.draw.circle(surface, light_col, (int(hl_l.x), int(hl_l.y)), 2)
            pygame.draw.circle(surface, light_col, (int(hl_r.x), int(hl_r.y)), 2)

            # Rear taillights (crimson)
            tl_l = pos - fwd * (l_half * 0.92) - right * (w_half * 0.65)
            tl_r = pos - fwd * (l_half * 0.92) + right * (w_half * 0.65)
            pygame.draw.circle(surface, COLOR_DANGER, (int(tl_l.x), int(tl_l.y)), 1)
            pygame.draw.circle(surface, COLOR_DANGER, (int(tl_r.x), int(tl_r.y)), 1)

        # Floating "BEST AGENT" indicator for current best car
        if self.is_best and self.is_alive:
            tag_x = int(pos.x)
            tag_y = int(pos.y - 18)
            bw, bh = 42, 13
            tag_rect = pygame.Rect(tag_x - bw // 2, tag_y - bh // 2, bw, bh)
            pygame.draw.rect(surface, (10, 22, 30), tag_rect, border_radius=2)
            pygame.draw.rect(surface, COLOR_ACCENT, tag_rect, width=1, border_radius=2)

            from ui.theme import FontManager
            font_tag = FontManager.get(9, bold=True)
            txt_s = font_tag.render("★ BEST", True, COLOR_ACCENT)
            surface.blit(txt_s, (tag_rect.centerx - txt_s.get_width() // 2, tag_rect.centery - txt_s.get_height() // 2))
