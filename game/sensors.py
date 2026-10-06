"""
Perception and ray-casting sensor system for NeuroDrive autonomous vehicles.
Provides 7-directional distance sensing with exact geometric boundary intersection.
"""

import math
from typing import List, Optional, Tuple
import pygame
from config import (
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    SENSOR_ANGLES,
    SENSOR_MAX_DISTANCE,
)


class SensorRay:
    """Represents a single directional perception ray."""

    def __init__(self, relative_angle: float):
        self.relative_angle: float = relative_angle  # Degrees relative to car heading
        self.distance: float = SENSOR_MAX_DISTANCE   # Actual distance to obstacle
        self.normalized: float = 1.0                # Normalized in [0.0, 1.0]
        self.start_pos: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.end_pos: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.hit_pos: Optional[pygame.math.Vector2] = None

    def cast(
        self,
        origin: pygame.math.Vector2,
        car_heading_deg: float,
        wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
    ) -> float:
        """
        Casts ray against all wall segments and computes closest intersection.
        Returns normalized distance [0.0, 1.0].
        """
        total_angle = math.radians(car_heading_deg + self.relative_angle)
        direction = pygame.math.Vector2(math.cos(total_angle), math.sin(total_angle))

        self.start_pos = pygame.math.Vector2(origin)
        self.end_pos = self.start_pos + direction * SENSOR_MAX_DISTANCE
        self.hit_pos = None

        min_t: float = 1.0  # Parametric distance along ray [0.0, 1.0]

        rx = self.end_pos.x - self.start_pos.x
        ry = self.end_pos.y - self.start_pos.y

        for w_start, w_end in wall_segments:
            sx = w_end.x - w_start.x
            sy = w_end.y - w_start.y

            r_cross_s = rx * sy - ry * sx
            if abs(r_cross_s) < 1e-9:
                continue

            q_minus_p_x = w_start.x - self.start_pos.x
            q_minus_p_y = w_start.y - self.start_pos.y

            t = (q_minus_p_x * sy - q_minus_p_y * sx) / r_cross_s
            u = (q_minus_p_x * ry - q_minus_p_y * rx) / r_cross_s

            if 0.0 <= t < min_t and 0.0 <= u <= 1.0:
                min_t = t

        self.normalized = max(0.0, min(1.0, min_t))
        self.distance = self.normalized * SENSOR_MAX_DISTANCE

        if min_t < 1.0:
            self.hit_pos = pygame.math.Vector2(
                self.start_pos.x + rx * min_t,
                self.start_pos.y + ry * min_t,
            )
        else:
            self.hit_pos = pygame.math.Vector2(self.end_pos)

        return self.normalized

    def draw(self, surface: pygame.Surface) -> None:
        """Renders ray line and collision endpoint with subtle cyan/white telemetry styling."""
        # Subtle cyan/white telemetry color scheme:
        # Safe distance (>0.55): Crisp cyan-white
        # Moderate proximity (0.25-0.55): Technical amber
        # Critical collision range (<0.25): High-visibility danger coral
        if self.normalized > 0.55:
            ray_color = (0, 210, 240)
            dot_color = (180, 245, 255)
        elif self.normalized > 0.25:
            ray_color = (235, 180, 45)
            dot_color = (255, 220, 120)
        else:
            ray_color = (255, 75, 75)
            dot_color = (255, 140, 140)

        target = self.hit_pos if self.hit_pos else self.end_pos

        # Draw sensor ray line
        pygame.draw.line(
            surface,
            ray_color,
            (int(self.start_pos.x), int(self.start_pos.y)),
            (int(target.x), int(target.y)),
            1,
        )

        # Draw precision collision endpoint
        if self.hit_pos:
            hx, hy = int(self.hit_pos.x), int(self.hit_pos.y)
            # Outer ring
            pygame.draw.circle(surface, ray_color, (hx, hy), 3, 1)
            # Center bright dot
            pygame.draw.circle(surface, (255, 255, 255), (hx, hy), 1)


class SensorArray:
    """Manages the 7-ray perception suite of an AI car."""

    def __init__(self):
        self.rays: List[SensorRay] = [SensorRay(deg) for deg in SENSOR_ANGLES]

    def update(
        self,
        car_pos: pygame.math.Vector2,
        car_angle: float,
        wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
    ) -> List[float]:
        """
        Updates all 7 sensors and returns normalized readings list:
        [Far Left, Left, Slight Left, Forward, Slight Right, Right, Far Right]
        """
        return [ray.cast(car_pos, car_angle, wall_segments) for ray in self.rays]

    def draw(self, surface: pygame.Surface) -> None:
        """Renders all 7 sensor rays."""
        for ray in self.rays:
            ray.draw(surface)

    def get_readings(self) -> List[float]:
        """Returns the most recent normalized readings."""
        return [ray.normalized for ray in self.rays]
