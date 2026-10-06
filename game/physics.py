"""
Physics module for NeuroDrive 2D vehicle simulation.
Provides smooth, deterministic vehicle dynamics with acceleration, braking,
steering dynamics, friction, and angular kinematics.
"""

import math
from typing import Tuple
import pygame
from config import (
    CAR_ACCELERATION,
    CAR_BRAKE_DECEL,
    CAR_FRICTION,
    CAR_MAX_SPEED,
    CAR_STEER_SPEED,
    CAR_MIN_TURN_SPEED,
)


class VehiclePhysics:
    """
    Simulates 2D vehicle kinematics.
    Maintains position, velocity vector, heading angle, and speed.
    """

    def __init__(self, x: float, y: float, angle_degrees: float = 0.0):
        self.position: pygame.math.Vector2 = pygame.math.Vector2(x, y)
        self.angle: float = angle_degrees  # Heading in degrees (0 = right, 90 = down)
        self.speed: float = 0.0            # Forward scalar speed (>= 0)
        self.acceleration: float = 0.0

    def apply_action(self, steer_action: int, throttle: float = 1.0) -> None:
        """
        Applies a discrete steering command and throttle:
          steer_action: 0 = Steer Left, 1 = Straight, 2 = Steer Right
          throttle: 1.0 for forward drive, 0.0 for coasting, -1.0 for braking
        """
        # Accelerate / brake
        if throttle > 0:
            self.speed += CAR_ACCELERATION * throttle
            if self.speed > CAR_MAX_SPEED:
                self.speed = CAR_MAX_SPEED
        elif throttle < 0:
            self.speed -= CAR_BRAKE_DECEL * abs(throttle)
            if self.speed < 0.0:
                self.speed = 0.0
        else:
            # Natural rolling friction
            if self.speed > 0:
                self.speed = max(0.0, self.speed - CAR_FRICTION)

        # Steering dynamics:
        # Steering effectiveness scales with speed so stationary cars cannot spin
        if self.speed > CAR_MIN_TURN_SPEED:
            speed_factor = min(1.0, (self.speed / CAR_MAX_SPEED) ** 0.5)
            turn_amount = CAR_STEER_SPEED * speed_factor

            if steer_action == 0:    # Turn Left
                self.angle -= turn_amount
            elif steer_action == 2:  # Turn Right
                self.angle += turn_amount

        # Normalize angle to [-180, 180]
        self.angle = (self.angle + 180.0) % 360.0 - 180.0

    def update(self) -> None:
        """Integrates velocity into position using current heading."""
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        self.position.x += dx
        self.position.y += dy

    def get_forward_vector(self) -> pygame.math.Vector2:
        """Returns normalized 2D direction vector."""
        rad = math.radians(self.angle)
        return pygame.math.Vector2(math.cos(rad), math.sin(rad))

    def reset(self, x: float, y: float, angle_degrees: float = 0.0) -> None:
        """Resets physics state."""
        self.position = pygame.math.Vector2(x, y)
        self.angle = angle_degrees
        self.speed = 0.0
        self.acceleration = 0.0
