"""
Collision detection routines for vehicle boundaries against circuit walls.
"""

import math
from typing import List, Tuple
import pygame
from config import CAR_LENGTH, CAR_WIDTH


def line_intersects_line(p1: pygame.math.Vector2, p2: pygame.math.Vector2,
                         p3: pygame.math.Vector2, p4: pygame.math.Vector2) -> bool:
    """Exact 2D line-segment intersection test."""
    d = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x)
    if abs(d) < 1e-9:
        return False

    u = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / d
    v = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / d

    return (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0)


def get_car_corners(
    center: pygame.math.Vector2,
    angle_deg: float,
    length: float = CAR_LENGTH,
    width: float = CAR_WIDTH,
) -> List[pygame.math.Vector2]:
    """Computes the 4 world-space corner points of the vehicle."""
    rad = math.radians(angle_deg)
    fwd = pygame.math.Vector2(math.cos(rad), math.sin(rad))
    right = pygame.math.Vector2(-math.sin(rad), math.cos(rad))

    half_l = length * 0.5
    half_w = width * 0.5

    # 4 corners: Front-Left, Front-Right, Rear-Right, Rear-Left
    return [
        center + fwd * half_l - right * half_w,
        center + fwd * half_l + right * half_w,
        center - fwd * half_l + right * half_w,
        center - fwd * half_l - right * half_w,
    ]


def check_vehicle_wall_collision(
    corners: List[pygame.math.Vector2],
    wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
) -> bool:
    """
    Checks if any of the 4 vehicle edges intersect any track wall segment.
    """
    num_corners = len(corners)
    for i in range(num_corners):
        c1 = corners[i]
        c2 = corners[(i + 1) % num_corners]

        for w1, w2 in wall_segments:
            if line_intersects_line(c1, c2, w1, w2):
                return True

    return False
