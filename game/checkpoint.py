"""
Checkpoint system for tracking race progress, laps, and preventing reward exploits.
"""

import math
from typing import List, Tuple
import pygame
from config import COLOR_CHECKPOINT, COLOR_CHECKPOINT_INACTIVE, COLOR_ACCENT


def lines_intersect(p1: pygame.math.Vector2, p2: pygame.math.Vector2,
                    p3: pygame.math.Vector2, p4: pygame.math.Vector2) -> bool:
    """
    Checks if line segment p1-p2 intersects line segment p3-p4.
    """
    d = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x)
    if abs(d) < 1e-9:
        return False

    u = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / d
    v = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / d

    return (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0)


class Checkpoint:
    """Represents a single gate across the track."""

    def __init__(self, index: int, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.index: int = index
        self.p1: pygame.math.Vector2 = pygame.math.Vector2(p1)
        self.p2: pygame.math.Vector2 = pygame.math.Vector2(p2)
        self.center: pygame.math.Vector2 = (self.p1 + self.p2) * 0.5

    def check_crossing(self, prev_pos: pygame.math.Vector2, curr_pos: pygame.math.Vector2) -> bool:
        """Determines if the vehicle crossed this checkpoint line."""
        return lines_intersect(prev_pos, curr_pos, self.p1, self.p2)

    def draw(self, surface: pygame.Surface, is_next: bool = False, is_start_finish: bool = False) -> None:
        """Draws checkpoint gate line."""
        if is_start_finish:
            # High-visibility checkered / gold line for start/finish
            color = (255, 215, 0) if is_next else (200, 200, 200)
            pygame.draw.line(surface, color, (int(self.p1.x), int(self.p1.y)), (int(self.p2.x), int(self.p2.y)), 3)
        elif is_next:
            # Active glowing cyan for immediate target
            pygame.draw.line(surface, COLOR_CHECKPOINT, (int(self.p1.x), int(self.p1.y)), (int(self.p2.x), int(self.p2.y)), 2)
            # Small center target node
            pygame.draw.circle(surface, COLOR_ACCENT, (int(self.center.x), int(self.center.y)), 3)
        else:
            # Inactive faint guide
            pygame.draw.line(surface, COLOR_CHECKPOINT_INACTIVE, (int(self.p1.x), int(self.p1.y)), (int(self.p2.x), int(self.p2.y)), 1)


class CheckpointManager:
    """Manages the full sequence of track checkpoints."""

    def __init__(self, gates: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        self.checkpoints: List[Checkpoint] = [
            Checkpoint(i, gate[0], gate[1]) for i, gate in enumerate(gates)
        ]
        self.total_count: int = len(self.checkpoints)

    def get_checkpoint(self, index: int) -> Checkpoint:
        """Returns checkpoint at specified wrapped index."""
        return self.checkpoints[index % self.total_count]

    def distance_to_checkpoint(self, pos: pygame.math.Vector2, index: int) -> float:
        """Euclidean distance from a point to checkpoint center."""
        target = self.get_checkpoint(index)
        return (pos - target.center).length()

    def draw(self, surface: pygame.Surface, active_index: int = -1) -> None:
        """Draws all checkpoints."""
        for cp in self.checkpoints:
            is_start = (cp.index == 0)
            is_next = (cp.index == active_index)
            cp.draw(surface, is_next=is_next, is_start_finish=is_start)
