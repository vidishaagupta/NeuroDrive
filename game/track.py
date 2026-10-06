"""
Track generation, rendering, and circuit geometry for NeuroDrive.
Constructs an FIA GP-style circuit with straightaways, sweeping turns,
chicanes, hairpin bends, kerbs, and orthogonal checkpoints.
"""

import math
from typing import List, Tuple
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_GRASS,
    COLOR_TRACK_ASPHALT,
    COLOR_TRACK_KERB_RED,
    COLOR_TRACK_KERB_WHITE,
    COLOR_TRACK_LINE,
)
from game.checkpoint import CheckpointManager


class Track:
    """Represents a racing circuit with geometric boundary segments and checkpoints."""

    def __init__(self, offset_x: float = 230.0, offset_y: float = 65.0, scale: float = 1.0):
        self.offset_x: float = offset_x
        self.offset_y: float = offset_y
        self.scale: float = scale

        self.track_width: float = 72.0 * scale

        # Base centerline control waypoints (GP-Style Circuit)
        # Clockwise progression: Start/Finish straight -> Turn 1 -> Chicane -> Hairpin -> Sweeper -> Back straight
        self.raw_centerline: List[Tuple[float, float]] = [
            (220, 510),  # 0: S/F Line
            (360, 510),  # 1: Main Straight
            (490, 510),  # 2: Straight approach
            (580, 500),  # 3: T1 entry
            (660, 450),  # 4: T1 apex
            (710, 380),  # 5: T1 curve
            (710, 300),  # 6: T1 exit
            (660, 240),  # 7: Chicane entry
            (590, 220),  # 8: Chicane kink
            (530, 180),  # 9: Infield transition
            (480, 120),  # 10: Hairpin entry
            (410, 80),   # 11: Hairpin apex
            (330, 70),   # 12: Hairpin top
            (250, 90),   # 13: Hairpin curve
            (190, 140),  # 14: Hairpin exit
            (160, 210),  # 15: Infield bend
            (150, 290),  # 16: Mid-straight
            (130, 370),  # 17: Banking entry
            (90, 440),   # 18: Banking apex
            (100, 485),  # 19: Final corner
            (150, 510),  # 20: Final exit into straight
        ]

        # Computed geometric entities
        self.centerline: List[pygame.math.Vector2] = []
        self.inner_boundary: List[pygame.math.Vector2] = []
        self.outer_boundary: List[pygame.math.Vector2] = []
        self.wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]] = []
        self.checkpoint_manager: CheckpointManager = None
        self.spawn_pos: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.spawn_angle: float = 0.0

        self._build_circuit()

    def _build_circuit(self) -> None:
        """Computes offset boundary walls, checkpoints, and wall segments."""
        # 1. Transform raw centerline with scale and offset
        self.centerline = [
            pygame.math.Vector2(x * self.scale + self.offset_x, y * self.scale + self.offset_y)
            for x, y in self.raw_centerline
        ]

        n = len(self.centerline)
        self.inner_boundary = []
        self.outer_boundary = []
        gates: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        half_w = self.track_width * 0.5

        for i in range(n):
            curr = self.centerline[i]
            prev = self.centerline[(i - 1 + n) % n]
            nxt = self.centerline[(i + 1) % n]

            # Tangent direction vector
            tangent = (nxt - prev).normalize()
            # Normal vector pointing outwards (to the right of tangent)
            normal = pygame.math.Vector2(-tangent.y, tangent.x)

            # Outer and inner points
            p_out = curr + normal * half_w
            p_in = curr - normal * half_w

            self.outer_boundary.append(p_out)
            self.inner_boundary.append(p_in)
            gates.append(((p_in.x, p_in.y), (p_out.x, p_out.y)))

        # 2. Build wall segments for ray casting and collision
        self.wall_segments = []
        for i in range(n):
            # Outer wall segments
            self.wall_segments.append((self.outer_boundary[i], self.outer_boundary[(i + 1) % n]))
            # Inner wall segments
            self.wall_segments.append((self.inner_boundary[i], self.inner_boundary[(i + 1) % n]))

        # 3. Instantiate checkpoint manager
        self.checkpoint_manager = CheckpointManager(gates)

        # 4. Set vehicle spawn parameters
        # Spawn just behind checkpoint 0 on the main straight, heading toward checkpoint 1
        p0 = self.centerline[0]
        p1 = self.centerline[1]
        spawn_dir = (p1 - p0).normalize()
        self.spawn_pos = p0 - spawn_dir * 15.0
        self.spawn_angle = math.degrees(math.atan2(spawn_dir.y, spawn_dir.x))

    def draw(
        self,
        surface: pygame.Surface,
        show_checkpoints: bool = True,
        active_checkpoint_idx: int = -1,
    ) -> None:
        """Renders circuit surface, boundaries, asphalt, kerbs, and road markings."""
        n = len(self.centerline)

        # 1. Draw track asphalt ribbon
        for i in range(n):
            poly = [
                self.outer_boundary[i],
                self.outer_boundary[(i + 1) % n],
                self.inner_boundary[(i + 1) % n],
                self.inner_boundary[i],
            ]
            pygame.draw.polygon(surface, COLOR_TRACK_ASPHALT, [(int(p.x), int(p.y)) for p in poly])

        # 2. Draw continuous subtle circuit wall barrier lines
        wall_color = (58, 72, 92)
        outer_pts = [(int(p.x), int(p.y)) for p in self.outer_boundary]
        inner_pts = [(int(p.x), int(p.y)) for p in self.inner_boundary]
        pygame.draw.lines(surface, wall_color, True, outer_pts, 1)
        pygame.draw.lines(surface, wall_color, True, inner_pts, 1)

        # 3. Draw alternating red-and-white kerbs on turns
        kerb_width = 3
        for i in range(n):
            p1_out = self.outer_boundary[i]
            p2_out = self.outer_boundary[(i + 1) % n]
            p1_in = self.inner_boundary[i]
            p2_in = self.inner_boundary[(i + 1) % n]

            kerb_color = COLOR_TRACK_KERB_RED if (i % 2 == 0) else COLOR_TRACK_KERB_WHITE

            # Outer kerb segment
            pygame.draw.line(surface, kerb_color, (int(p1_out.x), int(p1_out.y)), (int(p2_out.x), int(p2_out.y)), kerb_width)
            # Inner kerb segment
            pygame.draw.line(surface, kerb_color, (int(p1_in.x), int(p1_in.y)), (int(p2_in.x), int(p2_in.y)), kerb_width)

        # 4. Draw dashed track centerline
        for i in range(n):
            if i % 2 == 0:
                p_a = self.centerline[i]
                p_b = self.centerline[(i + 1) % n]
                pygame.draw.line(surface, COLOR_TRACK_LINE, (int(p_a.x), int(p_a.y)), (int(p_b.x), int(p_b.y)), 1)

        # 5. Draw start / finish line
        cp0 = self.checkpoint_manager.get_checkpoint(0)
        pygame.draw.line(surface, (240, 245, 250), (int(cp0.p1.x), int(cp0.p1.y)), (int(cp0.p2.x), int(cp0.p2.y)), 3)
        # Checkered gold accent mark at start/finish
        mid = cp0.center
        pygame.draw.circle(surface, (255, 215, 0), (int(mid.x), int(mid.y)), 3)

        # 5. Draw checkpoints if enabled
        if show_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.draw(surface, active_index=active_checkpoint_idx)
