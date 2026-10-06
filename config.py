"""
NeuroDrive Configuration Module
Centralized parameters for physics, sensors, NEAT, rendering, and UI styling.
"""

from pathlib import Path
from typing import List, Tuple

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
DATA_DIR: Path = BASE_DIR / "data"
TRACKS_DIR: Path = DATA_DIR / "tracks"
NEAT_CONFIG_PATH: Path = BASE_DIR / "config-feedforward.txt"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best_agent.pkl"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# Display & Window Settings
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
TARGET_FPS: int = 60
WINDOW_TITLE: str = "NEURODRIVE — Autonomous Racing Intelligence"

# High-DPI / Alternative 1080p preset
SUPPORTED_RESOLUTIONS: List[Tuple[int, int]] = [
    (1280, 720),
    (1600, 900),
    (1920, 1080),
]

# Vehicle Physics
CAR_WIDTH: float = 14.0             # Scaled sleek profile for clean multi-car racing
CAR_LENGTH: float = 27.0            # Sleek aerodynamic chassis length
CAR_MAX_SPEED: float = 8.5          # Scaled units (~150-170 km/h display)
CAR_ACCELERATION: float = 0.25      # Forward drive rate
CAR_BRAKE_DECEL: float = 0.40       # Braking rate
CAR_FRICTION: float = 0.06          # Natural rolling resistance
CAR_STEER_SPEED: float = 4.2        # Turning speed (degrees per frame at speed)
CAR_MIN_TURN_SPEED: float = 0.8     # Minimum speed needed to steer effectively

# Sensor Ray-Casting Configuration
# 7 sensors: Far Left (-75°), Left (-45°), Slight Left (-20°), Forward (0°),
# Slight Right (+20°), Right (+45°), Far Right (+75°)
SENSOR_ANGLES: List[float] = [-75.0, -45.0, -20.0, 0.0, 20.0, 45.0, 75.0]
SENSOR_MAX_DISTANCE: float = 230.0   # Maximum perception range in pixels
SENSOR_RAY_COUNT: int = len(SENSOR_ANGLES)

# Fitness & Reward Weights
FITNESS_STEP_SURVIVAL: float = 0.04   # Tiny incentive to stay alive
FITNESS_SPEED_MULT: float = 0.08      # Reward for velocity along track
FITNESS_CHECKPOINT_REWARD: float = 100.0  # Sequential checkpoint bonus
FITNESS_LAP_BONUS: float = 1200.0     # Full lap completion bonus
FITNESS_CRASH_PENALTY: float = -35.0  # Penalty deducted on wall collision
FITNESS_IDLE_TIMEOUT_FRAMES: int = 140 # Kill agent if no new checkpoint reached in N frames

# Simulation & Training Parameters
DEFAULT_POPULATION_SIZE: int = 35
SPEED_MULTIPLIERS: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
MAX_GENERATION_DURATION_SECS: float = 45.0 # Max generation duration in real-time seconds

# Color Palette — "Autonomous Driving Research Lab + F1 Telemetry"
COLOR_BG: Tuple[int, int, int] = (13, 17, 23)           # Deep charcoal background
COLOR_PANEL_BG: Tuple[int, int, int] = (22, 27, 34)     # Card panel background
COLOR_PANEL_BORDER: Tuple[int, int, int] = (48, 54, 61) # Subtle border
COLOR_ACCENT: Tuple[int, int, int] = (0, 240, 255)      # Cyber Cyan primary accent
COLOR_ACCENT_GLOW: Tuple[int, int, int] = (0, 180, 220) # Accent glow
COLOR_ACCENT_DIM: Tuple[int, int, int] = (16, 75, 95)   # Dimmed accent

COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (240, 246, 252) # Crisp white/light
COLOR_TEXT_SECONDARY: Tuple[int, int, int] = (139, 148, 158) # Slate secondary
COLOR_TEXT_MUTED: Tuple[int, int, int] = (80, 90, 105)     # Muted text

COLOR_SUCCESS: Tuple[int, int, int] = (63, 185, 80)     # Neon green (Alive / Online)
COLOR_WARNING: Tuple[int, int, int] = (210, 153, 34)    # Amber
COLOR_DANGER: Tuple[int, int, int] = (248, 81, 73)      # Crimson red (Crashed / Alert)

COLOR_TRACK_ASPHALT: Tuple[int, int, int] = (20, 24, 30)       # Rich deep asphalt
COLOR_TRACK_WALL: Tuple[int, int, int] = (58, 72, 92)          # Defined circuit barrier outline
COLOR_TRACK_KERB_RED: Tuple[int, int, int] = (195, 42, 45)     # High-contrast racing kerb red
COLOR_TRACK_KERB_WHITE: Tuple[int, int, int] = (230, 235, 242) # High-contrast racing kerb white
COLOR_TRACK_LINE: Tuple[int, int, int] = (42, 52, 65)          # Subtle dashed centerline
COLOR_GRASS: Tuple[int, int, int] = (11, 19, 16)
COLOR_CHECKPOINT: Tuple[int, int, int] = (0, 240, 255)         # Active target gate
COLOR_CHECKPOINT_INACTIVE: Tuple[int, int, int] = (24, 38, 52) # Faint non-intrusive guide gates
