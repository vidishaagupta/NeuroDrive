"""
Fitness Function Specification and Calculations for NeuroDrive.

FORMULATION:
The fitness function evaluates an autonomous agent's ability to safely,
rapidly, and accurately navigate a racing circuit without human intervention.

Fitness Components:
1. Checkpoint Progression (+100.0 pts per sequential checkpoint)
   - Rewards forward movement through ordered gates.
   - Enforces strict sequential order (0 -> 1 -> 2 -> ... -> N-1 -> 0).
   - Prevents looping/circling exploits.

2. Full Lap Completion (+1200.0 pts per lap)
   - Major reward milestone for successfully completing an entire circuit.

3. Speed Bonus (+ speed * 0.08 per frame)
   - Incentivizes higher velocities over timid crawling.

4. Step Survival Incentive (+0.04 per frame)
   - Minor incentive for maintaining track presence.

5. Collision Penalty (-35.0 pts)
   - Deducted upon impact with inner or outer circuit walls.

6. Idle / Reverse Stagnation Termination
   - If an agent fails to reach the next target checkpoint within 140 frames,
     it is disqualified and terminated to prevent stalling.
"""

from config import (
    FITNESS_CHECKPOINT_REWARD,
    FITNESS_CRASH_PENALTY,
    FITNESS_IDLE_TIMEOUT_FRAMES,
    FITNESS_LAP_BONUS,
    FITNESS_SPEED_MULT,
    FITNESS_STEP_SURVIVAL,
)


def compute_agent_fitness(
    checkpoints_passed: int,
    laps_completed: int,
    total_speed_accumulated: float,
    survival_frames: int,
    crashed: bool,
) -> float:
    """
    Computes cumulative fitness score for an agent based on full telemetry.
    """
    score = (
        (survival_frames * FITNESS_STEP_SURVIVAL)
        + (total_speed_accumulated * FITNESS_SPEED_MULT)
        + (checkpoints_passed * FITNESS_CHECKPOINT_REWARD)
        + (laps_completed * FITNESS_LAP_BONUS)
    )

    if crashed:
        score += FITNESS_CRASH_PENALTY

    return max(0.0, score)
