"""
Evolution manager handling NEAT population lifecycle, speciation,
and generation-level telemetry aggregation.
"""

from typing import Any, Dict, List, Optional, Tuple
import neat
from config import NEAT_CONFIG_PATH


class GenerationStats:
    """Historical records of real training progression."""

    def __init__(self):
        self.generations: List[int] = []
        self.best_fitness: List[float] = []
        self.avg_fitness: List[float] = []
        self.max_progress_pct: List[float] = []
        self.survival_times: List[float] = []
        self.best_speeds: List[float] = []

    def record(
        self,
        generation: int,
        best_fit: float,
        avg_fit: float,
        max_prog: float,
        survival_time: float,
        best_speed: float,
    ) -> None:
        """Appends authentic metric record from generation completion."""
        self.generations.append(generation)
        self.best_fitness.append(round(best_fit, 2))
        self.avg_fitness.append(round(avg_fit, 2))
        self.max_progress_pct.append(round(max_prog, 1))
        self.survival_times.append(round(survival_time, 2))
        self.best_speeds.append(round(best_speed, 1))

    def get_latest(self) -> Dict[str, Any]:
        """Returns the most recent generation metrics."""
        if not self.generations:
            return {
                "generation": 0,
                "best_fitness": 0.0,
                "avg_fitness": 0.0,
                "max_progress": 0.0,
                "survival_time": 0.0,
                "best_speed": 0.0,
            }
        return {
            "generation": self.generations[-1],
            "best_fitness": self.best_fitness[-1],
            "avg_fitness": self.avg_fitness[-1],
            "max_progress": self.max_progress_pct[-1],
            "survival_time": self.survival_times[-1],
            "best_speed": self.best_speeds[-1],
        }


def create_neat_population(config_path: Optional[str] = None) -> Tuple[neat.Population, neat.Config]:
    """
    Initializes a NEAT population from configuration file.
    """
    cfg_file = config_path or str(NEAT_CONFIG_PATH)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_file,
    )
    pop = neat.Population(config)
    return pop, config
