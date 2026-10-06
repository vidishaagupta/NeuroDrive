"""
Model persistence module for NeuroDrive.
Saves and loads evolved NEAT genomes, generation checkpoints, and training telemetry.
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import neat
from config import DEFAULT_MODEL_PATH


def save_best_agent(
    genome: neat.DefaultGenome,
    config: neat.Config,
    generation: int,
    fitness: float,
    file_path: Path = DEFAULT_MODEL_PATH,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Serializes the highest-fitness genome and training metadata to disk.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "genome": genome,
            "config": config,
            "generation": generation,
            "fitness": fitness,
            "timestamp": time.time(),
            "metadata": extra_metadata or {},
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[ModelPersistence] Successfully saved best agent (Gen {generation}, Fitness {fitness:.2f}) to {file_path}")
        return True
    except Exception as e:
        print(f"[ModelPersistence] Error saving model to {file_path}: {e}")
        return False


def load_best_agent(
    file_path: Path = DEFAULT_MODEL_PATH,
) -> Optional[Tuple[neat.DefaultGenome, neat.Config, int, float, Dict[str, Any]]]:
    """
    Deserializes a saved agent.
    Returns (genome, config, generation, fitness, metadata) or None if not found/invalid.
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, "rb") as f:
            payload = pickle.load(f)

        genome = payload.get("genome")
        config = payload.get("config")
        generation = payload.get("generation", 0)
        fitness = payload.get("fitness", 0.0)
        metadata = payload.get("metadata", {})

        if genome is None:
            return None

        print(f"[ModelPersistence] Loaded agent from {file_path}: Gen {generation}, Fitness {fitness:.2f}")
        return genome, config, generation, fitness, metadata
    except Exception as e:
        print(f"[ModelPersistence] Error loading model from {file_path}: {e}")
        return None
