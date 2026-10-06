"""
Autonomous AI Agent module connecting NEAT neural networks to racing cars.
Handles perception feedforward, discrete decision selection, and telemetry extraction.
"""

from typing import Any, List, Optional, Tuple
import neat
import pygame
from config import COLOR_ACCENT
from game.car import Car
from game.checkpoint import CheckpointManager


class AIAgent:
    """Bridges a NEAT genome and neural network with the physical Car entity."""

    def __init__(
        self,
        genome_id: int,
        genome: neat.DefaultGenome,
        config: neat.Config,
        spawn_pos: pygame.math.Vector2,
        spawn_angle: float,
        is_best: bool = False,
    ):
        self.genome_id: int = genome_id
        self.genome: neat.DefaultGenome = genome
        self.net: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)

        car_color = COLOR_ACCENT if is_best else (0, 160, 200)
        self.car: Car = Car(
            x=spawn_pos.x,
            y=spawn_pos.y,
            angle_deg=spawn_angle,
            genome_id=genome_id,
            color=car_color,
            is_best=is_best,
        )

        self.last_inputs: List[float] = [1.0] * 7
        self.last_outputs: List[float] = [0.0, 1.0, 0.0]
        self.last_action: int = 1  # 0: Left, 1: Straight, 2: Right

    @property
    def is_alive(self) -> bool:
        return self.car.is_alive

    @property
    def fitness(self) -> float:
        return self.car.fitness

    def step(
        self,
        wall_segments: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """
        Executes one full agent cycle:
        Perceive (Sensors) -> Think (Neural Net) -> Act (Physics) -> Evaluate (Fitness).
        """
        if not self.is_alive:
            return

        # 1. Perception: Read 7 ray sensors [0.0, 1.0]
        self.last_inputs = self.car.get_sensor_inputs(wall_segments)

        # 2. Cognition: Feedforward through evolved neural network
        raw_outputs = self.net.activate(self.last_inputs)
        self.last_outputs = list(raw_outputs)

        # 3. Action selection: Argmax over 3 outputs (Left, Straight, Right)
        best_action = 0
        best_val = self.last_outputs[0]
        for idx in range(1, len(self.last_outputs)):
            if self.last_outputs[idx] > best_val:
                best_val = self.last_outputs[idx]
                best_action = idx

        self.last_action = best_action

        # 4. Movement: Apply steering and forward drive
        self.car.apply_action(self.last_action)

        # 5. Physics & Collision update
        self.car.step(wall_segments, checkpoint_manager)

        # 6. Synchronize fitness back to the NEAT genome
        self.genome.fitness = self.car.fitness

    def get_brain_summary(self) -> dict:
        """Returns topology metrics for UI display."""
        # Hidden nodes: nodes that are not input (negative IDs) and not output (0, 1, 2)
        node_keys = set(self.genome.nodes.keys())
        # In neat-python, inputs have negative keys (-1 to -7), outputs are usually 0, 1, 2
        hidden_nodes = [k for k in node_keys if k not in (0, 1, 2) and k >= 0]

        # Enabled connections
        enabled_conns = [c for c in self.genome.connections.values() if c.enabled]

        return {
            "inputs": len(self.last_inputs),
            "hidden_count": len(hidden_nodes),
            "connection_count": len(enabled_conns),
            "outputs": len(self.last_outputs),
            "last_action": self.last_action,
            "inputs_val": self.last_inputs,
            "outputs_val": self.last_outputs,
        }
