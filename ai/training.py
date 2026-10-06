"""
Training and Inference Session Orchestrators for NeuroDrive.
Integrates NEAT generational population evaluation, real-time Pygame rendering,
interactive speed controls, collision events, and model persistence.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import neat
import pygame
from ai.agent import AIAgent
from ai.evolution import GenerationStats, create_neat_population
from ai.model import load_best_agent, save_best_agent
from config import (
    CAR_MAX_SPEED,
    DEFAULT_MODEL_PATH,
    MAX_GENERATION_DURATION_SECS,
    NEAT_CONFIG_PATH,
    TARGET_FPS,
)
from game.track import Track
from ui.dashboard import DashboardUI
from visualization.analytics import generate_analytics_plots
from visualization.neural_network import export_graphviz_network


class TrainingSession:
    """Orchestrates live NEAT population training with interactive Pygame GUI."""

    def __init__(
        self,
        surface: pygame.Surface,
        clock: pygame.time.Clock,
        track: Track,
        dashboard: DashboardUI,
        on_exit: Callable[[], None],
        on_view_network: Callable[[neat.DefaultGenome, neat.Config, int, float], None],
    ):
        self.surface: pygame.Surface = surface
        self.clock: pygame.time.Clock = clock
        self.track: Track = track
        self.dashboard: DashboardUI = dashboard
        self.on_exit: Callable[[], None] = on_exit
        self.on_view_network = on_view_network

        self.pop: Optional[neat.Population] = None
        self.config: Optional[neat.Config] = None
        self.stats: GenerationStats = GenerationStats()

        self.simulation_speed: float = 1.0
        self.is_paused: bool = False
        self.should_stop: bool = False
        self.reset_requested: bool = False

        self.current_generation: int = 0
        self.all_time_best_fitness: float = 0.0
        self.all_time_best_genome: Optional[neat.DefaultGenome] = None

        self._init_neat()

    def _init_neat(self) -> None:
        """Initializes or resets NEAT population."""
        self.pop, self.config = create_neat_population(str(NEAT_CONFIG_PATH))
        self.current_generation = 0
        self.all_time_best_fitness = 0.0
        self.all_time_best_genome = None

    def reset(self) -> None:
        """Resets the simulation and evolution history."""
        self._init_neat()
        self.stats = GenerationStats()
        self.reset_requested = True

    def set_speed(self, speed_val: float) -> None:
        self.simulation_speed = speed_val

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused

    def request_stop(self) -> None:
        self.should_stop = True

    def run_training_loop(self) -> None:
        """Runs generation by generation until user exits."""
        self.should_stop = False
        self.reset_requested = False

        while not self.should_stop:
            if self.reset_requested:
                self.reset_requested = False
                self._init_neat()

            # Run 1 generation
            try:
                self.pop.run(self._eval_generation, 1)
            except Exception as e:
                print(f"[TrainingSession] Generation aborted or interrupted: {e}")
                break

            if self.should_stop:
                break

    def _eval_generation(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
        """Evaluates all genomes in a generation simultaneously in real time."""
        self.current_generation += 1

        # Spawn AI cars for each genome
        agents: List[AIAgent] = []
        for gid, g in genomes:
            g.fitness = 0.0
            agent = AIAgent(
                genome_id=gid,
                genome=g,
                config=config,
                spawn_pos=self.track.spawn_pos,
                spawn_angle=self.track.spawn_angle,
                is_best=False,
            )
            agents.append(agent)

        gen_start_time = time.time()
        gen_duration_budget = MAX_GENERATION_DURATION_SECS

        gen_best_fitness = 0.0
        gen_max_progress = 0.0
        gen_best_speed = 0.0

        # Sub-step fractional accumulator for 0.5x speed
        speed_acc = 0.0

        while any(a.is_alive for a in agents) and not self.should_stop and not self.reset_requested:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.should_stop = True
                    self.on_exit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()
                    elif event.key == pygame.K_ESCAPE:
                        self.should_stop = True
                        self.on_exit()
                        return
                    elif event.key == pygame.K_v:
                        self.dashboard.show_vision = not self.dashboard.show_vision
                        self.dashboard.toggle_vision.state = self.dashboard.show_vision

                self.dashboard.handle_event(event)

            if self.should_stop or self.reset_requested:
                break

            # 2. Simulation stepping
            if not self.is_paused:
                steps_to_run = 0
                if self.simulation_speed >= 1.0:
                    steps_to_run = int(self.simulation_speed)
                else:
                    speed_acc += self.simulation_speed
                    if speed_acc >= 1.0:
                        steps_to_run = 1
                        speed_acc -= 1.0

                for _ in range(steps_to_run):
                    for a in agents:
                        if a.is_alive:
                            a.step(self.track.wall_segments, self.track.checkpoint_manager)
                            if a.car.crashed:
                                # Trigger visual collision alert
                                prog_pct = (a.car.total_checkpoints_passed / max(1, self.track.checkpoint_manager.total_count)) * 100.0
                                self.dashboard.crash_alert.trigger(a.fitness, prog_pct)

            # 3. Identify active best agent for HUD spotlight
            alive_agents = [a for a in agents if a.is_alive]
            leader_agent = None
            if alive_agents:
                leader_agent = max(alive_agents, key=lambda a: a.fitness)
            elif agents:
                leader_agent = max(agents, key=lambda a: a.fitness)

            # Mark leader visually
            for a in agents:
                a.car.is_best = (a is leader_agent)

            if leader_agent:
                gen_best_fitness = max(gen_best_fitness, leader_agent.fitness)
                leader_prog = (leader_agent.car.total_checkpoints_passed / max(1, self.track.checkpoint_manager.total_count)) * 100.0
                gen_max_progress = max(gen_max_progress, leader_prog)
                gen_best_speed = max(gen_best_speed, leader_agent.car.physics.speed)

            # 4. Prepare Telemetry Packets
            active_car_data: Dict[str, Any] = {
                "speed": leader_agent.car.physics.speed if leader_agent else 0.0,
                "progress_pct": (leader_agent.car.total_checkpoints_passed / max(1, self.track.checkpoint_manager.total_count)) * 100.0 if leader_agent else 0.0,
                "checkpoint_str": f"{leader_agent.car.current_checkpoint_idx}/{self.track.checkpoint_manager.total_count}" if leader_agent else "0/0",
                "fitness": leader_agent.fitness if leader_agent else 0.0,
                "survival_time": leader_agent.car.survival_frames / 60.0 if leader_agent else 0.0,
                "laps": leader_agent.car.laps_completed if leader_agent else 0,
            }

            status_str = "● PAUSED" if self.is_paused else "● EVOLVING"
            training_data: Dict[str, Any] = {
                "generation": self.current_generation,
                "alive_count": len(alive_agents),
                "population_size": len(agents),
                "best_fitness": max(self.all_time_best_fitness, gen_best_fitness),
                "avg_fitness": sum(a.fitness for a in agents) / max(1, len(agents)),
                "max_progress": gen_max_progress,
                "status_text": status_str,
            }

            brain_data = leader_agent.get_brain_summary() if leader_agent else {
                "inputs": 7, "hidden_count": 0, "connection_count": 0, "outputs": 3, "last_action": 1
            }

            sensor_readings = leader_agent.last_inputs if leader_agent else [1.0] * 7

            # 5. Render Scene
            self.dashboard.update()
            self.dashboard.draw(
                self.surface,
                track=self.track,
                is_paused=self.is_paused,
                mode_str="TRAINING MODE",
                active_car_data=active_car_data,
                training_data=training_data,
                brain_data=brain_data,
                sensor_readings=sensor_readings,
                fitness_history=self.stats.best_fitness,
                progress_history=self.stats.max_progress_pct,
            )

            # Draw track inside viewport
            active_cp = leader_agent.car.current_checkpoint_idx if leader_agent else -1
            self.track.draw(
                self.surface,
                show_checkpoints=self.dashboard.show_checkpoints,
                active_checkpoint_idx=active_cp,
            )

            # Draw agents (draw non-best first, best on top with sensors)
            for a in agents:
                if not a.car.is_best:
                    a.car.draw(
                        self.surface,
                        show_sensors=False,
                        show_trajectory=self.dashboard.show_trajectory,
                    )
            if leader_agent:
                leader_agent.car.draw(
                    self.surface,
                    show_sensors=self.dashboard.show_vision,
                    show_trajectory=self.dashboard.show_trajectory,
                )

            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

            # Check timeout
            if (time.time() - gen_start_time) > gen_duration_budget:
                print(f"[TrainingSession] Generation {self.current_generation} reached time limit.")
                break

        # Generation wrap-up
        if not self.should_stop and not self.reset_requested:
            # Commit fitness to genomes
            for a in agents:
                a.genome.fitness = a.fitness

            best_gen_agent = max(agents, key=lambda a: a.fitness)
            avg_fit = sum(a.fitness for a in agents) / max(1, len(agents))
            survival_sec = best_gen_agent.car.survival_frames / 60.0

            # Record stats
            self.stats.record(
                generation=self.current_generation,
                best_fit=best_gen_agent.fitness,
                avg_fit=avg_fit,
                max_prog=gen_max_progress,
                survival_time=survival_sec,
                best_speed=gen_best_speed * 19.5,
            )

            # Save best genome if all-time record beaten
            if best_gen_agent.fitness > self.all_time_best_fitness:
                self.all_time_best_fitness = best_gen_agent.fitness
                self.all_time_best_genome = best_gen_agent.genome

                save_best_agent(
                    genome=best_gen_agent.genome,
                    config=config,
                    generation=self.current_generation,
                    fitness=best_gen_agent.fitness,
                    file_path=DEFAULT_MODEL_PATH,
                )

                # Attempt Graphviz export in background
                export_graphviz_network(best_gen_agent.genome, config)

            print(f"[TrainingSession] Completed Gen {self.current_generation} | Best Fit: {best_gen_agent.fitness:.2f} | Avg Fit: {avg_fit:.2f} | Progress: {gen_max_progress:.1f}%")


class InferenceSession:
    """Runs a single trained autonomous vehicle using the saved best genome."""

    def __init__(
        self,
        surface: pygame.Surface,
        clock: pygame.time.Clock,
        track: Track,
        dashboard: DashboardUI,
        on_exit: Callable[[], None],
        model_data: Optional[Tuple[neat.DefaultGenome, neat.Config, int, float, Dict[str, Any]]] = None,
    ):
        self.surface: pygame.Surface = surface
        self.clock: pygame.time.Clock = clock
        self.track: Track = track
        self.dashboard: DashboardUI = dashboard
        self.on_exit: Callable[[], None] = on_exit
        self.model_data = model_data

        self.simulation_speed: float = 1.0
        self.is_paused: bool = False
        self.should_stop: bool = False

    def run(self) -> None:
        """Loads best model and executes autonomous driving demonstration."""
        model_data = self.model_data or load_best_agent(DEFAULT_MODEL_PATH)
        if model_data is None:
            print("[InferenceSession] No trained agent found to load. Please train first.")
            return

        genome, config, gen_num, trained_fitness, _ = model_data

        agent = AIAgent(
            genome_id=getattr(genome, "key", 0),
            genome=genome,
            config=config,
            spawn_pos=self.track.spawn_pos,
            spawn_angle=self.track.spawn_angle,
            is_best=True,
        )

        self.should_stop = False
        speed_acc = 0.0

        while not self.should_stop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.should_stop = True
                    self.on_exit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_ESCAPE:
                        self.should_stop = True
                        self.on_exit()
                        return

                self.dashboard.handle_event(event)

            if not self.is_paused:
                steps = int(self.simulation_speed) if self.simulation_speed >= 1.0 else 1
                for _ in range(steps):
                    if agent.is_alive:
                        agent.step(self.track.wall_segments, self.track.checkpoint_manager)
                    else:
                        # Auto respawn in inference mode to keep continuous run
                        agent.car.physics.reset(self.track.spawn_pos.x, self.track.spawn_pos.y, self.track.spawn_angle)
                        agent.car.is_alive = True
                        agent.car.crashed = False
                        agent.car.current_checkpoint_idx = 1
                        agent.car.frames_since_checkpoint = 0

            # Telemetry
            active_car_data: Dict[str, Any] = {
                "speed": agent.car.physics.speed,
                "progress_pct": (agent.car.total_checkpoints_passed / max(1, self.track.checkpoint_manager.total_count)) * 100.0,
                "checkpoint_str": f"{agent.car.current_checkpoint_idx}/{self.track.checkpoint_manager.total_count}",
                "fitness": agent.fitness,
                "survival_time": agent.car.survival_frames / 60.0,
                "laps": agent.car.laps_completed,
            }

            training_data: Dict[str, Any] = {
                "generation": gen_num,
                "alive_count": 1 if agent.is_alive else 0,
                "population_size": 1,
                "best_fitness": trained_fitness,
                "avg_fitness": trained_fitness,
                "max_progress": (agent.car.total_checkpoints_passed / max(1, self.track.checkpoint_manager.total_count)) * 100.0,
                "status_text": "● AUTONOMOUS RUN",
            }

            brain_data = agent.get_brain_summary()

            self.dashboard.update()
            self.dashboard.draw(
                self.surface,
                track=self.track,
                is_paused=self.is_paused,
                mode_str="INFERENCE MODE",
                active_car_data=active_car_data,
                training_data=training_data,
                brain_data=brain_data,
                sensor_readings=agent.last_inputs,
                fitness_history=[trained_fitness],
                progress_history=[100.0],
            )

            self.track.draw(
                self.surface,
                show_checkpoints=self.dashboard.show_checkpoints,
                active_checkpoint_idx=agent.car.current_checkpoint_idx,
            )

            agent.car.draw(
                self.surface,
                show_sensors=self.dashboard.show_vision,
                show_trajectory=self.dashboard.show_trajectory,
            )

            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
