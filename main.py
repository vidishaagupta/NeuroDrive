"""
NeuroDrive — Autonomous Racing AI
Main entry point orchestrating CLI options, landing screen, training session,
inference session, and neural network topology visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pygame
from ai.model import load_best_agent
from ai.training import InferenceSession, TrainingSession
from config import (
    DEFAULT_MODEL_PATH,
    TARGET_FPS,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    WINDOW_WIDTH,
)
from game.track import Track
from ui.dashboard import DashboardUI
from ui.landing import LandingScreen
from ui.theme import FontManager, draw_panel
from visualization.analytics import generate_analytics_plots
from visualization.neural_network import NeuralNetworkViewerScreen, export_graphviz_network


class NeuroDriveApp:
    """Master application controller managing transitions between modes."""

    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(WINDOW_TITLE)

        self.clock = pygame.time.Clock()
        self.running = True

        # State: "landing", "training", "inference", "network_viewer", "empty_state", "about"
        self.current_state = "landing"

        # Game world objects
        self.track = Track(offset_x=225, offset_y=60, scale=1.0)

        # Screens & Sessions
        self.landing_screen = LandingScreen(
            width=self.width,
            height=self.height,
            on_start_training=self.start_training,
            on_load_best=self.load_best_agent_mode,
            on_view_network=self.open_network_viewer,
            on_view_about=self.open_about,
        )

        self.dashboard = DashboardUI(
            width=self.width,
            height=self.height,
            on_toggle_pause=self._on_dashboard_pause,
            on_reset_training=self._on_dashboard_reset,
            on_switch_mode=self._on_dashboard_switch_mode,
            on_speed_change=self._on_dashboard_speed_change,
            on_view_network=self.open_network_viewer,
            on_export_analytics=self.export_analytics,
            on_back_to_menu=self.return_to_landing,
        )

        self.training_session: Optional[TrainingSession] = None
        self.inference_session: Optional[InferenceSession] = None
        self.network_viewer: Optional[NeuralNetworkViewerScreen] = None

    def _on_dashboard_pause(self) -> None:
        if self.current_state == "training" and self.training_session:
            self.training_session.toggle_pause()
        elif self.current_state == "inference" and self.inference_session:
            self.inference_session.is_paused = not self.inference_session.is_paused

    def _on_dashboard_reset(self) -> None:
        if self.current_state == "training" and self.training_session:
            self.training_session.reset()

    def _on_dashboard_switch_mode(self, mode: str) -> None:
        if mode == "INFERENCE":
            self.load_best_agent_mode()
        else:
            self.start_training()

    def _on_dashboard_speed_change(self, speed_val: float) -> None:
        if self.current_state == "training" and self.training_session:
            self.training_session.set_speed(speed_val)
        elif self.current_state == "inference" and self.inference_session:
            self.inference_session.simulation_speed = speed_val

    def start_training(self) -> None:
        """Transitions into NEAT population training mode."""
        self.current_state = "training"
        self.training_session = TrainingSession(
            surface=self.screen,
            clock=self.clock,
            track=self.track,
            dashboard=self.dashboard,
            on_exit=self.return_to_landing,
            on_view_network=self._view_active_network,
        )
        self.training_session.run_training_loop()

    def load_best_agent_mode(self) -> None:
        """Loads serialized best model or prompts empty state."""
        model_data = load_best_agent(DEFAULT_MODEL_PATH)
        if model_data is None:
            self.current_state = "empty_state"
            return

        self.current_state = "inference"
        self.inference_session = InferenceSession(
            surface=self.screen,
            clock=self.clock,
            track=self.track,
            dashboard=self.dashboard,
            on_exit=self.return_to_landing,
            model_data=model_data,
        )
        self.inference_session.run()

    def _view_active_network(self, genome, config, gen, fit) -> None:
        self.network_viewer = NeuralNetworkViewerScreen(
            width=self.width,
            height=self.height,
            genome=genome,
            config=config,
            generation=gen,
            fitness=fit,
            on_close=self._close_network_viewer,
        )
        self.current_state = "network_viewer"

    def open_network_viewer(self) -> None:
        """Opens neural network visualizer screen or modal inspector."""
        genome, config, gen, fit = None, None, 0, 0.0
        if self.training_session and self.training_session.all_time_best_genome:
            genome = self.training_session.all_time_best_genome
            config = self.training_session.config
            gen = self.training_session.current_generation
            fit = self.training_session.all_time_best_fitness
        else:
            model_data = load_best_agent(DEFAULT_MODEL_PATH)
            if model_data:
                genome, config, gen, fit, _ = model_data

        if genome is None:
            self.current_state = "empty_state"
            return

        viewing = True

        def close_cb():
            nonlocal viewing
            viewing = False

        viewer = NeuralNetworkViewerScreen(
            width=self.width,
            height=self.height,
            genome=genome,
            config=config,
            generation=gen,
            fitness=fit,
            on_close=close_cb,
        )

        # Run viewer until user clicks BACK or ESC
        while viewing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    viewing = False
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    viewing = False
                    break
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    viewer.width = self.width
                    viewer.height = self.height
                    viewer.btn_back.rect.x = self.width - 150

                viewer.handle_event(event)

            viewer.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def open_about(self) -> None:
        self.current_state = "about"

    def export_analytics(self) -> None:
        """Renders high-res Matplotlib telemetry figures."""
        if self.training_session and self.training_session.stats.generations:
            generate_analytics_plots(self.training_session.stats)
        else:
            print("[NeuroDrive] No training telemetry recorded yet to export.")

    def return_to_landing(self) -> None:
        if self.training_session:
            self.training_session.request_stop()
        if self.inference_session:
            self.inference_session.should_stop = True
        self.current_state = "landing"

    def run(self) -> None:
        """Primary application event and frame loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.landing_screen.resize(self.width, self.height)
                    self.dashboard.resize(self.width, self.height)
                    if self.network_viewer:
                        self.network_viewer.width = self.width
                        self.network_viewer.height = self.height

                # Route events based on current view
                if self.current_state == "landing":
                    self.landing_screen.handle_event(event)
                elif self.current_state == "network_viewer" and self.network_viewer:
                    self.network_viewer.handle_event(event)
                elif self.current_state == "empty_state":
                    self._handle_empty_state_event(event)
                elif self.current_state == "about":
                    self._handle_about_event(event)

            # Draw current screen
            if self.current_state == "landing":
                self.landing_screen.draw(self.screen)
            elif self.current_state == "network_viewer" and self.network_viewer:
                self.network_viewer.draw(self.screen)
            elif self.current_state == "empty_state":
                self._draw_empty_state()
            elif self.current_state == "about":
                self._draw_about_screen()

            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

        pygame.quit()
        sys.exit(0)

    def _draw_empty_state(self) -> None:
        self.screen.fill((13, 17, 23))
        center_x = self.width // 2
        center_y = self.height // 2

        card_rect = pygame.Rect(center_x - 220, center_y - 120, 440, 240)
        draw_panel(self.screen, card_rect, bg_color=(22, 27, 34), border_color=(48, 54, 61), border_radius=6)

        font_head = FontManager.get(18, bold=True)
        font_sub = FontManager.get(12, bold=False)
        font_btn = FontManager.get(12, bold=True)

        t_s = font_head.render("NO TRAINED AGENT AVAILABLE", True, (240, 246, 252))
        s_s = font_sub.render("Train your first autonomous driver to generate a genome.", True, (139, 148, 158))

        self.screen.blit(t_s, (center_x - t_s.get_width() // 2, center_y - 70))
        self.screen.blit(s_s, (center_x - s_s.get_width() // 2, center_y - 35))

        # Action button: START TRAINING
        btn_train_rect = pygame.Rect(center_x - 140, center_y + 10, 280, 40)
        draw_panel(self.screen, btn_train_rect, bg_color=(16, 65, 80), border_color=(0, 240, 255), border_radius=4)
        b_s = font_btn.render("▶  START TRAINING NOW", True, (240, 246, 252))
        self.screen.blit(b_s, (center_x - b_s.get_width() // 2, center_y + 22))

        # Back button
        btn_back_rect = pygame.Rect(center_x - 140, center_y + 60, 280, 32)
        draw_panel(self.screen, btn_back_rect, bg_color=(26, 32, 40), border_color=(48, 54, 61), border_radius=4)
        bk_s = font_btn.render("←  BACK TO MAIN MENU", True, (139, 148, 158))
        self.screen.blit(bk_s, (center_x - bk_s.get_width() // 2, center_y + 68))

    def _handle_empty_state_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            center_x = self.width // 2
            center_y = self.height // 2
            btn_train_rect = pygame.Rect(center_x - 140, center_y + 10, 280, 40)
            btn_back_rect = pygame.Rect(center_x - 140, center_y + 60, 280, 32)

            if btn_train_rect.collidepoint(event.pos):
                self.start_training()
            elif btn_back_rect.collidepoint(event.pos):
                self.current_state = "landing"

    def _draw_about_screen(self) -> None:
        self.screen.fill((13, 17, 23))
        center_x = self.width // 2
        card_rect = pygame.Rect(center_x - 340, 60, 680, 580)
        draw_panel(self.screen, card_rect, bg_color=(22, 27, 34), border_color=(48, 54, 61), border_radius=6)

        font_head = FontManager.get(20, bold=True)
        font_sub = FontManager.get(12, bold=True)
        font_body = FontManager.get(11, bold=False)

        h_s = font_head.render("NEURODRIVE — SYSTEM ARCHITECTURE", True, (240, 246, 252))
        self.screen.blit(h_s, (card_rect.x + 24, card_rect.y + 24))

        sections = [
            ("1. NEAT Neuroevolution", "Utilizes NeuroEvolution of Augmenting Topologies. Instead of fixed backprop weights, genomes evolve both connection weights and hidden topology structures (nodes and edges) simultaneously."),
            ("2. Perception Array (7 Sensors)", "Casts 7 directional rays (-75°, -45°, -20°, 0°, +20°, +45°, +75°) calculating normalized track distance [0.0, 1.0]. Zero hardcoded racing trajectories."),
            ("3. Action Space (3 Outputs)", "Outputs: [Steer Left, Straight, Steer Right]. The highest neural activation controls the vehicle steering angle in combination with deterministic 2D physics."),
            ("4. Anti-Exploit Fitness Formulation", "Rewards sequential checkpoint traversal (+100) and lap completions (+1200) with speed incentives. Detects stalling and wall impacts with immediate termination."),
            ("5. Telemetry & Analytics", "Captures genuine per-generation metrics (Best & Avg Fitness, Track Progress, Speeds). High-resolution charts exported directly via Matplotlib."),
        ]

        curr_y = card_rect.y + 70
        for title, desc in sections:
            t_s = font_sub.render(title, True, (0, 240, 255))
            self.screen.blit(t_s, (card_rect.x + 24, curr_y))
            curr_y += 20

            words = desc.split(" ")
            line = ""
            for w in words:
                if len(line) + len(w) < 85:
                    line += w + " "
                else:
                    self.screen.blit(font_body.render(line.strip(), True, (139, 148, 158)), (card_rect.x + 24, curr_y))
                    curr_y += 16
                    line = w + " "
            if line:
                self.screen.blit(font_body.render(line.strip(), True, (139, 148, 158)), (card_rect.x + 24, curr_y))
                curr_y += 24

        # Back Button
        btn_back = pygame.Rect(card_rect.x + 24, card_rect.bottom - 48, 140, 32)
        draw_panel(self.screen, btn_back, bg_color=(16, 65, 80), border_color=(0, 240, 255), border_radius=4)
        b_s = font_sub.render("←  MAIN MENU", True, (240, 246, 252))
        self.screen.blit(b_s, (btn_back.centerx - b_s.get_width() // 2, btn_back.centery - b_s.get_height() // 2))

    def _handle_about_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            center_x = self.width // 2
            card_rect = pygame.Rect(center_x - 340, 60, 680, 580)
            btn_back = pygame.Rect(card_rect.x + 24, card_rect.bottom - 48, 140, 32)
            if btn_back.collidepoint(event.pos):
                self.current_state = "landing"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroDrive — Autonomous Racing AI")
    parser.add_argument("--mode", type=str, default="landing", choices=["landing", "train", "inference", "visualize"], help="Launch mode")
    parser.add_argument("--width", type=int, default=WINDOW_WIDTH, help="Window display width")
    parser.add_argument("--height", type=int, default=WINDOW_HEIGHT, help="Window display height")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    app = NeuroDriveApp(width=args.width, height=args.height)

    if args.mode == "train":
        app.start_training()
    elif args.mode == "inference":
        app.load_best_agent_mode()
    elif args.mode == "visualize":
        app.open_network_viewer()

    app.run()


if __name__ == "__main__":
    main()
