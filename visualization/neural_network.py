"""
Neural Network Visualization module for NeuroDrive.
Renders real evolved NEAT genome topologies:
  1. System Graphviz export (when Graphviz dot is present) with graceful fallback.
  2. Standalone Pygame interactive neural network visualization screen.
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import neat
import pygame
from config import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_DANGER,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
    OUTPUTS_DIR,
)
from ui.controls import UIButton
from ui.theme import FontManager, draw_header_banner, draw_metric_row, draw_panel


def is_graphviz_installed() -> bool:
    """Checks if system 'dot' executable is available in PATH."""
    return shutil.which("dot") is not None


def export_graphviz_network(
    genome: neat.DefaultGenome,
    config: neat.Config,
    filename_prefix: str = "neural_network",
) -> Optional[Path]:
    """
    Renders NEAT genome topology via Graphviz if binary is present.
    Gracefully returns None if dot is unavailable.
    """
    if not is_graphviz_installed():
        print("[NeuralViz] Graphviz 'dot' binary is not installed in system PATH. Skipping dot export.")
        return None

    try:
        import graphviz
        dot = graphviz.Digraph(format="png", engine="dot")
        dot.attr(bgcolor="#0D1117")
        dot.attr("node", shape="circle", style="filled", color="#30363D", fontcolor="#F0F6FC")
        dot.attr("edge", fontcolor="#8B949E")

        # Input node labels
        sensor_names = {
            -1: "Far Left (-75°)",
            -2: "Left (-45°)",
            -3: "Slight Left (-20°)",
            -4: "Forward (0°)",
            -5: "Slight Right (+20°)",
            -6: "Right (+45°)",
            -7: "Far Right (+75°)",
        }
        for k, name in sensor_names.items():
            dot.node(str(k), label=name, fillcolor="#161B22", color="#00F0FF")

        # Output node labels
        out_names = {0: "Steer Left", 1: "Straight", 2: "Steer Right"}
        for k, name in out_names.items():
            dot.node(str(k), label=name, fillcolor="#1A3B32", color="#3FB950")

        # Hidden nodes & connections
        for cg in genome.connections.values():
            if cg.enabled:
                color = "#3FB950" if cg.weight > 0 else "#F85149"
                width = str(max(0.5, min(4.0, abs(cg.weight))))
                dot.edge(str(cg.key[0]), str(cg.key[1]), color=color, penwidth=width)

        out_path = OUTPUTS_DIR / filename_prefix
        dot.render(str(out_path), cleanup=True)
        print(f"[NeuralViz] Graphviz diagram saved to {out_path}.png")
        return out_path.with_suffix(".png")
    except Exception as e:
        print(f"[NeuralViz] Graphviz rendering failed: {e}")
        return None


class NeuralNetworkViewerScreen:
    """Full-screen interactive visualizer for evolved neural network topology."""

    def __init__(
        self,
        width: int,
        height: int,
        genome: Optional[neat.DefaultGenome] = None,
        config: Optional[neat.Config] = None,
        generation: int = 0,
        fitness: float = 0.0,
        on_close: Optional[Any] = None,
    ):
        self.width: int = width
        self.height: int = height
        self.genome: Optional[neat.DefaultGenome] = genome
        self.config: Optional[neat.Config] = config
        self.generation: int = generation
        self.fitness: float = fitness
        self.on_close = on_close

        self.btn_back = UIButton(
            pygame.Rect(width - 150, 12, 130, 32),
            "BACK",
            callback=self.on_close,
            icon="←",
        )

    def set_genome(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        generation: int,
        fitness: float,
    ) -> None:
        self.genome = genome
        self.config = config
        self.generation = generation
        self.fitness = fitness

    def handle_event(self, event: pygame.event.Event) -> None:
        self.btn_back.handle_event(event)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(COLOR_BG)

        # 1. Header Banner
        draw_header_banner(
            surface,
            pygame.Rect(0, 0, self.width, 56),
            title="NEURODRIVE — NEURAL NETWORK VISUALIZER",
            subtitle="GENOME TOPOLOGY & SYNAPTIC WEIGHTS",
            system_status="GRAPHVIZ ACTIVE" if is_graphviz_installed() else "INTERACTIVE VIEWER",
            status_color=COLOR_SUCCESS if is_graphviz_installed() else COLOR_ACCENT,
            generation_badge=self.generation,
            mode_text="TOPOLOGY INSPECTOR",
        )
        self.btn_back.draw(surface)

        if not self.genome:
            font = FontManager.get(18, bold=True)
            txt = font.render("NO GENOME LOADED TO VISUALIZE", True, COLOR_TEXT_MUTED)
            surface.blit(txt, (self.width // 2 - txt.get_width() // 2, self.height // 2))
            return

        # 2. Main Visual Canvas
        canvas_rect = pygame.Rect(30, 80, self.width - 320, self.height - 110)
        draw_panel(surface, canvas_rect, bg_color=COLOR_PANEL_BG)

        # 3. Telemetry Sidebar
        sidebar_rect = pygame.Rect(self.width - 270, 80, 240, self.height - 110)
        draw_panel(surface, sidebar_rect)

        font_sec = FontManager.get(12, bold=True)
        t_surf = font_sec.render("GENOME METRICS", True, COLOR_ACCENT)
        sidebar_rect_pad = sidebar_rect.x + 14
        curr_y = sidebar_rect.y + 16
        surface.blit(t_surf, (sidebar_rect_pad, curr_y))
        curr_y += 24

        enabled_conns = [c for c in self.genome.connections.values() if c.enabled]
        hidden_nodes = [k for k in self.genome.nodes.keys() if k not in (0, 1, 2) and k >= 0]

        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "GENOME ID", str(getattr(self.genome, "key", "N/A")))
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "GENERATION", str(self.generation))
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "FITNESS", f"{self.fitness:.2f}", COLOR_SUCCESS)
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "INPUT NODES", "7")
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "HIDDEN NODES", str(len(hidden_nodes)), COLOR_ACCENT)
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "OUTPUT NODES", "3")
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "ENABLED CONNS", str(len(enabled_conns)))

        curr_y += 16
        g_status = "INSTALLED" if is_graphviz_installed() else "UNAVAILABLE"
        g_col = COLOR_SUCCESS if is_graphviz_installed() else COLOR_WARNING
        curr_y = draw_metric_row(surface, sidebar_rect_pad, curr_y, 212, "GRAPHVIZ BINARY", g_status, g_col)

        # 4. Render Topology Nodes & Connections inside Canvas
        sensor_labels = [
            (-1, "Far Left (-75°)"),
            (-2, "Left (-45°)"),
            (-3, "Slight Left (-20°)"),
            (-4, "Forward (0°)"),
            (-5, "Slight Right (+20°)"),
            (-6, "Right (+45°)"),
            (-7, "Far Right (+75°)"),
        ]
        output_labels = [
            (0, "Steer Left"),
            (1, "Straight"),
            (2, "Steer Right"),
        ]

        node_positions: Dict[int, Tuple[int, int]] = {}

        # Compute positions for Input layer
        in_x = canvas_rect.x + 130
        in_gap = (canvas_rect.height - 60) // (len(sensor_labels) + 1)
        for i, (node_id, _) in enumerate(sensor_labels):
            node_positions[node_id] = (in_x, canvas_rect.y + 40 + (i + 1) * in_gap)

        # Compute positions for Output layer
        out_x = canvas_rect.right - 140
        out_gap = (canvas_rect.height - 60) // (len(output_labels) + 1)
        for i, (node_id, _) in enumerate(output_labels):
            node_positions[node_id] = (out_x, canvas_rect.y + 60 + (i + 1) * out_gap)

        # Compute positions for Hidden layer
        if hidden_nodes:
            hid_x = canvas_rect.centerx
            hid_gap = (canvas_rect.height - 60) // (len(hidden_nodes) + 1)
            for i, node_id in enumerate(hidden_nodes):
                node_positions[node_id] = (hid_x, canvas_rect.y + 40 + (i + 1) * hid_gap)

        # Draw Connections
        for cg in self.genome.connections.values():
            if not cg.enabled:
                continue
            src, dst = cg.key
            if src in node_positions and dst in node_positions:
                p1 = node_positions[src]
                p2 = node_positions[dst]

                # Positive weights = Green, Negative = Crimson
                line_col = COLOR_SUCCESS if cg.weight > 0 else COLOR_DANGER
                thick = max(1, min(4, int(abs(cg.weight) * 1.5)))
                pygame.draw.line(surface, line_col, p1, p2, thick)

        # Draw Nodes & Labels
        font_node = FontManager.get(10, bold=True)

        # Inputs
        for node_id, label in sensor_labels:
            pos = node_positions[node_id]
            pygame.draw.circle(surface, (20, 26, 35), pos, 12)
            pygame.draw.circle(surface, COLOR_ACCENT, pos, 12, 2)
            t_s = font_node.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            surface.blit(t_s, (pos[0] - t_s.get_width() // 2, pos[1] - t_s.get_height() // 2))

            lbl_s = font_node.render(label, True, COLOR_TEXT_SECONDARY)
            surface.blit(lbl_s, (pos[0] - lbl_s.get_width() - 18, pos[1] - lbl_s.get_height() // 2))

        # Hidden
        for node_id in hidden_nodes:
            pos = node_positions[node_id]
            pygame.draw.circle(surface, (28, 36, 48), pos, 12)
            pygame.draw.circle(surface, (180, 220, 240), pos, 12, 2)
            t_s = font_node.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            surface.blit(t_s, (pos[0] - t_s.get_width() // 2, pos[1] - t_s.get_height() // 2))

        # Outputs
        for node_id, label in output_labels:
            pos = node_positions[node_id]
            pygame.draw.circle(surface, (18, 45, 32), pos, 14)
            pygame.draw.circle(surface, COLOR_SUCCESS, pos, 14, 2)
            t_s = font_node.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            surface.blit(t_s, (pos[0] - t_s.get_width() // 2, pos[1] - t_s.get_height() // 2))

            lbl_s = font_node.render(label, True, COLOR_TEXT_PRIMARY)
            surface.blit(lbl_s, (pos[0] + 20, pos[1] - lbl_s.get_height() // 2))
