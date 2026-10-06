"""
Training Analytics module using Matplotlib.
Generates genuine publication-quality telemetry charts from real simulation data.
"""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for clean background rendering
import matplotlib.pyplot as plt
from ai.evolution import GenerationStats
from config import OUTPUTS_DIR


def generate_analytics_plots(
    stats: GenerationStats,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Renders 4 genuine training telemetry charts and saves to outputs/:
      1. Best Fitness vs Generation
      2. Average Fitness vs Generation
      3. Track Progress (%) vs Generation
      4. Survival Time (s) vs Generation
    """
    if not stats.generations:
        print("[Analytics] No training data available yet to plot.")
        return None

    save_file = output_path or (OUTPUTS_DIR / "training_analytics.png")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    gens = stats.generations

    # Professional Dark Research Lab Style
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    fig.patch.set_facecolor("#0D1117")

    titles = [
        ("Best Fitness vs Generation", stats.best_fitness, "#00F0FF", "Fitness Score"),
        ("Average Fitness vs Generation", stats.avg_fitness, "#3FB950", "Fitness Score"),
        ("Track Progress vs Generation", stats.max_progress_pct, "#D29922", "Progress (%)"),
        ("Survival Time vs Generation", stats.survival_times, "#A371F7", "Time (seconds)"),
    ]

    for ax, (title, data, color, ylabel) in zip(axes.flat, titles):
        ax.set_facecolor("#161B22")
        ax.plot(gens, data, color=color, marker="o", linewidth=2.0, markersize=4, label=ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold", color="#F0F6FC", pad=8)
        ax.set_xlabel("Generation", fontsize=10, color="#8B949E")
        ax.set_ylabel(ylabel, fontsize=10, color="#8B949E")
        ax.grid(True, linestyle="--", alpha=0.25, color="#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

        # Highlight final value
        if len(data) > 0:
            ax.annotate(
                f"{data[-1]:.1f}",
                xy=(gens[-1], data[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontweight="bold",
                fontsize=9,
            )

    plt.suptitle("NEURODRIVE — TRAINING TELEMETRY & EVOLUTION METRICS", fontsize=14, fontweight="bold", color="#F0F6FC", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(save_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[Analytics] Generated real analytics charts at {save_file}")
    return save_file
