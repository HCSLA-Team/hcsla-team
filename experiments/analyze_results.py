"""
Results Analysis & Visualization
==================================
Generates 4 publication-quality figures from comparison_results.json:

  1. convergence_curves.png   ? eval# vs best-fitness-so-far per optimizer
  2. final_performance_boxplot.png ? distribution of final fitness over 10 seeds
  3. time_vs_performance.png  ? scatter: wall-time vs best fitness
  4. ablation_bar.png         ? % improvement of each optimizer vs PPO baseline

All figures are saved to paper/figures/.

Usage:
    python experiments/analyze_results.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless ? no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FILE = Path("experiments/results/comparison_results.json")
FIGURES_DIR  = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette ? one distinct color per optimizer
# ---------------------------------------------------------------------------
COLORS = {
    "Random": "#9E9E9E",
    "GA":     "#F44336",
    "PSO":    "#2196F3",
    "ACO":    "#4CAF50",
    "HBA":    "#FF9800",
    "RSA":    "#9C27B0",
    "SSA":    "#00BCD4",
}

MARKERS = {
    "Random": "o", "GA": "s", "PSO": "^",
    "ACO": "D", "HBA": "P", "RSA": "X", "SSA": "*",
}

# Consistent order (best?worst will be determined from data, but keep legend stable)
OPTIMIZER_ORDER = ["Random", "GA", "PSO", "ACO", "HBA", "RSA", "SSA"]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results() -> dict:
    """Load and return the JSON results dict."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results not found: {RESULTS_FILE}\n"
            "Run `python experiments/run_comparison.py` first."
        )
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(single: dict, out_dir: Path) -> None:
    """
    Plot best-fitness-so-far vs evaluation number for each optimizer.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in OPTIMIZER_ORDER:
        if name not in single:
            continue
        bpe = single[name]["best_per_eval"]
        xs  = list(range(1, len(bpe) + 1))
        ax.plot(xs, bpe,
                label=name,
                color=COLORS.get(name, "#333"),
                marker=MARKERS.get(name, "o"),
                markevery=max(1, len(xs) // 10),
                linewidth=2.0,
                markersize=7)

    ax.set_xlabel("Number of Evaluations", fontsize=13)
    ax.set_ylabel("Best Fitness (so far)", fontsize=13)
    ax.set_title("Convergence Curves ? HP Optimizer Comparison\n"
                 "(HCSLA-CSM Quadruped Locomotion, Surrogate Fitness)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(1, None)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "convergence_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Plot 2: Final performance box plot (10 seeds)
# ---------------------------------------------------------------------------

def plot_boxplot(multi: dict, out_dir: Path) -> None:
    """
    Box plot of final fitness values across 10 random seeds per optimizer.
    """
    names  = [n for n in OPTIMIZER_ORDER if n in multi]
    data   = [multi[n] for n in names]
    colors = [COLORS.get(n, "#333") for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))

    bp = ax.boxplot(data,
                    patch_artist=True,
                    notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=5, alpha=0.6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(data, colors), 1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=col, edgecolors="white",
                   s=40, zorder=3, alpha=0.9)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Final Best Fitness", fontsize=13)
    ax.set_title("Final Performance Distribution (10 Seeds)\n"
                 "HCSLA-CSM Quadruped Locomotion HP Search",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate medians
    for i, vals in enumerate(data, 1):
        med = np.median(vals)
        ax.text(i, med + 0.005, f"{med:.3f}",
                ha="center", va="bottom", fontsize=9, color="black")

    path = out_dir / "final_performance_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Plot 3: Time vs Performance scatter
# ---------------------------------------------------------------------------

def plot_time_vs_performance(single: dict, multi: dict, out_dir: Path) -> None:
    """
    Scatter plot: wall-clock time vs best fitness.
    Marker size proportional to fitness std across seeds (larger = more variance).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name in OPTIMIZER_ORDER:
        if name not in single:
            continue
        t       = single[name]["time_taken"]
        f       = single[name]["best_fitness"]
        std     = np.std(multi.get(name, [f]))
        size    = max(80, std * 3000)  # scale marker by variance

        ax.scatter(t, f,
                   color=COLORS.get(name, "#333"),
                   marker=MARKERS.get(name, "o"),
                   s=size,
                   label=name,
                   edgecolors="white",
                   linewidths=1.2,
                   alpha=0.9,
                   zorder=4)
        ax.annotate(f"  {name}",
                    xy=(t, f),
                    fontsize=10,
                    color=COLORS.get(name, "#333"),
                    va="center")

    ax.set_xlabel("Wall-Clock Time (seconds)", fontsize=13)
    ax.set_ylabel("Best Fitness Found", fontsize=13)
    ax.set_title("Time vs Performance\n"
                 "(marker size proportional to fitness std across 10 seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "time_vs_performance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Plot 4: Ablation bar chart (% improvement vs PPO baseline)
# ---------------------------------------------------------------------------

def plot_ablation(single: dict, multi: dict, out_dir: Path) -> None:
    """
    Horizontal bar chart showing % improvement of each optimizer over
    the PPO random-search baseline, with error bars from multi-seed runs.

    PPO baseline = random search best fitness (seed=42).
    """
    baseline_name = "Random"
    baseline_fit  = single.get(baseline_name, {}).get("best_fitness", 0.01)
    baseline_vals = multi.get(baseline_name, [baseline_fit])

    names   = [n for n in OPTIMIZER_ORDER if n != baseline_name and n in single]
    improv  = []
    err     = []

    for name in names:
        vals = multi.get(name, [single[name]["best_fitness"]])
        pct  = [(v - baseline_fit) / (abs(baseline_fit) + 1e-9) * 100
                for v in vals]
        improv.append(np.mean(pct))
        err.append(np.std(pct))

    # Sort descending
    order  = np.argsort(improv)[::-1]
    names  = [names[i] for i in order]
    improv = [improv[i] for i in order]
    err    = [err[i] for i in order]
    colors = [COLORS.get(n, "#333") for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(names, improv, xerr=err,
                   color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2,
                   error_kw=dict(elinewidth=1.5, capsize=4, ecolor="gray"))

    # Annotate values
    for bar, val, e in zip(bars, improv, err):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_width() + e + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.1f}%",
                va="center", fontsize=10)

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("% Improvement over PPO + Random Search Baseline", fontsize=12)
    ax.set_title("Ablation: PPO Baseline vs PPO + Optimizer\n"
                 "(mean +- std over 10 seeds, HCSLA-CSM surrogate)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Add baseline annotation
    ax.text(0.02, -0.08,
            f"Baseline (Random Search): {baseline_fit:.4f}  "
            f"[mean={np.mean(baseline_vals):.4f} +- {np.std(baseline_vals):.4f}]",
            transform=ax.transAxes, fontsize=9, color="gray")

    path = out_dir / "ablation_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_analysis_summary(single: dict, multi: dict) -> None:
    """Print a detailed statistical comparison table."""
    print("\n" + "=" * 72)
    print(f"  {'OPTIMIZER':<12} {'BEST':>8} {'MEAN':>8} {'STD':>8} "
          f"{'MIN':>8} {'MAX':>8} {'TIME':>8}")
    print("=" * 72)

    ranked = sorted(
        [(n, single[n]) for n in OPTIMIZER_ORDER if n in single],
        key=lambda x: x[1]["best_fitness"], reverse=True
    )

    baseline = single.get("Random", {}).get("best_fitness", 0.01)

    for rank, (name, r) in enumerate(ranked, 1):
        vals = multi.get(name, [r["best_fitness"]])
        mean = np.mean(vals)
        std  = np.std(vals)
        mn   = np.min(vals)
        mx   = np.max(vals)
        t    = r["time_taken"]
        tag  = " *" if rank == 1 else ""
        print(f"  {rank}. {name:<10} {r['best_fitness']:>8.4f} "
              f"{mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f} "
              f"{t:>7.2f}s{tag}")

    print("=" * 72)

    best_name, best_r = ranked[0]
    improv = (best_r["best_fitness"] - baseline) / (abs(baseline) + 1e-9) * 100
    print(f"\n  Winner: {best_name}  ({best_r['best_fitness']:.4f})")
    print(f"  Improvement over Random baseline: {improv:+.1f}%")

    print("\n  Best hyperparameters found:")
    for k, v in best_r["best_hp"].items():
        val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"    {k:<18}: {val_str}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("  HCSLA-CSM HP Optimizer ? Results Analysis")
    print("=" * 72)

    data   = load_results()
    single = data["single_seed"]
    multi  = data["multi_seed"]

    print_analysis_summary(single, multi)

    print(f"\n  Generating figures -> {FIGURES_DIR.resolve()}")
    plot_convergence(single, FIGURES_DIR)
    plot_boxplot(multi, FIGURES_DIR)
    plot_time_vs_performance(single, multi, FIGURES_DIR)
    plot_ablation(single, multi, FIGURES_DIR)

    print("\n  All 4 figures generated successfully.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
