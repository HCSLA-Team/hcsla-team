"""
Easy-to-Understand Optimizer Comparison
========================================
Generates 4 simple, clear graphs comparing all optimizers vs PPO-alone
(random search baseline = running PPO with no smart HP tuning).

Figures saved to paper/figures/easy/:
  1. winner_bar.png         ? simple bar chart: who finds the best HP?
  2. reliability_chart.png  ? who is CONSISTENTLY good? (mean +- std)
  3. convergence_simple.png ? how fast does each optimizer improve?
  4. summary_scorecard.png  ? overall scorecard (rank across all metrics)

Usage:
    python experiments/easy_comparison.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_FILE = Path("experiments/results/comparison_results.json")
OUT_DIR      = Path("paper/figures/easy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
COLORS = {
    "PPO alone\n(no optimizer)": "#BDBDBD",
    "GA":  "#F44336",
    "PSO": "#2196F3",
    "ACO": "#4CAF50",
    "HBA": "#FF9800",
    "RSA": "#9C27B0",
    "SSA": "#00BCD4",
}

FULL_NAMES = {
    "PPO alone\n(no optimizer)": "PPO alone (Random HP)",
    "GA":  "Genetic Algorithm",
    "PSO": "Particle Swarm",
    "ACO": "Ant Colony",
    "HBA": "Honey Badger",
    "RSA": "Reptile Search",
    "SSA": "Squirrel Search",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "#F9F9F9",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load():
    with open(RESULTS_FILE) as f:
        d = json.load(f)
    single = d["single_seed"]
    multi  = d["multi_seed"]

    # Rename "Random" -> display label
    label = "PPO alone\n(no optimizer)"
    if "Random" in single:
        single[label] = single.pop("Random")
    if "Random" in multi:
        multi[label]  = multi.pop("Random")
    return single, multi


# ---------------------------------------------------------------------------
# Graph 1: Winner Bar Chart
# ---------------------------------------------------------------------------
def plot_winner_bar(single, multi, out_dir):
    """
    Simple horizontal bar chart ? best fitness found.
    PPO alone is the baseline; everything else shows % gain.
    """
    label     = "PPO alone\n(no optimizer)"
    baseline  = single[label]["best_fitness"]

    # Build sorted list (best first), PPO alone always last
    others = [(n, single[n]["best_fitness"])
              for n in single if n != label]
    others.sort(key=lambda x: x[1], reverse=True)
    names  = [n for n, _ in others] + [label]
    values = [v for _, v in others] + [baseline]
    colors = [COLORS.get(n, "#333") for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(names[::-1], values[::-1],
                   color=colors[::-1], edgecolor="white",
                   linewidth=1.5, height=0.55)

    # Baseline vertical line
    ax.axvline(baseline, color="#333", linewidth=2,
               linestyle="--", label=f"PPO alone = {baseline:.3f}")

    # Annotate bars
    for bar, name, val in zip(bars[::-1], names, values):
        pct = (val - baseline) / baseline * 100
        label_txt = (f"  {val:.4f}"
                     if name == "PPO alone\n(no optimizer)"
                     else f"  {val:.4f}  (+{pct:.1f}% vs PPO alone)")
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", fontsize=10,
                color="#222" if name != "PPO alone\n(no optimizer)" else "#666")

    # Gold medal on winner
    best_bar = bars[-1]   # last drawn = highest (best) after reversal
    ax.text(values[0] + 0.002,
            best_bar.get_y() + best_bar.get_height() / 2 + 0.35,
            "WINNER WINNER", fontsize=11, color="#F44336", fontweight="bold")

    ax.set_xlabel("Best Fitness Score Found  (higher = better PPO training)", fontsize=12)
    ax.set_title("Which Optimizer Finds the Best PPO Hyperparameters?\n"
                 "(each bar = best result from 50 evaluations, seed=42)",
                 pad=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0.55, max(values) + 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "1_winner_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Graph 2: Reliability Chart (mean +- std over 10 seeds)
# ---------------------------------------------------------------------------
def plot_reliability(single, multi, out_dir):
    """
    Dot-and-whisker chart showing mean +- std across 10 seeds.
    Answers: 'which optimizer is reliably good, not just lucky once?'
    """
    label    = "PPO alone\n(no optimizer)"
    baseline = np.mean(multi[label])

    names  = [n for n in single if n != label]
    names.sort(key=lambda n: np.mean(multi[n]), reverse=True)
    names += [label]

    means  = [np.mean(multi[n]) for n in names]
    stds   = [np.std(multi[n])  for n in names]
    colors = [COLORS.get(n, "#333") for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))

    y_pos = range(len(names))

    for i, (n, m, s, c) in enumerate(zip(names, means, stds, colors)):
        ax.barh(i, m, color=c, alpha=0.75, height=0.5,
                edgecolor="white", linewidth=1.2)
        ax.errorbar(m, i, xerr=s, fmt="none",
                    ecolor="#333", elinewidth=2, capsize=6, capthick=2)
        ax.scatter(m, i, color=c, s=120, zorder=5,
                   edgecolors="white", linewidth=1.5)

    ax.axvline(baseline, color="#555", linewidth=2,
               linestyle="--", label=f"PPO alone mean = {baseline:.3f}")

    # Shade "better than PPO" region
    ax.axvspan(baseline, ax.get_xlim()[1] if ax.get_xlim()[1] > baseline else baseline + 0.1,
               alpha=0.06, color="green", label="Better than PPO alone")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([FULL_NAMES.get(n, n) for n in names], fontsize=11)
    ax.set_xlabel("Average Fitness over 10 Seeds  (bar = mean, whiskers = +-1 std)",
                  fontsize=11)
    ax.set_title("Reliability: Which Optimizer is CONSISTENTLY Good?\n"
                 "(not just lucky on one run ? averaged over 10 random seeds)",
                 pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.55, max(means) + max(stds) + 0.06)

    # Annotate mean +- std
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(0.555, i, f"{m:.3f} +- {s:.3f}",
                va="center", fontsize=9, color="#444")

    path = out_dir / "2_reliability_chart.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Graph 3: Convergence (how fast does each improve?)
# ---------------------------------------------------------------------------
def plot_convergence_simple(single, out_dir):
    """
    Clean convergence: best-fitness-so-far vs evaluation number.
    Shaded region shows where smart optimizers beat PPO alone.
    """
    label    = "PPO alone\n(no optimizer)"
    baseline = single[label]["best_per_eval"]

    fig, ax  = plt.subplots(figsize=(11, 6))

    # Shade PPO-alone region
    xs = list(range(1, len(baseline) + 1))
    ax.fill_between(xs, 0, baseline,
                    alpha=0.12, color="#999", label="_nolegend_")

    # PPO alone line
    ax.plot(xs, baseline,
            color=COLORS[label], linewidth=2.5,
            linestyle="--", label="PPO alone (no HP optimizer)",
            zorder=3)

    lw_map = {"GA": 2.5, "PSO": 2.5, "ACO": 2.5,
              "HBA": 2.0, "RSA": 2.0, "SSA": 2.0}

    for name in [n for n in single if n != label]:
        bpe = single[name]["best_per_eval"]
        xs2 = list(range(1, len(bpe) + 1))
        ax.plot(xs2, bpe,
                color=COLORS.get(name, "#333"),
                linewidth=lw_map.get(name, 2.0),
                label=FULL_NAMES.get(name, name),
                marker="o", markersize=4,
                markevery=max(1, len(xs2) // 8))

    ax.set_xlabel("Number of Fitness Evaluations (= PPO training runs tried)", fontsize=12)
    ax.set_ylabel("Best Fitness Found So Far", fontsize=12)
    ax.set_title("How Fast Does Each Optimizer Improve?\n"
                 "(lines above the dashed line = better than PPO alone at that point)",
                 pad=12)
    ax.legend(fontsize=10, loc="lower right",
              framealpha=0.9, edgecolor="#ccc")
    ax.set_xlim(1, max(len(single[n]["best_per_eval"])
                       for n in single))
    ax.set_ylim(0.5, None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation arrow at end
    best_name = max((n for n in single if n != label),
                    key=lambda n: single[n]["best_per_eval"][-1])
    best_val  = single[best_name]["best_per_eval"][-1]
    n_evals   = len(single[best_name]["best_per_eval"])
    ax.annotate(f"Best: {FULL_NAMES.get(best_name, best_name)}\n{best_val:.4f}",
                xy=(n_evals, best_val),
                xytext=(n_evals - 12, best_val - 0.04),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#333"),
                color="#333")

    path = out_dir / "3_convergence_simple.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Graph 4: Summary Scorecard
# ---------------------------------------------------------------------------
def plot_scorecard(single, multi, out_dir):
    """
    Table-style scorecard: each optimizer rated on 4 criteria.
    Green = good, red = bad.  Easy to read at a glance.
    """
    label    = "PPO alone\n(no optimizer)"
    names    = [n for n in single if n != label]
    baseline = single[label]["best_fitness"]
    base_mean = np.mean(multi[label])

    # Compute 4 scores (0-10) for each optimizer
    best_fits  = np.array([single[n]["best_fitness"]  for n in names])
    means      = np.array([np.mean(multi[n])          for n in names])
    stds       = np.array([np.std(multi[n])           for n in names])
    times      = np.array([single[n]["time_taken"]    for n in names])

    def norm(arr, higher_better=True):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr) * 5
        s = (arr - mn) / (mx - mn) * 10
        return s if higher_better else 10 - s

    score_peak    = norm(best_fits,  higher_better=True)
    score_mean    = norm(means,      higher_better=True)
    score_stable  = norm(stds,       higher_better=False)   # lower std = better
    score_speed   = norm(times,      higher_better=False)   # lower time = better

    # Overall = weighted average
    overall = (0.35 * score_peak + 0.35 * score_mean +
               0.20 * score_stable + 0.10 * score_speed)

    # Sort by overall
    order   = np.argsort(overall)[::-1]
    names_s = [names[i] for i in order]
    scores  = {
        "Peak\nFitness":   score_peak[order],
        "Avg\nFitness":    score_mean[order],
        "Consistency\n(low std)": score_stable[order],
        "Speed":           score_speed[order],
        "OVERALL\nSCORE":  overall[order],
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(-0.5, len(scores) - 0.5)
    ax.set_ylim(-0.5, len(names_s) - 0.5)
    ax.axis("off")

    cols    = list(scores.keys())
    n_rows  = len(names_s)
    n_cols  = len(cols)

    # Column headers
    for j, col in enumerate(cols):
        weight = "bold" if col.startswith("OVERALL") else "normal"
        ax.text(j, n_rows - 0.1, col, ha="center", va="bottom",
                fontsize=11, fontweight=weight, color="#333")

    # Row labels + cells
    for i, name in enumerate(names_s):
        row = n_rows - 1 - i
        # Rank badge
        rank_col = "#FFD700" if i == 0 else ("#C0C0C0" if i == 1 else
                   "#CD7F32" if i == 2 else "#EEE")
        badge = FancyBboxPatch((-0.9, row - 0.35), 0.4, 0.7,
                               boxstyle="round,pad=0.05",
                               facecolor=rank_col, edgecolor="white")
        ax.add_patch(badge)
        ax.text(-0.68, row, f"#{i+1}", ha="center", va="center",
                fontsize=10, fontweight="bold")

        # Optimizer name
        disp_name = FULL_NAMES.get(name, name).replace("\n", " ")
        ax.text(-0.2, row, disp_name, ha="left", va="center",
                fontsize=11, color=COLORS.get(name, "#333"),
                fontweight="bold" if i == 0 else "normal")

        for j, col in enumerate(cols):
            val   = scores[col][i]
            # Color: green (high score) to red (low score)
            r     = max(0, min(1, (10 - val) / 10))
            g     = max(0, min(1, val / 10))
            color = (r * 0.85, g * 0.75, 0.15)

            rect = FancyBboxPatch((j - 0.45, row - 0.38), 0.9, 0.76,
                                  boxstyle="round,pad=0.05",
                                  facecolor=(*color, 0.25),
                                  edgecolor=(*color, 0.7),
                                  linewidth=1.5)
            ax.add_patch(rect)

            stars = "?" * int(round(val / 2))
            ax.text(j, row + 0.08, f"{val:.1f}/10",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if col.startswith("OVERALL") else "normal",
                    color="#222")
            ax.text(j, row - 0.22, stars,
                    ha="center", va="center", fontsize=7,
                    color=(*color, 0.9))

    ax.set_title(
        "Overall Scorecard: PPO Hyperparameter Optimizers\n"
        "Criteria: Peak Fitness (35%) ? Avg Fitness (35%) ? "
        "Consistency (20%) ? Speed (10%)",
        fontsize=13, pad=18)

    # Legend for PPO alone
    ax.text(0.5, -0.55,
            f"PPO alone (no optimizer): best={baseline:.4f}  "
            f"mean={base_mean:.4f}  ? all optimizers scored relative to each other",
            ha="center", transform=ax.transAxes,
            fontsize=9, color="#666",
            style="italic")

    path = out_dir / "4_summary_scorecard.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_simple_summary(single, multi):
    label     = "PPO alone\n(no optimizer)"
    baseline  = single[label]["best_fitness"]
    base_mean = np.mean(multi[label])

    names = [n for n in single if n != label]
    names.sort(key=lambda n: single[n]["best_fitness"], reverse=True)

    print("\n" + "=" * 62)
    print("  EASY COMPARISON ? PPO HP Optimizer Results")
    print("=" * 62)
    print(f"  PPO alone (no optimizer): {baseline:.4f}  "
          f"(mean={base_mean:.4f})")
    print("-" * 62)
    print(f"  {'RANK':<5} {'OPTIMIZER':<18} {'BEST':>7} {'IMPROVEMENT':>13} "
          f"{'RELIABLE?':>11}")
    print("-" * 62)

    medals = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
    for i, name in enumerate(names):
        best  = single[name]["best_fitness"]
        mean  = np.mean(multi[name])
        std   = np.std(multi[name])
        pct   = (best - baseline) / baseline * 100
        rel   = "YES" if std < 0.03 else "variable"
        medal = medals[i] if i < len(medals) else f"{i+1} "
        dname = FULL_NAMES.get(name, name).replace("\n", " ")
        print(f"  {medal}  {dname:<17} {best:>7.4f}  "
              f"{pct:>+10.1f}%  {rel:>11}")

    print("-" * 62)
    winner = names[0]
    wbest  = single[winner]["best_fitness"]
    print(f"\n  WINNER: {FULL_NAMES.get(winner, winner).replace(chr(10),' ')}")
    print(f"  Best fitness: {wbest:.4f}  "
          f"({(wbest-baseline)/baseline*100:+.1f}% vs PPO alone)")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 62)
    print("  HCSLA-CSM ? Easy Optimizer Comparison")
    print("=" * 62)

    single, multi = load()
    print_simple_summary(single, multi)

    print(f"  Generating easy graphs -> {OUT_DIR.resolve()}\n")
    plot_winner_bar(single, multi, OUT_DIR)
    plot_reliability(single, multi, OUT_DIR)
    plot_convergence_simple(single, OUT_DIR)
    plot_scorecard(single, multi, OUT_DIR)

    print("\n  Done! Open the graphs:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"    start {f}")
    print()


if __name__ == "__main__":
    main()
