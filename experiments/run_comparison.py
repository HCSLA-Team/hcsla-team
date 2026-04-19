"""
Hyperparameter Optimizer Comparison — Main Runner
==================================================
Runs all 6 optimizers + a random-search baseline on the surrogate fitness
function, each with n_evals=50 and seed=42 for reproducibility.

Results are saved to experiments/results/comparison_results.json.

Isaac Gym compatibility:
  To use real Isaac Gym rollouts, replace the ``fitness_fn`` passed to each
  optimizer with your Isaac Gym evaluator wrapper.  The evaluator must
  accept (hp_dict, optimizer_name, eval_id) and return a scalar float.

Usage:
    python experiments/run_comparison.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Ensure project root is on the path when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.evaluator import get_evaluator
from src.rl.hp_space  import decode_from_vector, random_sample

# Single line to swap between surrogate and Isaac Gym — controlled by config.yaml
evaluate = get_evaluator()

from src.optimizers.ga_optimizer  import GAOptimizer
from src.optimizers.pso_optimizer import PSOOptimizer
from src.optimizers.aco_optimizer import ACOOptimizer
from src.optimizers.hba_optimizer import HBAOptimizer
from src.optimizers.rsa_optimizer import RSAOptimizer
from src.optimizers.ssa_optimizer import SSAOptimizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EVALS   = 50
SEED      = 42
MULTI_SEED_N = 10          # number of seeds for box-plot data
RESULTS_DIR  = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = RESULTS_DIR / "comparison_results.json"


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomOptimizer:
    """
    Pure random search baseline.

    Samples n_evals hyperparameter vectors uniformly at random and
    returns the best found.  Not a subclass of BaseOptimizer to keep it
    self-contained, but exposes the same .optimize() interface.
    """

    def __init__(self, fitness_fn, n_evals: int = 50, seed: int = 42):
        self.fitness_fn     = fitness_fn
        self.n_evals        = n_evals
        self.seed           = seed
        self.optimizer_name = "Random"

    def optimize(self):
        rng        = np.random.default_rng(self.seed)
        history    = []
        best_hp    = None
        best_fit   = -np.inf
        t0         = time.perf_counter()

        for i in range(self.n_evals):
            vec = random_sample(rng=rng)
            hp  = decode_from_vector(vec)
            f   = self.fitness_fn(hp, optimizer_name=self.optimizer_name,
                                  eval_id=i)
            history.append((i, f))
            if f > best_fit:
                best_fit = f
                best_hp  = hp

        elapsed = time.perf_counter() - t0
        best_per_eval = []
        running_best  = -np.inf
        for _, f in history:
            running_best = max(running_best, f)
            best_per_eval.append(running_best)

        return {
            "optimizer":     self.optimizer_name,
            "best_fitness":  best_fit,
            "best_hp":       best_hp,
            "history":       history,
            "time_taken":    elapsed,
            "best_per_eval": best_per_eval,
            "all_fitnesses": [f for _, f in history],
        }


# ---------------------------------------------------------------------------
# Build optimizer registry
# ---------------------------------------------------------------------------

def build_optimizers(seed: int = SEED):
    """Return list of (name, optimizer_instance) tuples."""
    kwargs = dict(fitness_fn=evaluate, n_evals=N_EVALS, seed=seed)
    return [
        ("Random",         RandomOptimizer(evaluate, n_evals=N_EVALS, seed=seed)),
        ("GA",             GAOptimizer(**kwargs)),
        ("PSO",            PSOOptimizer(**kwargs)),
        ("ACO",            ACOOptimizer(**kwargs)),
        ("HBA",            HBAOptimizer(**kwargs)),
        ("RSA",            RSAOptimizer(**kwargs)),
        ("SSA",            SSAOptimizer(**kwargs)),
    ]


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_single(seed: int = SEED) -> dict:
    """Run all optimizers once with the given seed."""
    results = {}
    optimizers = build_optimizers(seed=seed)

    print("\n" + "=" * 65)
    print(f"  HCSLA-CSM HP Optimizer Comparison  |  n_evals={N_EVALS}  seed={seed}")
    print("=" * 65)

    for name, opt in optimizers:
        print(f"\n>>> Running: {name}")
        if isinstance(opt, RandomOptimizer):
            r = opt.optimize()
        else:
            opt.seed = seed
            result   = opt.optimize()
            r = {
                "optimizer":     result.optimizer,
                "best_fitness":  result.best_fitness,
                "best_hp":       result.best_hp,
                "history":       result.history,
                "time_taken":    result.time_taken,
                "best_per_eval": result.best_per_eval,
                "all_fitnesses": result.all_fitnesses,
            }

        results[name] = r
        print(f"    -> best_fitness={r['best_fitness']:.4f}  "
              f"time={r['time_taken']:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Multi-seed run (for box plots)
# ---------------------------------------------------------------------------

def run_multi_seed(n_seeds: int = MULTI_SEED_N) -> dict:
    """Run all optimizers across multiple seeds for statistical analysis."""
    print("\n" + "=" * 65)
    print(f"  Multi-seed runs ({n_seeds} seeds) for box-plot data")
    print("=" * 65)

    multi = {}
    for seed_idx in range(n_seeds):
        seed = SEED + seed_idx
        print(f"\n--- Seed {seed} ({seed_idx+1}/{n_seeds}) ---")
        single = run_single(seed=seed)
        for name, r in single.items():
            multi.setdefault(name, []).append(r["best_fitness"])

    return multi


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_summary(results: dict, multi: dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n")
    print("=" * 65)
    print(f"  {'OPTIMIZER':<18} {'BEST':>8} {'TIME(s)':>9} "
          f"{'MEAN±STD (10 seeds)':>22}")
    print("=" * 65)

    # Rank by best fitness
    ranked = sorted(results.items(),
                    key=lambda x: x[1]["best_fitness"], reverse=True)

    for rank, (name, r) in enumerate(ranked, 1):
        fits  = multi.get(name, [r["best_fitness"]])
        mean  = np.mean(fits)
        std   = np.std(fits)
        print(f"  {rank}. {name:<16} {r['best_fitness']:>8.4f} "
              f"{r['time_taken']:>9.2f}  {mean:.4f} ± {std:.4f}")

    print("=" * 65)

    best_name = ranked[0][0]
    best_val  = ranked[0][1]["best_fitness"]
    baseline  = results.get("Random", {}).get("best_fitness", 0.0)
    improve   = (best_val - baseline) / (baseline + 1e-9) * 100
    print(f"\n  Best optimizer : {best_name} ({best_val:.4f})")
    print(f"  vs Random      : {baseline:.4f}  ({improve:+.1f}% improvement)")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Primary run (seed=42)
    results = run_single(seed=SEED)

    # Multi-seed for statistics
    multi = run_multi_seed(n_seeds=MULTI_SEED_N)

    # Print summary table
    print_summary(results, multi)

    # Serialise results — convert numpy types and tuples for JSON
    def _jsonify(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    def _deep_jsonify(d):
        if isinstance(d, dict):
            return {k: _deep_jsonify(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_jsonify(i) for i in d]
        return _jsonify(d)

    payload = {
        "config": {"n_evals": N_EVALS, "seed": SEED,
                   "multi_seed_n": MULTI_SEED_N},
        "single_seed": _deep_jsonify(results),
        "multi_seed":  _deep_jsonify(multi),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Results saved -> {OUT_FILE.resolve()}")
    print(f"  Fitness log   -> experiments/fitness_log.csv\n")


if __name__ == "__main__":
    main()
