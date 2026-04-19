"""
Surrogate Fitness Function for PPO Hyperparameter Optimization
==============================================================
Simulates the reward landscape that a real PPO training run on a
quadruped locomotion task (e.g. Isaac Gym) would produce.

Landscape properties:
  - Multimodal: 2 local optima + 1 global optimum
  - Gaussian noise std=0.05 to mimic stochastic RL training variance
  - Bounded output in approximately [0, 1]
  - Sensitive to lr, clip_eps, gamma, and reward weights in realistic ways

Isaac Gym compatibility:
  The function signature ``evaluate(hp_dict, optimizer_name, eval_id)``
  is designed to be drop-in replaceable with a real Isaac Gym rollout
  evaluator.  When Isaac Gym is available, implement a subclass or
  wrapper that runs actual PPO episodes and returns cumulative reward
  normalized to [0, 1].

Logging:
  Every call is appended to experiments/fitness_log.csv with columns:
    optimizer, eval_id, fitness, lr, clip_eps, gamma, w_velocity,
    w_stability, w_energy, w_smooth, w_survival, batch_size, timestamp
"""

import os
import csv
import time
import numpy as np
from pathlib import Path

# Ensure log directory exists
LOG_PATH = Path("experiments/fitness_log.csv")
LOG_FIELDS = [
    "optimizer", "eval_id", "fitness", "timestamp",
    "lr", "clip_eps", "gamma",
    "w_velocity", "w_stability", "w_energy", "w_smooth", "w_survival",
    "batch_size",
]


def _ensure_log() -> None:
    """Create log file with header if it does not exist."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def _log_call(optimizer: str, eval_id: int, fitness: float, hp: dict) -> None:
    """Append one evaluation record to the CSV log."""
    _ensure_log()
    row = {
        "optimizer":   optimizer,
        "eval_id":     eval_id,
        "fitness":     round(fitness, 6),
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lr":          hp.get("lr", ""),
        "clip_eps":    hp.get("clip_eps", ""),
        "gamma":       hp.get("gamma", ""),
        "w_velocity":  hp.get("w_velocity", ""),
        "w_stability": hp.get("w_stability", ""),
        "w_energy":    hp.get("w_energy", ""),
        "w_smooth":    hp.get("w_smooth", ""),
        "w_survival":  hp.get("w_survival", ""),
        "batch_size":  hp.get("batch_size", ""),
    }
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Internal landscape helpers
# ---------------------------------------------------------------------------

def _lr_response(lr: float) -> float:
    """
    Learning-rate sensitivity curve.
    Peaks near lr=3e-4 (well-known sweet spot for PPO on locomotion tasks).
    Falls off steeply for very small or very large lr.
    """
    log_lr = np.log10(lr)
    # Peak at log10(3e-4) ≈ -3.52
    return np.exp(-((log_lr + 3.52) ** 2) / 0.4)


def _clip_response(clip_eps: float) -> float:
    """Clip epsilon: moderate values (0.15-0.25) work best."""
    return np.exp(-((clip_eps - 0.20) ** 2) / 0.008)


def _gamma_response(gamma: float) -> float:
    """Discount factor: high gamma (0.99) preferred for locomotion."""
    return np.exp(-((gamma - 0.99) ** 2) / 0.0002)


def _weight_interaction(hp: dict) -> float:
    """
    Reward-weight interaction term.
    The global optimum has a balanced weighting across all components.
    Simulates the multi-objective nature of the locomotion reward.
    """
    # Target weights (global optimum region)
    targets = {
        "w_velocity":  1.8,
        "w_stability": 2.5,
        "w_energy":    0.003,
        "w_smooth":    0.08,
        "w_survival":  0.5,
    }
    scales = {
        "w_velocity":  0.8,
        "w_stability": 1.5,
        "w_energy":    0.002,
        "w_smooth":    0.04,
        "w_survival":  0.3,
    }
    score = 0.0
    for key, target in targets.items():
        diff = (hp[key] - target) / scales[key]
        score += diff ** 2
    return np.exp(-score / len(targets))


def _batch_size_response(batch_size: int) -> float:
    """Larger batches slightly better for locomotion; 2048 is sweet spot."""
    mapping = {512: 0.70, 1024: 0.85, 2048: 1.00, 4096: 0.92}
    return mapping.get(int(batch_size), 0.75)


# ---------------------------------------------------------------------------
# Local optima (decoys) — encoded-space Gaussians
# ---------------------------------------------------------------------------

# Each local optimum is defined in natural HP space
_LOCAL_OPTIMA = [
    # Local optimum 1: high lr, low clip
    {
        "lr": 8e-4, "clip_eps": 0.12, "gamma": 0.97,
        "w_velocity": 2.5, "w_stability": 1.0, "w_energy": 0.005,
        "w_smooth": 0.15, "w_survival": 0.8, "batch_size": 1024,
        "peak": 0.62, "width": 1.8,
    },
    # Local optimum 2: very low lr, high gamma
    {
        "lr": 2e-5, "clip_eps": 0.35, "gamma": 0.998,
        "w_velocity": 0.8, "w_stability": 4.0, "w_energy": 0.001,
        "w_smooth": 0.03, "w_survival": 0.2, "batch_size": 4096,
        "peak": 0.55, "width": 2.0,
    },
]


def _gaussian_bump(hp: dict, optimum: dict, width: float) -> float:
    """Gaussian bump around a local optimum in natural-log HP space."""
    keys = ["lr", "clip_eps", "gamma", "w_velocity",
            "w_stability", "w_energy", "w_smooth", "w_survival"]
    # Normalise each dimension by the range
    from src.rl.hp_space import CONTINUOUS_PARAMS, LOG_PARAMS
    sq_dist = 0.0
    for key in keys:
        lo, hi = CONTINUOUS_PARAMS[key]
        if key in LOG_PARAMS:
            v = (np.log(hp[key]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            c = (np.log(optimum[key]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            v = (hp[key] - lo) / (hi - lo)
            c = (optimum[key] - lo) / (hi - lo)
        sq_dist += (v - c) ** 2
    return np.exp(-sq_dist / (2 * width ** 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def surrogate_fitness(hp: dict, noise_std: float = 0.05,
                      seed: int = None) -> float:
    """
    Evaluate the surrogate fitness for a hyperparameter configuration.

    Combines:
      1. Global optimum response (product of per-dimension curves)
      2. Two local optima (Gaussian bumps)
      3. Gaussian noise to mimic RL variance

    Args:
        hp       : Hyperparameter dict (decoded natural scale).
        noise_std: Standard deviation of evaluation noise (default 0.05).
        seed     : Optional seed for reproducible noise (None = random).

    Returns:
        Scalar fitness in approximately [0, 1].
    """
    rng = np.random.default_rng(seed)

    # --- Global optimum component ---
    global_score = (
        _lr_response(hp["lr"]) *
        _clip_response(hp["clip_eps"]) *
        _gamma_response(hp["gamma"]) *
        _weight_interaction(hp) *
        _batch_size_response(hp["batch_size"])
    )
    global_score = 0.90 * global_score  # global peak ≈ 0.90

    # --- Local optima components ---
    local_score = 0.0
    for opt in _LOCAL_OPTIMA:
        bump = _gaussian_bump(hp, opt, opt["width"])
        local_score = max(local_score, opt["peak"] * bump)

    # --- Combined (take best + small blend) ---
    base_fitness = max(global_score, local_score) + 0.05 * min(global_score, local_score)

    # --- Noise ---
    noise = rng.normal(0.0, noise_std)
    fitness = float(np.clip(base_fitness + noise, 0.0, 1.0))
    return fitness


def evaluate(hp: dict, optimizer_name: str = "unknown",
             eval_id: int = 0, noise_std: float = 0.05,
             seed: int = None) -> float:
    """
    Evaluate fitness and log the result.

    This is the primary entry point for all optimizers.

    Isaac Gym drop-in: replace the body of this function with a call to
    your Isaac Gym rollout evaluator and return normalised cumulative reward.

    Args:
        hp            : Hyperparameter dict (decoded natural scale).
        optimizer_name: Name tag written to the CSV log.
        eval_id       : Sequential evaluation counter (per optimizer run).
        noise_std     : Noise standard deviation passed to surrogate.
        seed          : Noise seed (None = stochastic, use for repeatability).

    Returns:
        Scalar fitness in [0, 1].
    """
    fitness = surrogate_fitness(hp, noise_std=noise_std, seed=seed)
    _log_call(optimizer_name, eval_id, fitness, hp)
    return fitness


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.rl.hp_space import decode_from_vector, random_sample

    rng = np.random.default_rng(42)
    print("=== Surrogate Fitness Self-Test ===")

    # Sample 5 random configs
    for i in range(5):
        vec = random_sample(rng=rng)
        hp = decode_from_vector(vec)
        f = evaluate(hp, optimizer_name="test", eval_id=i)
        print(f"  eval {i}: fitness={f:.4f}  lr={hp['lr']:.2e}  "
              f"clip={hp['clip_eps']:.3f}  gamma={hp['gamma']:.4f}")

    print(f"\nLog written to: {LOG_PATH.resolve()}")
