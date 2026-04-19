"""
Hyperparameter Search Space Definition
=======================================
Defines the hyperparameter space for PPO + CPG locomotion training.
Provides encode/decode utilities to convert between dict and numpy vector
representations used by the optimizers.

Hyperparameters:
    lr            : Learning rate           [1e-5, 1e-3]  (log scale)
    clip_eps      : PPO clip epsilon        [0.1, 0.4]
    gamma         : Discount factor         [0.95, 0.999]
    w_velocity    : Velocity reward weight  [0.5, 3.0]
    w_stability   : Stability reward weight [0.5, 5.0]
    w_energy      : Energy reward weight    [0.0001, 0.01]
    w_smooth      : Smoothness reward wt    [0.01, 0.2]
    w_survival    : Survival reward weight  [0.1, 1.0]
    batch_size    : PPO batch size          {512, 1024, 2048, 4096}

Isaac Gym compatibility note:
    When connecting to Isaac Gym, pass the decoded dict directly to the
    PPO trainer — all keys match the train_ppo.py interface exactly.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

# Continuous parameters: (min, max) in natural (decoded) scale
CONTINUOUS_PARAMS = {
    "lr":           (1e-5,  1e-3),
    "clip_eps":     (0.1,   0.4),
    "gamma":        (0.95,  0.999),
    "w_velocity":   (0.5,   3.0),
    "w_stability":  (0.5,   5.0),
    "w_energy":     (1e-4,  1e-2),
    "w_smooth":     (0.01,  0.2),
    "w_survival":   (0.1,   1.0),
}

# Discrete parameter: encoded as float index 0..3 → actual values
BATCH_SIZE_OPTIONS = [512, 1024, 2048, 4096]

# Ordered list of all keys — defines vector layout
# Indices 0-7: continuous; index 8: batch_size (encoded as 0..1 float)
PARAM_KEYS = list(CONTINUOUS_PARAMS.keys()) + ["batch_size"]

# Bounds in encoded space (all dimensions live in [0, 1])
ENCODED_BOUNDS = np.array([[0.0, 1.0]] * len(PARAM_KEYS))  # shape (9, 2)

# Natural-space bounds for continuous params (used for clipping)
NATURAL_BOUNDS = {k: v for k, v in CONTINUOUS_PARAMS.items()}
NATURAL_BOUNDS["batch_size"] = (512, 4096)  # informational only


def _to_log_scale(value: float, lo: float, hi: float) -> float:
    """Map linear [lo, hi] → log-encoded [0, 1]."""
    return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))


def _from_log_scale(encoded: float, lo: float, hi: float) -> float:
    """Map log-encoded [0, 1] → linear [lo, hi]."""
    return np.exp(encoded * (np.log(hi) - np.log(lo)) + np.log(lo))


def _to_linear_scale(value: float, lo: float, hi: float) -> float:
    """Map natural [lo, hi] → encoded [0, 1]."""
    return (value - lo) / (hi - lo)


def _from_linear_scale(encoded: float, lo: float, hi: float) -> float:
    """Map encoded [0, 1] → natural [lo, hi]."""
    return lo + encoded * (hi - lo)


# Parameters that use log encoding (span multiple orders of magnitude)
LOG_PARAMS = {"lr", "w_energy"}


def encode_to_vector(hp_dict: dict) -> np.ndarray:
    """
    Encode a hyperparameter dict into a unit-hypercube vector.

    All 9 dimensions are mapped to [0, 1]:
      - lr, w_energy : log scale
      - other continuous : linear scale
      - batch_size : index / (n_options - 1)

    Args:
        hp_dict: Dict with keys matching PARAM_KEYS.

    Returns:
        np.ndarray of shape (9,), dtype float64.
    """
    vec = np.zeros(len(PARAM_KEYS))
    for i, key in enumerate(PARAM_KEYS):
        if key == "batch_size":
            idx = BATCH_SIZE_OPTIONS.index(int(hp_dict[key]))
            vec[i] = idx / (len(BATCH_SIZE_OPTIONS) - 1)
        elif key in LOG_PARAMS:
            lo, hi = CONTINUOUS_PARAMS[key]
            vec[i] = _to_log_scale(hp_dict[key], lo, hi)
        else:
            lo, hi = CONTINUOUS_PARAMS[key]
            vec[i] = _to_linear_scale(hp_dict[key], lo, hi)
    return vec


def decode_from_vector(vec: np.ndarray) -> dict:
    """
    Decode a unit-hypercube vector back to a hyperparameter dict.

    Clips each encoded value to [0, 1] before decoding so out-of-bounds
    optimizer proposals are handled gracefully.

    Args:
        vec: np.ndarray of shape (9,).

    Returns:
        dict with keys matching PARAM_KEYS and natural-scale values.
    """
    vec = np.clip(vec, 0.0, 1.0)
    hp = {}
    for i, key in enumerate(PARAM_KEYS):
        v = float(vec[i])
        if key == "batch_size":
            idx = int(round(v * (len(BATCH_SIZE_OPTIONS) - 1)))
            idx = np.clip(idx, 0, len(BATCH_SIZE_OPTIONS) - 1)
            hp[key] = BATCH_SIZE_OPTIONS[idx]
        elif key in LOG_PARAMS:
            lo, hi = CONTINUOUS_PARAMS[key]
            hp[key] = _from_log_scale(v, lo, hi)
        else:
            lo, hi = CONTINUOUS_PARAMS[key]
            hp[key] = _from_linear_scale(v, lo, hi)
    return hp


def clip_to_bounds(vec: np.ndarray) -> np.ndarray:
    """
    Clip an encoded vector to the valid hypercube [0, 1]^9.

    Args:
        vec: np.ndarray of shape (9,).

    Returns:
        Clipped np.ndarray of shape (9,).
    """
    return np.clip(vec, ENCODED_BOUNDS[:, 0], ENCODED_BOUNDS[:, 1])


def random_sample(rng: np.random.Generator = None, seed: int = 42) -> np.ndarray:
    """
    Draw a uniformly random encoded vector from the search space.

    Args:
        rng  : Optional numpy Generator. If None, one is created with seed.
        seed : Seed used only when rng is None.

    Returns:
        np.ndarray of shape (9,) in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=len(PARAM_KEYS))


def get_bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return lower and upper bound arrays for the encoded space.

    Returns:
        (lb, ub) each of shape (9,), all zeros and ones respectively.
    """
    lb = ENCODED_BOUNDS[:, 0].copy()
    ub = ENCODED_BOUNDS[:, 1].copy()
    return lb, ub


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_hp = {
        "lr":           3e-4,
        "clip_eps":     0.2,
        "gamma":        0.99,
        "w_velocity":   1.5,
        "w_stability":  2.0,
        "w_energy":     0.001,
        "w_smooth":     0.05,
        "w_survival":   0.5,
        "batch_size":   2048,
    }
    vec = encode_to_vector(sample_hp)
    recovered = decode_from_vector(vec)
    print("Original :", sample_hp)
    print("Encoded  :", vec.round(4))
    print("Recovered:", recovered)
    print("Bounds OK:", np.all((vec >= 0) & (vec <= 1)))
