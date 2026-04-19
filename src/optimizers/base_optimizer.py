"""
Abstract Base Optimizer
========================
All hyperparameter optimizers inherit from BaseOptimizer and return an
OptimizationResult namedtuple.  The interface is intentionally minimal so
that new optimizers (or real Isaac Gym evaluators) can be plugged in with
minimal changes.

Isaac Gym compatibility:
  Pass any callable ``fitness_fn(hp_dict) -> float`` as the first argument.
  When Isaac Gym is available, wrap your rollout evaluator accordingly.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

from src.rl.hp_space import (
    PARAM_KEYS,
    decode_from_vector,
    encode_to_vector,
    clip_to_bounds,
    random_sample,
    get_bounds_arrays,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """
    Container for a completed optimization run.

    Attributes:
        best_hp      : Best hyperparameter dict found (natural scale).
        best_fitness : Fitness value of best_hp.
        history      : List of (eval_id, fitness) tuples in call order.
        time_taken   : Wall-clock seconds for the entire optimize() call.
        optimizer    : Name of the optimizer that produced this result.
        all_fitnesses: Flat list of all fitness values (mirrors history).
        best_per_eval: Best-so-far fitness at each evaluation index.
    """
    best_hp:       dict
    best_fitness:  float
    history:       List[tuple]
    time_taken:    float
    optimizer:     str
    all_fitnesses: List[float] = field(default_factory=list)
    best_per_eval: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.all_fitnesses and self.history:
            self.all_fitnesses = [f for _, f in self.history]
        if not self.best_per_eval and self.all_fitnesses:
            best = -np.inf
            for f in self.all_fitnesses:
                best = max(best, f)
                self.best_per_eval.append(best)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Optimizer  : {self.optimizer}",
            f"Best fitness: {self.best_fitness:.6f}",
            f"Evaluations : {len(self.history)}",
            f"Time taken  : {self.time_taken:.2f}s",
            "Best hyperparameters:",
        ]
        for k, v in self.best_hp.items():
            if isinstance(v, float):
                lines.append(f"  {k:15s}: {v:.6g}")
            else:
                lines.append(f"  {k:15s}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseOptimizer(ABC):
    """
    Abstract base class for all hyperparameter optimizers.

    Subclasses must implement ``_run()`` which performs the optimization
    loop and calls ``self._evaluate(vec)`` for each candidate.

    Args:
        fitness_fn : Callable mapping hp_dict -> float.  The surrogate
                     ``fitness.evaluate`` or a real Isaac Gym evaluator.
        hp_space   : Module reference (default: src.rl.hp_space). Passed
                     so that subclasses can query bounds without imports.
        n_evals    : Total number of fitness evaluations allowed.
        seed       : Random seed for full reproducibility.
        optimizer_name: Human-readable tag used in CSV logging.
    """

    def __init__(
        self,
        fitness_fn: Callable,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        optimizer_name: str = "base",
    ):
        self.fitness_fn     = fitness_fn
        self.n_evals        = n_evals
        self.seed           = seed
        self.optimizer_name = optimizer_name
        self.rng            = np.random.default_rng(seed)

        # Shared state reset each call to optimize()
        self._history: List[tuple]  = []
        self._eval_count: int       = 0
        self._best_vec: np.ndarray  = None
        self._best_fitness: float   = -np.inf

        # Bounds
        self._lb, self._ub = get_bounds_arrays()
        self._dim = len(PARAM_KEYS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization and return results.

        Resets internal state, calls ``_run()``, then packages results.

        Returns:
            OptimizationResult with best hyperparameters and full history.
        """
        # Reset state
        self._history      = []
        self._eval_count   = 0
        self._best_vec     = None
        self._best_fitness = -np.inf
        self.rng           = np.random.default_rng(self.seed)

        t0 = time.perf_counter()
        self._run()
        elapsed = time.perf_counter() - t0

        best_hp = decode_from_vector(self._best_vec) if self._best_vec is not None else {}

        result = OptimizationResult(
            best_hp      = best_hp,
            best_fitness = self._best_fitness,
            history      = list(self._history),
            time_taken   = elapsed,
            optimizer    = self.optimizer_name,
        )
        return result

    # ------------------------------------------------------------------
    # Protected helpers for subclasses
    # ------------------------------------------------------------------

    def _evaluate(self, vec: np.ndarray) -> float:
        """
        Evaluate a candidate vector, log it, and update best solution.

        Clips the vector to bounds before decoding.

        Args:
            vec: Encoded hyperparameter vector (shape: (9,)).

        Returns:
            Scalar fitness value.
        """
        if self._eval_count >= self.n_evals:
            raise StopIteration(
                f"{self.optimizer_name}: evaluation budget ({self.n_evals}) exhausted"
            )

        vec = clip_to_bounds(vec)
        hp  = decode_from_vector(vec)

        fitness = self.fitness_fn(
            hp,
            optimizer_name=self.optimizer_name,
            eval_id=self._eval_count,
        )

        self._history.append((self._eval_count, fitness))
        self._eval_count += 1

        if fitness > self._best_fitness:
            self._best_fitness = fitness
            self._best_vec     = vec.copy()

        return fitness

    def _budget_remaining(self) -> int:
        """Return number of evaluations still available."""
        return self.n_evals - self._eval_count

    def _random_vector(self) -> np.ndarray:
        """Draw a uniformly random encoded vector."""
        return random_sample(rng=self.rng)

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------

    @abstractmethod
    def _run(self) -> None:
        """
        Execute the optimization loop.

        Subclasses call ``self._evaluate(vec)`` for each candidate and
        must respect ``self.n_evals`` (check ``self._budget_remaining()``).
        """
        raise NotImplementedError
