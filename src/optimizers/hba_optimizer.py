"""
Honey Badger Algorithm (HBA) Optimizer
=======================================
Implements the Honey Badger Algorithm (Hashim et al., 2022) adapted for
continuous hyperparameter optimization in [0,1]^9 space.

The honey badger alternates between two foraging strategies:
  - Digging mode  (exploitation): move toward the best solution with
    decreasing step size driven by a smell intensity gradient.
  - Honey-guide mode (exploration): follow a "honey guide" (random global
    direction) scaled by an intensity factor.

Parameters:
  N=20 agents, alternates modes each iteration, beta controls intensity decay.

Isaac Gym compatibility:
  No changes needed — fitness_fn is called via BaseOptimizer._evaluate().
"""

import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer


class HBAOptimizer(BaseOptimizer):
    """
    Honey Badger Algorithm optimizer for hyperparameter search.

    Args:
        fitness_fn  : Callable hp_dict -> float.
        hp_space    : hp_space module (unused directly).
        n_evals     : Total evaluation budget (default 50).
        seed        : Random seed (default 42).
        n_agents    : Number of honey badgers (default 20).
        beta        : Ability indicator (controls intensity; default 6).
        C           : Constant controlling digging/honey-guide (default 2).
    """

    def __init__(
        self,
        fitness_fn,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        n_agents: int = 20,
        beta: float = 6.0,
        C: float = 2.0,
    ):
        super().__init__(
            fitness_fn=fitness_fn,
            hp_space=hp_space,
            n_evals=n_evals,
            seed=seed,
            optimizer_name="HBA",
        )
        self.n_agents = n_agents
        self.beta     = beta
        self.C        = C

    def _run(self) -> None:
        """Execute HBA optimization loop."""
        # Initialise population
        pop  = self.rng.uniform(0.0, 1.0, size=(self.n_agents, self._dim))
        fits = np.full(self.n_agents, -np.inf)

        print(f"\n[HBA] Starting | N={self.n_agents} | beta={self.beta} "
              f"| budget={self.n_evals}")

        # Initial evaluation
        for i in range(self.n_agents):
            if self._budget_remaining() <= 0:
                break
            fits[i] = self._evaluate(pop[i])

        max_iter   = self.n_evals // self.n_agents + 1
        iteration  = 0

        while self._budget_remaining() > 0:
            iteration += 1

            # Compute alpha (decreasing over iterations)
            alpha = C_val = self.C * np.exp(-iteration / max_iter)

            # Best solution
            best_idx = np.argmax(fits)
            best_pos = pop[best_idx].copy()

            for i in range(self.n_agents):
                if self._budget_remaining() <= 0:
                    break

                # Smell intensity (distance-based)
                dist = np.linalg.norm(pop[i] - best_pos) + 1e-10
                I    = self.rng.random() * (fits[best_idx] - fits[i] + 1e-10) / (
                    4 * np.pi * dist ** 2 + 1e-10
                )

                r1 = self.rng.random()
                r2 = self.rng.random()
                r3 = self.rng.random()
                r4 = self.rng.random()
                F  = 1 if self.rng.random() > 0.5 else -1  # direction flag

                # Alternate digging / honey-guide based on random flag
                if self.rng.random() < 0.5:
                    # Digging mode (exploitation)
                    di  = self.rng.random(self._dim)  # random prey direction
                    new_pos = (best_pos
                               + F * self.beta * I * best_pos
                               + F * r1 * alpha * di * np.abs(
                                   np.cos(2 * np.pi * r2)
                                   * (1 - np.cos(2 * np.pi * r3))
                               ))
                else:
                    # Honey-guide mode (exploration)
                    new_pos = (best_pos
                               + F * r4 * alpha * (best_pos - pop[i]))

                new_pos = np.clip(new_pos, 0.0, 1.0)
                new_fit = self._evaluate(new_pos)

                if new_fit > fits[i]:
                    pop[i]  = new_pos
                    fits[i] = new_fit

            best_fit = np.max(fits)
            print(f"  Iter {iteration:3d} | best={best_fit:.4f} | "
                  f"evals={self._eval_count}/{self.n_evals}")

        print(f"[HBA] Done | best_fitness={self._best_fitness:.4f} "
              f"| evals={self._eval_count}")
