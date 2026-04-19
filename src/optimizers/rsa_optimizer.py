"""
Reptile Search Algorithm (RSA) Optimizer
=========================================
Implements the Reptile Search Algorithm (Abualigah et al., 2022) adapted
for continuous hyperparameter search in [0,1]^9 space.

Crocodile hunting strategy:
  - First half of population (or iteration): EXPLORATION
      Large random steps, mimic wide area search (encircling prey from far).
  - Second half of population (or iteration): EXPLOITATION
      Small refined steps toward best solution (attacking prey).

Two exploration equations (alternate each iteration) and two exploitation
equations (alternate each iteration) are used to maintain diversity.

Parameters: N=20 agents.

Isaac Gym compatibility:
  No changes needed — fitness_fn is called via BaseOptimizer._evaluate().
"""

import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer


class RSAOptimizer(BaseOptimizer):
    """
    Reptile Search Algorithm optimizer.

    Args:
        fitness_fn  : Callable hp_dict -> float.
        hp_space    : hp_space module (unused directly).
        n_evals     : Total evaluation budget (default 50).
        seed        : Random seed (default 42).
        n_agents    : Population size (default 20).
        ES          : Escalation factor (default 0.5) — exploration scale.
    """

    def __init__(
        self,
        fitness_fn,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        n_agents: int = 20,
        ES: float = 0.5,
    ):
        super().__init__(
            fitness_fn=fitness_fn,
            hp_space=hp_space,
            n_evals=n_evals,
            seed=seed,
            optimizer_name="RSA",
        )
        self.n_agents = n_agents
        self.ES       = ES

    def _run(self) -> None:
        """Execute RSA optimization loop."""
        # Initialise population
        pop  = self.rng.uniform(0.0, 1.0, size=(self.n_agents, self._dim))
        fits = np.full(self.n_agents, -np.inf)

        print(f"\n[RSA] Starting | N={self.n_agents} | ES={self.ES} "
              f"| budget={self.n_evals}")

        for i in range(self.n_agents):
            if self._budget_remaining() <= 0:
                break
            fits[i] = self._evaluate(pop[i])

        max_iter  = max(1, self.n_evals // self.n_agents)
        iteration = 0

        while self._budget_remaining() > 0:
            iteration += 1
            t         = iteration / max_iter  # normalised iteration [0,1]

            best_idx  = np.argmax(fits)
            best_pos  = pop[best_idx].copy()

            # Random individual from population (not best)
            rand_idx  = self.rng.integers(0, self.n_agents)
            rand_pos  = pop[rand_idx].copy()

            half      = self.n_agents // 2

            for i in range(self.n_agents):
                if self._budget_remaining() <= 0:
                    break

                r1 = self.rng.random()
                r2 = self.rng.random()
                r3 = self.rng.random()

                if i < half:
                    # --- EXPLORATION (first half) ---
                    # Alternate between two exploration equations
                    if iteration % 2 == 0:
                        # Eq1: encircling prey from far with random direction
                        new_pos = (best_pos
                                   - self.ES * (
                                       pop[i] * r1
                                       - best_pos * r2
                                   ))
                    else:
                        # Eq2: hunting toward random individual
                        new_pos = (best_pos
                                   + self.ES * (
                                       rand_pos * r3
                                       - pop[i] * r1
                                   ))
                else:
                    # --- EXPLOITATION (second half) ---
                    if iteration % 2 == 0:
                        # Eq3: cooperative attack toward best
                        new_pos = (best_pos
                                   - pop[i] * self.ES * t
                                   + best_pos * r1 * self.ES * t)
                    else:
                        # Eq4: refined local search
                        new_pos = (best_pos
                                   + pop[i] * self.ES * t
                                   - best_pos * r2 * self.ES * t)

                new_pos = np.clip(new_pos, 0.0, 1.0)
                new_fit = self._evaluate(new_pos)

                if new_fit > fits[i]:
                    pop[i]  = new_pos
                    fits[i] = new_fit

            best_fit = np.max(fits)
            print(f"  Iter {iteration:3d} | best={best_fit:.4f} | "
                  f"evals={self._eval_count}/{self.n_evals}")

        print(f"[RSA] Done | best_fitness={self._best_fitness:.4f} "
              f"| evals={self._eval_count}")
