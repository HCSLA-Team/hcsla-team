"""
Squirrel Search Algorithm (SSA) Optimizer
==========================================
Implements the Squirrel Search Algorithm (Jain et al., 2019) adapted for
continuous hyperparameter optimization in [0,1]^9 space.

Squirrel foraging strategy:
  - 1  squirrel on acorn tree   (best solution — exploitation)
  - 3  squirrels on hickory trees (medium-good solutions — semi-exploitation)
  - Rest on normal trees          (random positions — exploration)

Each iteration squirrels glide toward better food sources using a
gliding constant Gc=1.9.  A seasonal scatter is triggered when population
diversity drops below 1e-6 (prevents premature convergence).

Parameters: N=20 agents, Gc=1.9.

Isaac Gym compatibility:
  No changes needed — fitness_fn is called via BaseOptimizer._evaluate().
"""

import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer


class SSAOptimizer(BaseOptimizer):
    """
    Squirrel Search Algorithm optimizer for hyperparameter search.

    Args:
        fitness_fn       : Callable hp_dict -> float.
        hp_space         : hp_space module (unused directly).
        n_evals          : Total evaluation budget (default 50).
        seed             : Random seed (default 42).
        n_agents         : Total squirrel population (default 20).
        n_hickory        : Number of hickory-tree squirrels (default 3).
        Gc               : Gliding constant (default 1.9).
        Pdp              : Predator presence probability (default 0.1).
        diversity_thresh : Scatter trigger threshold (default 1e-6).
    """

    def __init__(
        self,
        fitness_fn,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        n_agents: int = 20,
        n_hickory: int = 3,
        Gc: float = 1.9,
        Pdp: float = 0.1,
        diversity_thresh: float = 1e-6,
    ):
        super().__init__(
            fitness_fn=fitness_fn,
            hp_space=hp_space,
            n_evals=n_evals,
            seed=seed,
            optimizer_name="SSA",
        )
        self.n_agents        = n_agents
        self.n_hickory       = n_hickory
        self.Gc              = Gc
        self.Pdp             = Pdp
        self.diversity_thresh = diversity_thresh

        # Remaining squirrels forage on normal trees
        self.n_normal = n_agents - 1 - n_hickory  # 1 acorn + 3 hickory + rest

    def _population_diversity(self, pop: np.ndarray) -> float:
        """Mean pairwise Euclidean distance as diversity metric."""
        n = len(pop)
        if n < 2:
            return 1.0
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        return dists.sum() / (n * (n - 1))

    def _seasonal_scatter(self, pop: np.ndarray) -> np.ndarray:
        """
        Seasonal scatter: re-initialise non-acorn squirrels with Lévy-like
        random positions to escape local optima.
        """
        n = len(pop)
        # Keep the best (acorn) squirrel, reset the rest
        new_pop    = pop.copy()
        new_pop[1:] = self.rng.uniform(0.0, 1.0, size=(n - 1, self._dim))
        return new_pop

    def _glide(self, squirrel: np.ndarray, target: np.ndarray,
               dg: float) -> np.ndarray:
        """
        Glide update: squirrel moves toward target.

        new_pos = squirrel + Gc * dg * (target - squirrel)
        where dg is the gliding distance drawn from uniform(0,1).
        """
        new_pos = squirrel + self.Gc * dg * (target - squirrel)
        return np.clip(new_pos, 0.0, 1.0)

    def _run(self) -> None:
        """Execute SSA optimization loop."""
        # Initialise population randomly
        pop  = self.rng.uniform(0.0, 1.0, size=(self.n_agents, self._dim))
        fits = np.full(self.n_agents, -np.inf)

        print(f"\n[SSA] Starting | N={self.n_agents} | hickory={self.n_hickory} "
              f"| Gc={self.Gc} | budget={self.n_evals}")

        # Initial evaluation
        for i in range(self.n_agents):
            if self._budget_remaining() <= 0:
                break
            fits[i] = self._evaluate(pop[i])

        iteration = 0

        while self._budget_remaining() > 0:
            iteration += 1

            # Sort by fitness descending
            order = np.argsort(fits)[::-1]
            pop   = pop[order]
            fits  = fits[order]

            # Role assignment
            acorn_pos   = pop[0]                          # best
            hickory_pos = pop[1: 1 + self.n_hickory]     # medium
            # normal squirrels: pop[1+n_hickory:]

            new_pop  = pop.copy()
            new_fits = fits.copy()

            # --- Hickory-tree squirrels glide toward acorn ---
            for i in range(1, 1 + self.n_hickory):
                if self._budget_remaining() <= 0:
                    break
                if self.rng.random() >= self.Pdp:
                    dg      = self.rng.random()
                    new_pos = self._glide(pop[i], acorn_pos, dg)
                else:
                    new_pos = self._random_vector()
                f = self._evaluate(new_pos)
                new_pop[i]  = new_pos
                new_fits[i] = f

            # --- Normal-tree squirrels glide toward random hickory ---
            for i in range(1 + self.n_hickory, self.n_agents):
                if self._budget_remaining() <= 0:
                    break
                if self.rng.random() >= self.Pdp:
                    hk_idx  = self.rng.integers(0, self.n_hickory)
                    target  = hickory_pos[hk_idx]
                    dg      = self.rng.random()
                    new_pos = self._glide(pop[i], target, dg)
                else:
                    new_pos = self._random_vector()
                f = self._evaluate(new_pos)
                new_pop[i]  = new_pos
                new_fits[i] = f

            pop  = new_pop
            fits = new_fits

            # --- Seasonal scatter if diversity too low ---
            div = self._population_diversity(pop)
            if div < self.diversity_thresh:
                pop  = self._seasonal_scatter(pop)
                # Re-evaluate scattered squirrels
                for i in range(1, self.n_agents):
                    if self._budget_remaining() <= 0:
                        break
                    fits[i] = self._evaluate(pop[i])
                print(f"  Iter {iteration:3d} | SCATTER triggered (div={div:.2e})")

            best_fit = np.max(fits)
            print(f"  Iter {iteration:3d} | best={best_fit:.4f} | div={div:.2e} | "
                  f"evals={self._eval_count}/{self.n_evals}")

        print(f"[SSA] Done | best_fitness={self._best_fitness:.4f} "
              f"| evals={self._eval_count}")
