"""
Genetic Algorithm Optimizer
============================
Classic evolutionary optimization with:
  - Tournament selection (k=3)
  - Single-point crossover
  - Gaussian mutation (sigma=0.05, probability=0.15 per gene)
  - Elitism: top-2 individuals always survive

Population size: 20
Works entirely in the encoded [0,1]^9 hypercube.

Isaac Gym compatibility:
  No changes needed — fitness_fn is called via BaseOptimizer._evaluate().
"""

import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer


class GAOptimizer(BaseOptimizer):
    """
    Genetic Algorithm for hyperparameter optimisation.

    Args:
        fitness_fn   : Callable hp_dict -> float.
        hp_space     : hp_space module (unused directly; parent uses it).
        n_evals      : Total evaluation budget (default 50).
        seed         : Random seed (default 42).
        pop_size     : Population size (default 20).
        tournament_k : Tournament selection size (default 3).
        crossover_pt : Fixed crossover point; None = random each time.
        mutation_sigma: Gaussian mutation std dev (default 0.05).
        mutation_prob: Per-gene mutation probability (default 0.15).
        n_elites     : Number of elite individuals kept each gen (default 2).
    """

    def __init__(
        self,
        fitness_fn,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        pop_size: int = 20,
        tournament_k: int = 3,
        crossover_pt: int = None,
        mutation_sigma: float = 0.05,
        mutation_prob: float = 0.15,
        n_elites: int = 2,
    ):
        super().__init__(
            fitness_fn=fitness_fn,
            hp_space=hp_space,
            n_evals=n_evals,
            seed=seed,
            optimizer_name="GeneticAlgorithm",
        )
        self.pop_size       = pop_size
        self.tournament_k   = tournament_k
        self.crossover_pt   = crossover_pt
        self.mutation_sigma = mutation_sigma
        self.mutation_prob  = mutation_prob
        self.n_elites       = n_elites

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        """Randomly initialise population in [0,1]^dim."""
        return self.rng.uniform(0.0, 1.0, size=(self.pop_size, self._dim))

    def _tournament_select(self, population: np.ndarray,
                           fitnesses: np.ndarray) -> np.ndarray:
        """
        Select one individual via tournament selection.

        Randomly samples ``tournament_k`` individuals and returns the one
        with the highest fitness.
        """
        indices = self.rng.choice(len(population), size=self.tournament_k,
                                  replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def _single_point_crossover(self, parent_a: np.ndarray,
                                parent_b: np.ndarray) -> tuple:
        """
        Single-point crossover.

        Returns two children.  Crossover point is random unless
        ``self.crossover_pt`` is fixed.
        """
        pt = (self.crossover_pt if self.crossover_pt is not None
              else int(self.rng.integers(1, self._dim)))
        child_a = np.concatenate([parent_a[:pt], parent_b[pt:]])
        child_b = np.concatenate([parent_b[:pt], parent_a[pt:]])
        return child_a, child_b

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian mutation to each gene with probability mutation_prob.

        Result is clipped to [0, 1].
        """
        mask  = self.rng.random(self._dim) < self.mutation_prob
        noise = self.rng.normal(0.0, self.mutation_sigma, size=self._dim)
        individual = individual + mask * noise
        return np.clip(individual, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Execute the GA optimization loop."""
        # Initialise population
        population = self._init_population()
        fitnesses  = np.full(self.pop_size, -np.inf)

        # Evaluate initial population
        for i in range(self.pop_size):
            if self._budget_remaining() <= 0:
                break
            fitnesses[i] = self._evaluate(population[i])

        generation = 0
        print(f"\n[GA] Starting | pop={self.pop_size} | budget={self.n_evals}")

        while self._budget_remaining() > 0:
            generation += 1

            # --- Elitism: keep top-n_elites ---
            elite_idx = np.argsort(fitnesses)[-self.n_elites:]
            elites    = population[elite_idx].copy()

            # --- Breed new generation ---
            new_pop  = list(elites)
            new_fits = list(fitnesses[elite_idx])

            while len(new_pop) < self.pop_size and self._budget_remaining() > 0:
                # Selection
                p_a = self._tournament_select(population, fitnesses)
                p_b = self._tournament_select(population, fitnesses)

                # Crossover
                c_a, c_b = self._single_point_crossover(p_a, p_b)

                # Mutation
                c_a = self._mutate(c_a)
                c_b = self._mutate(c_b)

                # Evaluate children (respect budget)
                for child in [c_a, c_b]:
                    if self._budget_remaining() <= 0:
                        break
                    f = self._evaluate(child)
                    new_pop.append(child)
                    new_fits.append(f)

            # Replace population
            population = np.array(new_pop[:self.pop_size])
            fitnesses  = np.array(new_fits[:self.pop_size])

            best_gen = np.max(fitnesses)
            print(f"  Gen {generation:3d} | best={best_gen:.4f} | "
                  f"evals={self._eval_count}/{self.n_evals}")

        print(f"[GA] Done | best_fitness={self._best_fitness:.4f} "
              f"| evals={self._eval_count}")
