"""
Particle Swarm Optimizer (PSO)
================================
Classic PSO with velocity clamping.

Configuration:
  - 20 particles
  - Inertia weight w=0.7
  - Cognitive coefficient c1=1.5 (personal best attraction)
  - Social coefficient c2=1.5 (global best attraction)
  - Velocity clamped to ±20% of dimension range (i.e. ±0.20 in [0,1] space)

Works in the encoded [0,1]^9 hypercube.

Isaac Gym compatibility:
  No changes needed — fitness_fn is called via BaseOptimizer._evaluate().
"""

import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimizer for hyperparameter search.

    Args:
        fitness_fn  : Callable hp_dict -> float.
        hp_space    : hp_space module (unused directly).
        n_evals     : Total evaluation budget (default 50).
        seed        : Random seed (default 42).
        n_particles : Swarm size (default 20).
        w           : Inertia weight (default 0.7).
        c1          : Cognitive coefficient (default 1.5).
        c2          : Social coefficient (default 1.5).
        v_clamp_frac: Velocity clamped to ±(v_clamp_frac * range).
    """

    def __init__(
        self,
        fitness_fn,
        hp_space=None,
        n_evals: int = 50,
        seed: int = 42,
        n_particles: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_clamp_frac: float = 0.20,
    ):
        super().__init__(
            fitness_fn=fitness_fn,
            hp_space=hp_space,
            n_evals=n_evals,
            seed=seed,
            optimizer_name="PSO",
        )
        self.n_particles  = n_particles
        self.w            = w
        self.c1           = c1
        self.c2           = c2
        self.v_clamp_frac = v_clamp_frac

    def _run(self) -> None:
        """Execute PSO optimization loop."""
        v_max = self.v_clamp_frac  # max |velocity| per dimension (range=1)

        # Initialise positions uniformly in [0,1]^dim
        positions  = self.rng.uniform(0.0, 1.0, size=(self.n_particles, self._dim))
        velocities = self.rng.uniform(-v_max, v_max,
                                      size=(self.n_particles, self._dim))

        # Personal bests
        pbest_pos = positions.copy()
        pbest_fit = np.full(self.n_particles, -np.inf)

        # Global best
        gbest_pos = positions[0].copy()
        gbest_fit = -np.inf

        # Evaluate initial positions
        print(f"\n[PSO] Starting | particles={self.n_particles} | budget={self.n_evals}")

        for i in range(self.n_particles):
            if self._budget_remaining() <= 0:
                break
            f = self._evaluate(positions[i])
            pbest_fit[i] = f
            if f > gbest_fit:
                gbest_fit = f
                gbest_pos = positions[i].copy()

        iteration = 0
        while self._budget_remaining() > 0:
            iteration += 1

            r1 = self.rng.uniform(0.0, 1.0, size=(self.n_particles, self._dim))
            r2 = self.rng.uniform(0.0, 1.0, size=(self.n_particles, self._dim))

            # Velocity update
            velocities = (
                self.w  * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            # Clamp velocity
            velocities = np.clip(velocities, -v_max, v_max)

            # Position update and boundary handling (reflect)
            positions = positions + velocities
            # Reflect out-of-bound particles
            too_low  = positions < 0.0
            too_high = positions > 1.0
            positions  = np.where(too_low,  -positions,        positions)
            positions  = np.where(too_high,  2.0 - positions,  positions)
            velocities = np.where(too_low | too_high, -velocities, velocities)
            positions  = np.clip(positions, 0.0, 1.0)

            # Evaluate updated particles
            for i in range(self.n_particles):
                if self._budget_remaining() <= 0:
                    break
                f = self._evaluate(positions[i])
                if f > pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = positions[i].copy()
                if f > gbest_fit:
                    gbest_fit = f
                    gbest_pos = positions[i].copy()

            print(f"  Iter {iteration:3d} | gbest={gbest_fit:.4f} | "
                  f"evals={self._eval_count}/{self.n_evals}")

        print(f"[PSO] Done | best_fitness={self._best_fitness:.4f} "
              f"| evals={self._eval_count}")
