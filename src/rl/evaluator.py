"""
Evaluator Factory & Abstract Base
===================================
Single entry point for all fitness evaluations.

HOW IT WORKS
------------
``get_evaluator()`` reads ``config.yaml`` and returns either:
  - SurrogateEvaluator  — fast, no Isaac Gym needed  (evaluator: surrogate)
  - IsaacGymEvaluator   — real PPO rollouts           (evaluator: isaac_gym)

The returned object is callable:
    evaluator(hp_dict, optimizer_name, eval_id) -> float

This is the EXACT same signature the optimizers already use, so nothing
else in the codebase needs to change when you switch to Isaac Gym.

USAGE
-----
    from src.rl.evaluator import get_evaluator

    fitness_fn = get_evaluator()          # reads config.yaml automatically
    score = fitness_fn(hp, "GA", 0)      # works the same in both modes
"""

import sys
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Abstract base — every evaluator must follow this contract
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """
    Abstract evaluator.

    All evaluators must be callable with:
        __call__(hp_dict, optimizer_name, eval_id) -> float

    and must implement the ``evaluate`` method.
    """

    def __call__(self, hp_dict: dict, optimizer_name: str = "unknown",
                 eval_id: int = 0) -> float:
        return self.evaluate(hp_dict, optimizer_name, eval_id)

    @abstractmethod
    def evaluate(self, hp_dict: dict, optimizer_name: str,
                 eval_id: int) -> float:
        """
        Evaluate a hyperparameter configuration.

        Args:
            hp_dict        : Decoded (natural-scale) hyperparameter dict.
            optimizer_name : Name of calling optimizer (for logging).
            eval_id        : Sequential evaluation index (for logging).

        Returns:
            Scalar fitness in [0, 1].  Higher = better PPO performance.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup (e.g., shut down Isaac Gym environments)."""
        pass


# ---------------------------------------------------------------------------
# Surrogate evaluator (no Isaac Gym required)
# ---------------------------------------------------------------------------

class SurrogateEvaluator(BaseEvaluator):
    """
    Wraps the surrogate fitness function from src.rl.fitness.

    Identical results to calling ``fitness.evaluate()`` directly —
    this class just conforms to the BaseEvaluator interface.
    """

    def __init__(self, noise_std: float = 0.05):
        from src.rl.fitness import evaluate as _surrogate_evaluate
        self._fn       = _surrogate_evaluate
        self.noise_std = noise_std
        print(f"[Evaluator] Mode: SURROGATE  (noise_std={noise_std})")

    def evaluate(self, hp_dict: dict, optimizer_name: str = "unknown",
                 eval_id: int = 0) -> float:
        return self._fn(hp_dict,
                        optimizer_name=optimizer_name,
                        eval_id=eval_id,
                        noise_std=self.noise_std)


# ---------------------------------------------------------------------------
# Isaac Gym evaluator — plug your code in here
# ---------------------------------------------------------------------------

class IsaacGymEvaluator(BaseEvaluator):
    """
    Real Isaac Gym evaluator.

    Runs actual PPO training for a fixed number of steps with the given
    hyperparameters and returns the normalized cumulative reward.

    TO CONNECT ISAAC GYM:
      1. Set ``evaluator: isaac_gym`` in config.yaml
      2. Fill in the TODO sections below
      3. Run ``python experiments/run_comparison.py`` — nothing else changes

    The ``evaluate()`` method signature and return value are identical to
    SurrogateEvaluator, so all optimizers work without any modification.
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: The ``isaac_gym`` section from config.yaml.
        """
        self.cfg = cfg
        self.num_envs        = cfg.get("num_envs", 4096)
        self.episode_length  = cfg.get("episode_length", 1000)
        self.headless        = cfg.get("headless", True)
        self.sim_dt          = cfg.get("sim_dt", 0.005)
        self.control_dt      = cfg.get("control_dt", 0.02)
        self.n_eval_episodes = cfg.get("n_eval_episodes", 3)
        self.normalize       = cfg.get("normalize_reward", True)
        self.max_return      = cfg.get("max_return", 5000.0)
        self.ppo_steps       = cfg.get("ppo_train_steps", 500)
        self.task            = cfg.get("task", "Quadruped")

        print(f"[Evaluator] Mode: ISAAC GYM")
        print(f"  num_envs={self.num_envs}  episode_length={self.episode_length}")
        print(f"  headless={self.headless}  ppo_steps={self.ppo_steps}")

        self._setup()

    def _setup(self) -> None:
        """
        Initialise Isaac Gym and create environments.

        TODO: Replace the block below with your Isaac Gym init code.
        ----------------------------------------------------------------
        Example:
            import isaacgym
            from isaacgymenvs.tasks import isaacgym_task_map

            self.gym = isaacgym.acquire_gym()
            self.sim = ...  # create sim
            self.envs = isaacgym_task_map[self.task](
                cfg=self.cfg,
                sim_device="cuda:0",
                graphics_device_id=0,
                headless=self.headless,
            )
        ----------------------------------------------------------------
        """
        # --- TODO: your Isaac Gym environment setup here ---
        self._gym  = None   # replace with: isaacgym.acquire_gym()
        self._envs = None   # replace with: your task env
        # ----------------------------------------------------

    def _build_ppo_config(self, hp_dict: dict) -> dict:
        """
        Convert an hp_dict into the PPO trainer config format.

        Matches the keys expected by your train_ppo.py / W&B config.
        No changes needed here — hp_space keys already match train_ppo.py.
        """
        return {
            "lr":           hp_dict["lr"],
            "clip_range":   hp_dict["clip_eps"],
            "gamma":        hp_dict["gamma"],
            "batch_size":   hp_dict["batch_size"],
            # reward weights passed to the env
            "w_velocity":   hp_dict["w_velocity"],
            "w_stability":  hp_dict["w_stability"],
            "w_energy":     hp_dict["w_energy"],
            "w_smooth":     hp_dict["w_smooth"],
            "w_survival":   hp_dict["w_survival"],
        }

    def evaluate(self, hp_dict: dict, optimizer_name: str = "unknown",
                 eval_id: int = 0) -> float:
        """
        Train PPO with hp_dict for ``ppo_train_steps`` steps and return
        normalised cumulative reward.

        TODO: Replace the block marked TODO with your PPO training call.
        -----------------------------------------------------------------
        The return value MUST be a float in [0, 1].
        Use self.max_return to normalise:
            fitness = total_reward / self.max_return
            return float(np.clip(fitness, 0.0, 1.0))
        -----------------------------------------------------------------
        """
        import numpy as np
        from src.rl.fitness import _log_call   # reuse the CSV logger

        ppo_cfg = self._build_ppo_config(hp_dict)

        # --- TODO: replace this block with real PPO training ---
        #
        # Example pseudocode:
        #
        #   trainer = PPOTrainer(
        #       env=self._envs,
        #       lr=ppo_cfg["lr"],
        #       clip_range=ppo_cfg["clip_range"],
        #       gamma=ppo_cfg["gamma"],
        #       batch_size=ppo_cfg["batch_size"],
        #       reward_weights={k: ppo_cfg[k] for k in
        #                       ["w_velocity","w_stability","w_energy",
        #                        "w_smooth","w_survival"]},
        #   )
        #   total_reward = trainer.train(steps=self.ppo_steps)
        #   fitness = float(np.clip(total_reward / self.max_return, 0, 1))
        #
        raise NotImplementedError(
            "IsaacGymEvaluator.evaluate() is not yet implemented.\n"
            "Fill in the TODO block in src/rl/isaac_gym_evaluator.py\n"
            "or directly in src/rl/evaluator.py -> IsaacGymEvaluator.evaluate()"
        )
        # -------------------------------------------------------

        _log_call(optimizer_name, eval_id, fitness, hp_dict)
        return fitness

    def close(self) -> None:
        """Shut down Isaac Gym simulation."""
        # TODO: self.gym.destroy_sim(self.sim)
        pass


# ---------------------------------------------------------------------------
# Factory — the ONLY function everything else should import
# ---------------------------------------------------------------------------

def get_evaluator(config_path: str = None) -> BaseEvaluator:
    """
    Read config.yaml and return the correct evaluator.

    This is the single import used by run_comparison.py and any future
    training scripts.  Switching between surrogate and Isaac Gym requires
    changing ONE line in config.yaml only.

    Args:
        config_path: Path to config.yaml. Defaults to project root.

    Returns:
        A callable BaseEvaluator instance.

    Raises:
        ValueError: If ``evaluator`` key in config is unknown.
    """
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        print(f"[Evaluator] config.yaml not found at {path}, using surrogate.")
        return SurrogateEvaluator()

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("evaluator", "surrogate").strip().lower()

    if mode == "surrogate":
        noise_std = cfg.get("surrogate", {}).get("noise_std", 0.05)
        return SurrogateEvaluator(noise_std=noise_std)

    elif mode == "isaac_gym":
        isaac_cfg = cfg.get("isaac_gym", {})
        return IsaacGymEvaluator(isaac_cfg)

    else:
        raise ValueError(
            f"Unknown evaluator '{mode}' in config.yaml.\n"
            f"Valid options: surrogate | isaac_gym"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.rl.hp_space import decode_from_vector, random_sample
    import numpy as np

    ev  = get_evaluator()
    rng = np.random.default_rng(42)
    hp  = decode_from_vector(random_sample(rng=rng))
    f   = ev(hp, optimizer_name="test", eval_id=0)
    print(f"Fitness: {f:.4f}")
    ev.close()
