# HCSLA-CSM

**Hybrid Constrained Safe Locomotion Architecture with Continuous Speed Morphism**

A constrained reinforcement learning framework for quadruped locomotion that combines Central Pattern Generators (CPGs) with a formally computed stable gait manifold and real-time OSQP projection to guarantee dynamically stable walking across continuous speed transitions.

## Architecture

The system consists of four layers:
1. **Structured Gait Manifold** — Hopf oscillator CPG with Bezier foot trajectories and analytic IK
2. **Stable Region S** — Offline-computed convex hull of dynamically stable (ω, L, β) parameters
3. **Projection Operator** — Real-time OSQP projection ensuring RL actions remain within S
4. **RL Policy** — PPO agent outputting CPG parameter deltas (Δω, ΔL, Δβ)

## Project Structure

```
hcsla-csm/
├── src/                  # Source code
│   ├── cpg/              # CPG oscillator, Bezier trajectories, IK
│   ├── stable_region/    # Grid scan, convex hull, OSQP projection
│   ├── rl/               # PPO policy, reward, curriculum
│   └── sim/              # legged_gym integration, PD control
├── experiments/          # Training configs, evaluation scripts, results
├── docs/                 # Design docs, interface spec, meeting notes
├── paper/                # LaTeX source, figures, bibliography
│   └── figures/
└── README.md
```

## Robot

- **Unitree A1** quadruped in Isaac Gym / legged_gym

## Team

| Name | Role |
|------|------|
| Tanishq | RL Architect & Research Lead |
| Harsimran Singh Dalal | Constrained RL + Evaluation Lead |
| Ishaan Sharma | Simulation & Integration Lead |
| Eliza Arora | Gait Manifold & Morphism Lead |
| Samiksha | Infrastructure & Paper Lead |

## Mentor

Dr. Sachin Kansal, Thapar Institute of Engineering and Technology

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.