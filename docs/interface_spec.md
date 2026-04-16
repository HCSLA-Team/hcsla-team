# Interface Specification v1.0
## HCSLA-CSM — API Contract

> **Status:** DRAFT — Must be signed off by all 5 members before end of Week 2.  
> **After Week 2, this document is LOCKED.** No changes without a full team meeting.

---

## 1. Action Space (RL → CPG)

The RL policy outputs **deltas** to the CPG parameters, not absolute values.

| Parameter | Symbol | Delta Output | Unit | Clip Range |
|-----------|--------|-------------|------|------------|
| Frequency | Δω | `action[0]` | Hz | [-0.3, +0.3] |
| Stride Length | ΔL | `action[1]` | m | [-0.05, +0.05] |
| Duty Factor | Δβ | `action[2]` | - | [-0.1, +0.1] |

**Action vector:** `a = [Δω, ΔL, Δβ] ∈ ℝ³`

**Application rule:**
```
z_new = z_current + a
z_projected = OSQP_Projector.project(z_new)  # Ensures z ∈ S
CPGController.update(z_projected)
```

**Absolute parameter bounds (physical limits):**

| Parameter | Symbol | Min | Max | Unit |
|-----------|--------|-----|-----|------|
| Frequency | ω | 0.5 | 3.0 | Hz |
| Stride Length | L | 0.05 | 0.35 | m |
| Duty Factor | β | 0.3 | 0.8 | - |

---

## 2. Observation Vector (Sim → RL)

Total dimension: **47**

| Index | Element | Dim | Unit | Source |
|-------|---------|-----|------|--------|
| 0–2 | Base linear velocity (x, y, z) | 3 | m/s | IMU / sim |
| 3–5 | Base angular velocity (roll, pitch, yaw rate) | 3 | rad/s | IMU / sim |
| 6–8 | Projected gravity vector | 3 | - | IMU / sim |
| 9–11 | Velocity commands (v_x, v_y, ω_yaw) | 3 | m/s, rad/s | Command |
| 12–23 | Joint positions (FL hip/thigh/calf, FR, RL, RR) | 12 | rad | Encoders / sim |
| 24–35 | Joint velocities (same order) | 12 | rad/s | Encoders / sim |
| 36–38 | Last RL action (Δω, ΔL, Δβ) | 3 | mixed | Policy buffer |
| 39–42 | CPG phases (φ_FL, φ_FR, φ_RL, φ_RR) | 4 | rad [0, 2π] | CPGController |
| 43–45 | Current CPG params (ω, L, β) | 3 | mixed | CPGController |
| 46 | Base height | 1 | m | Sim |

**Joint ordering convention (matches Unitree A1 URDF):**
```
Index 0–2:   FL_hip, FL_thigh, FL_calf
Index 3–5:   FR_hip, FR_thigh, FR_calf
Index 6–8:   RL_hip, RL_thigh, RL_calf
Index 9–11:  RR_hip, RR_thigh, RR_calf
```

**Observation normalization:** Running mean/std normalization applied during training.

---

## 3. CPGController Interface

```python
class CPGController:
    def __init__(self, omega=1.5, L=0.15, beta=0.55, dt=0.01, gait="trot"):
        """
        Args:
            omega: Oscillator frequency (Hz)
            L: Stride length (m)
            beta: Duty factor (0 = all swing, 1 = all stance)
            dt: Timestep (s), default 100 Hz
            gait: "trot" or "walk"
        """

    def update(self, omega, L, beta):
        """Update CPG parameters (called after OSQP projection)."""

    def step(self) -> np.ndarray:
        """
        Advance oscillator by one timestep.
        Returns:
            q_desired: np.ndarray of shape (12,) — target joint angles in rad
                       Order: [FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...]
        """

    def get_phases(self) -> np.ndarray:
        """
        Returns:
            phases: np.ndarray of shape (4,) — [φ_FL, φ_FR, φ_RL, φ_RR] in [0, 2π]
        """

    def get_params(self) -> np.ndarray:
        """
        Returns:
            params: np.ndarray of shape (3,) — [ω, L, β]
        """
```

---

## 4. OSQP Projection Interface

```python
class OSQP_Projector:
    def __init__(self, hull_A, hull_b):
        """
        Args:
            hull_A: np.ndarray (n_constraints, 3) — from ConvexHull.equations[:, :-1]
            hull_b: np.ndarray (n_constraints,) — from -ConvexHull.equations[:, -1]
        """

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Project z onto stable region S.
        Args:
            z: np.ndarray (3,) — [ω, L, β]
        Returns:
            z_proj: np.ndarray (3,) — projected point guaranteed to be in S
        
        Performance target: < 0.3 ms mean solve time
        """
```

---

## 5. Reward Function

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| Velocity tracking | `-\|v_x - v_cmd\|²` | 1.0 | Track commanded forward velocity |
| Angular vel tracking | `-\|ω_yaw - ω_cmd\|²` | 0.5 | Track commanded yaw rate |
| Body height | `-\|h - 0.28\|²` | 0.3 | Maintain nominal A1 height |
| Orientation | `-\|roll\|² - \|pitch\|²` | 0.2 | Keep body level |
| Energy penalty | `-Σ\|τ · dq\|` | 0.001 | Minimize energy consumption |
| Joint acceleration | `-Σ\|ddq\|²` | 0.0001 | Smooth joint motion |
| Action smoothness | `-\|a_t - a_{t-1}\|²` | 0.05 | Prevent jerky CPG changes |
| Foot contact regularity | `+1 if contact matches gait phase` | 0.1 | Encourage proper gait pattern |
| Survival bonus | `+1 per timestep alive` | 0.5 | Don't fall |

**Total reward:** Weighted sum of all terms, computed per timestep.

---

## 6. Gait Coupling Definitions

**Trot:** Diagonal pairs in phase, lateral pairs π apart.
```
φ_desired = {FL: 0, FR: π, RL: π, RR: 0}
```

**Walk:** Sequential phase offset.
```
φ_desired = {FL: 0, FR: π/2, RL: π, RR: 3π/2}
```

---

## 7. PD Controller

```
τ = Kp * (q_desired - q_actual) - Kd * dq_actual
```

| Parameter | Value | Unit |
|-----------|-------|------|
| Kp | 40.0 | Nm/rad |
| Kd | 1.0 | Nm·s/rad |
| Torque limit | 33.5 | Nm |

**Joint limits (Unitree A1):**

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| Hip | -0.80 | +0.80 |
| Thigh | -1.05 | +4.19 |
| Calf | -2.70 | -0.92 |

---

## 8. Morphism Schedule

Speed transitions use a 5th-order polynomial:

```
σ(t) = 6t⁵ − 15t⁴ + 10t³,  t ∈ [0, 1]

z(t) = z_start + σ(t/T_morph) · (z_target − z_start)
```

Properties: σ(0) = 0, σ(1) = 1, σ'(0) = σ'(1) = 0, σ''(0) = σ''(1) = 0.

---

## 9. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Simulator | Isaac Gym Preview 4 + legged_gym |
| Robot | Unitree A1 |
| Control freq | 100 Hz (dt = 0.01s) |
| Parallel envs | 4096 |
| PPO implementation | rsl_rl |

---

## Sign-Off

| Person | Role | Signed | Date |
|--------|------|--------|------|
| Tanishq | RL Architect | ☐ | |
| Harsimran | Constrained RL | ☐ | |
| Person 3 | Simulation | ☐ | |
| Person 4 | Gait Manifold | ☐ | |
| Person 5 | Infrastructure | ☐ | |