"""
Hopf Oscillator — Single Leg Implementation

A Hopf oscillator generates a stable limit cycle, producing rhythmic
phase signals that drive leg swing/stance patterns in quadruped locomotion.

The canonical Hopf oscillator equations:
    dr/dt = γ(μ - r²)r
    dφ/dt = ω

Where:
    r   = amplitude (converges to √μ)
    φ   = phase angle [0, 2π]
    ω   = oscillator frequency (Hz, converted to rad/s internally)
    γ   = convergence rate (how fast amplitude stabilizes)
    μ   = target amplitude squared (r converges to √μ)

In Cartesian form:
    dx/dt = γ(μ - r²)x - ω·y
    dy/dt = γ(μ - r²)y + ω·x
    where r² = x² + y²

For locomotion, we use the phase φ to determine:
    - Stance phase: φ ∈ [0, 2πβ)     → foot on ground
    - Swing phase:  φ ∈ [2πβ, 2π)    → foot in air
    where β = duty factor
"""

import numpy as np
import matplotlib.pyplot as plt


class HopfOscillator:
    """Single-leg Hopf oscillator for CPG-based locomotion."""

    def __init__(self, omega: float = 1.5, amplitude: float = 1.0,
                 dt: float = 0.01, gamma: float = 50.0, phase_init: float = 0.0):
        """
        Args:
            omega: Oscillation frequency in Hz
            amplitude: Target amplitude (√μ)
            dt: Integration timestep in seconds (100 Hz default)
            gamma: Convergence rate — higher = faster lock to limit cycle
            phase_init: Initial phase in radians [0, 2π]
        """
        self.omega = omega
        self.mu = amplitude ** 2  # Target r² value
        self.dt = dt
        self.gamma = gamma

        # Initialize state in Cartesian form on the limit cycle
        r0 = amplitude
        self.x = r0 * np.cos(phase_init)
        self.y = r0 * np.sin(phase_init)

    def step(self) -> tuple[float, float]:
        """
        Advance oscillator by one timestep using Euler integration.

        Returns:
            phi: Current phase angle in [0, 2π]
            x_offset: x-component of oscillator (used for foot offset)
        """
        r2 = self.x ** 2 + self.y ** 2
        omega_rad = 2 * np.pi * self.omega  # Convert Hz to rad/s

        # Hopf oscillator ODEs (Cartesian form)
        dx = self.gamma * (self.mu - r2) * self.x - omega_rad * self.y
        dy = self.gamma * (self.mu - r2) * self.y + omega_rad * self.x

        # Euler integration
        self.x += dx * self.dt
        self.y += dy * self.dt

        # Extract phase
        phi = np.arctan2(self.y, self.x) % (2 * np.pi)

        return phi, self.x

    def get_phase(self) -> float:
        """Return current phase in [0, 2π]."""
        return np.arctan2(self.y, self.x) % (2 * np.pi)

    def get_amplitude(self) -> float:
        """Return current amplitude r."""
        return np.sqrt(self.x ** 2 + self.y ** 2)


def main():
    """Run oscillator for 5 seconds at ω=1.5 Hz and plot results."""

    omega = 1.5     # Hz
    dt = 0.01       # 100 Hz control loop
    duration = 5.0  # seconds
    n_steps = int(duration / dt)

    osc = HopfOscillator(omega=omega, amplitude=1.0, dt=dt)

    # Storage
    times = np.zeros(n_steps)
    phases = np.zeros(n_steps)
    x_offsets = np.zeros(n_steps)
    amplitudes = np.zeros(n_steps)

    # Simulate
    for i in range(n_steps):
        phi, x_off = osc.step()
        times[i] = i * dt
        phases[i] = phi
        x_offsets[i] = x_off
        amplitudes[i] = osc.get_amplitude()

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Hopf Oscillator — Single Leg (ω = {omega} Hz, dt = {dt}s)",
                 fontsize=14, fontweight='bold')

    # Plot 1: Phase over time
    axes[0].plot(times, phases, color='#2196F3', linewidth=1.0)
    axes[0].set_ylabel("Phase φ (rad)")
    axes[0].set_ylim(-0.2, 2 * np.pi + 0.2)
    axes[0].axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5, label='π')
    axes[0].axhline(y=2 * np.pi, color='gray', linestyle=':', alpha=0.5, label='2π')
    axes[0].legend(loc='upper right')
    axes[0].set_title("Phase Advancement")

    # Plot 2: x-component (foot offset signal)
    axes[1].plot(times, x_offsets, color='#4CAF50', linewidth=1.0)
    axes[1].set_ylabel("x offset")
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].set_title("Oscillator x-component (Foot Offset Signal)")

    # Plot 3: Amplitude convergence
    axes[2].plot(times, amplitudes, color='#FF5722', linewidth=1.0)
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target amplitude')
    axes[2].set_ylabel("Amplitude r")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc='upper right')
    axes[2].set_title("Amplitude Convergence to Limit Cycle")

    plt.tight_layout()
    plt.savefig("docs/hopf_single_leg_validation.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Hopf Oscillator Validation Summary")
    print(f"{'='*50}")
    print(f"Frequency:        {omega} Hz")
    print(f"Timestep:         {dt} s ({int(1/dt)} Hz)")
    print(f"Duration:         {duration} s")
    print(f"Total steps:      {n_steps}")
    print(f"Final amplitude:  {amplitudes[-1]:.6f} (target: 1.0)")
    print(f"Phase cycles:     {omega * duration:.1f} expected")
    print(f"Phase wraps:      {np.sum(np.diff(phases) < -np.pi)} detected")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()