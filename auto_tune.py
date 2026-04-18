#!/usr/bin/env python
"""Automatic gain tuning for spacecraft attitude control."""

import numpy as np
import json
from pathlib import Path
from typing import Dict

from configs.config import ConfigurationGenerator
from src.dynamics import SpacecraftDynamics
from src.representations import MRP, Quaternion, EulerAngles


def analytical_gain_tuning(inertia_matrix: np.ndarray, representation: str) -> Dict[str, float]:
    """
    Analytical gain tuning based on spacecraft control theory.

    Args:
        inertia_matrix: Spacecraft inertia matrix
        representation: Attitude representation ('MRP', 'Quaternion', 'Euler')

    Returns:
        Dictionary of tuned gains
    """
    # Get inertia properties
    I_diag = np.diag(inertia_matrix)
    I_max = np.max(I_diag)
    I_min = np.min(I_diag)
    I_avg = np.mean(I_diag)

    # Base gains scaled by inertia
    base_gain = 10.0 * I_avg / 10.0  # Scale factor based on typical spacecraft

    if representation == 'MRP':
        # MRP representation - good for large angles, stable
        Kp = base_gain * 2.0
        Kd = 2.0 * np.sqrt(Kp * I_avg)  # Critical damping
        Ki = Kp / 100.0  # Conservative integral gain

    elif representation == 'Quaternion':
        # Quaternion representation - better for small angles
        Kp = base_gain * 0.5  # Lower proportional gain
        Kd = 2.0 * np.sqrt(Kp * I_avg)
        Ki = Kp / 200.0  # Very conservative for quaternion

    else:  # Euler
        # Euler angles - problematic for large angles
        Kp = base_gain * 0.1  # Much lower gain due to singularities
        Kd = 2.0 * np.sqrt(Kp * I_avg)
        Ki = Kp / 500.0  # Very low integral gain

    # Saturation based on inertia
    saturation_limit = 2.0 * np.sqrt(I_max)  # Torque limit scales with sqrt(inertia)

    return {
        'Kp': float(Kp),
        'Kd': float(Kd),
        'Ki': float(Ki),
        'saturation_limit': float(saturation_limit)
    }


def tune_all_gains() -> Dict[str, Dict]:
    """Automatically tune gains for all control configurations using analytical methods."""

    print("🔧 Analytical Gain Tuning")
    print("=" * 50)

    # Create dynamics
    dynamics = SpacecraftDynamics()

    # Representations to tune
    representations = {
        'MRP': MRP(),
        'Quaternion': Quaternion(),
        'Euler': EulerAngles()
    }

    tuned_gains = {}

    for rep_name, rep in representations.items():
        print(f"\n📐 Tuning gains for {rep_name} representation:")

        # Get analytical gains for this representation
        gains = analytical_gain_tuning(dynamics.I, rep_name)
        print(f"  Base gains: Kp={gains['Kp']:.2f}, Kd={gains['Kd']:.2f}, Ki={gains['Ki']:.4f}")
        print(f"  Saturation: {gains['saturation_limit']:.2f} N·m")

        # Apply to all disturbance cases (gains are representation-dependent, not disturbance-dependent)
        disturbance_cases = ConfigurationGenerator.disturbance_cases()

        for dist_case in disturbance_cases.keys():
            # PD control
            config_key = f"{rep_name}_PD_{dist_case}"
            pd_gains = gains.copy()
            del pd_gains['Ki']  # PD doesn't use Ki
            tuned_gains[config_key] = pd_gains

            # PID control
            config_key = f"{rep_name}_PID_{dist_case}"
            pid_gains = gains.copy()
            tuned_gains[config_key] = pid_gains

            # Lyapunov control (uses same gains as PD)
            config_key = f"{rep_name}_Lyapunov_{dist_case}"
            lyapunov_gains = gains.copy()
            del lyapunov_gains['Ki']  # Lyapunov typically doesn't use integral
            tuned_gains[config_key] = lyapunov_gains

    return tuned_gains


def save_tuned_gains(gains: Dict[str, Dict], filename: str = "tuned_gains.json"):
    """Save tuned gains to file."""
    with open(filename, 'w') as f:
        json.dump(gains, f, indent=2)
    print(f"\n💾 Saved tuned gains to {filename}")


def load_tuned_gains(filename: str = "tuned_gains.json") -> Dict[str, Dict]:
    """Load tuned gains from file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def compare_gains():
    """Compare manual vs automatic gains."""
    print("\n📊 Gain Comparison: Manual vs Analytical")
    print("=" * 50)

    # Load automatic gains
    auto_gains = load_tuned_gains()

    # Manual gains from config
    manual_gains = ConfigurationGenerator.control_gains()

    if not auto_gains:
        print("No automatic gains found. Run tune_all_gains() first.")
        return

    for config in ['MRP_PD_NoDisturbance', 'Quaternion_PD_NoDisturbance', 'Euler_PD_NoDisturbance']:
        if config in auto_gains and config.split('_')[1] in manual_gains:
            controller = config.split('_')[1]
            print(f"\n{config}:")
            manual = manual_gains[controller]
            auto = auto_gains[config]

            print("  Manual:   " +
                  ", ".join([f"{k}={manual.get(k, 'N/A'):.2f}" for k in ['Kp', 'Kd', 'Ki'] if k in manual]))
            print("  Analytical: " +
                  ", ".join([f"{k}={auto.get(k, 'N/A'):.2f}" for k in ['Kp', 'Kd', 'Ki'] if k in auto]))


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'tune':
        gains = tune_all_gains()
        save_tuned_gains(gains)
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_gains()
    else:
        print("Usage:")
        print("  python auto_tune.py tune     # Run analytical tuning")
        print("  python auto_tune.py compare  # Compare manual vs analytical gains")