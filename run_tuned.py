#!/usr/bin/env python
"""Run simulations with automatically tuned gains."""

import numpy as np
import json
import pickle
from pathlib import Path

from configs.config import ConfigurationGenerator
from src.dynamics import SpacecraftDynamics
from src.representations import MRP, Quaternion, EulerAngles
from src.control import PDControl, PIDControl, LyapunovControl
from src.simulation import SpacecraftSimulator
from src.analysis import PerformanceMetrics


def load_tuned_gains(filename: str = "tuned_gains.json") -> dict:
    """Load tuned gains from file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tuned gains file not found: {filename}. Run `python auto_tune.py tune` first."
        )


def create_controller(controller_type: str, gains: dict, dynamics: SpacecraftDynamics):
    """Create controller instance from gains dictionary."""
    if controller_type == 'PD':
        return PDControl(Kp=gains['Kp'], Kd=gains['Kd'], saturation_limit=gains.get('saturation_limit', 5.0))
    elif controller_type == 'PID':
        return PIDControl(Kp=gains['Kp'], Kd=gains['Kd'], Ki=gains['Ki'], saturation_limit=gains.get('saturation_limit', 5.0))
    elif controller_type == 'Lyapunov':
        # For Lyapunov, map gains to k1, k2 parameters
        k1 = gains.get('Kp', 0.5)  # Use Kp as k1
        k2 = gains.get('Kd', 0.1)  # Use Kd as k2
        return LyapunovControl(dynamics.I, k1=k1, k2=k2, saturation_limit=gains.get('saturation_limit', 0.1))
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def run_tuned_simulations():
    """Run all simulations with tuned gains."""

    print("🚀 Running Simulations with Tuned Gains")
    print("=" * 50)

    # Load tuned gains
    tuned_gains = load_tuned_gains()

    # Get configurations
    initial_attitude, initial_omega, desired_attitude = ConfigurationGenerator.default_initial_conditions()
    disturbance_cases = ConfigurationGenerator.disturbance_cases()

    # Create dynamics
    dynamics = SpacecraftDynamics()

    # Representations
    representations = {
        'MRP': MRP(),
        'Quaternion': Quaternion(),
        'Euler': EulerAngles()
    }

    # Controllers
    controllers = ['PD', 'PID', 'Lyapunov']

    results = {}

    for rep_name, rep in representations.items():
        for controller_name in controllers:
            for dist_name, disturbance in disturbance_cases.items():

                config_name = f"{rep_name}_{controller_name}_{dist_name}"
                print(f"Running {config_name}...")

                try:
                    # Get tuned gains for this configuration
                    gains = tuned_gains[config_name]

                    # Create controller
                    controller = create_controller(controller_name, gains, dynamics)

                    # Create simulator
                    simulator = SpacecraftSimulator(dynamics, rep, controller, disturbance)

                    # Run simulation
                    state = simulator.simulate(
                        initial_attitude, initial_omega, desired_attitude,
                        t_final=100.0, num_points=2000
                    )

                    # Compute metrics
                    conv_time, conv_idx = PerformanceMetrics.convergence_time(state, threshold=1.0)
                    ss_error = PerformanceMetrics.steady_state_error(state)
                    final_error = PerformanceMetrics.final_error(state)
                    control_effort = PerformanceMetrics.control_effort(state)

                    # Store results
                    results[config_name] = {
                        'convergence_time_s': conv_time,
                        'steady_state_error_deg': np.degrees(ss_error),
                        'final_error_deg': np.degrees(final_error),
                        'control_effort_Nms': control_effort,
                        'gains': gains
                    }

                    status = f"Conv={conv_time:.1f}s" if conv_time != float('inf') else "No convergence"
                    print(f"  ✓ {status}, SS={np.degrees(ss_error):.3f}°, Effort={control_effort:.1f} N·m·s")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    results[config_name] = {
                        'convergence_time_s': float('inf'),
                        'steady_state_error_deg': float('inf'),
                        'final_error_deg': float('inf'),
                        'control_effort_Nms': 0.0,
                        'gains': gains if 'gains' in locals() else {}
                    }

    return results


def save_tuned_results(results: dict, filename: str = "results/tuned_metrics_summary.json"):
    """Save tuned simulation results."""
    Path(filename).parent.mkdir(exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved tuned results to {filename}")


def compare_manual_vs_tuned():
    """Compare manual vs tuned gain performance."""

    print("\n📊 Performance Comparison: Manual vs Tuned Gains")
    print("=" * 60)

    # Load both results
    try:
        with open('results/metrics_summary.json', 'r') as f:
            manual_results = json.load(f)
    except FileNotFoundError:
        print("Manual results not found. Run manual simulations first.")
        return

    try:
        with open('results/tuned_metrics_summary.json', 'r') as f:
            tuned_results = json.load(f)
    except FileNotFoundError:
        print("Tuned results not found. Run tuned simulations first.")
        return

    manual_gains = ConfigurationGenerator.control_gains()

    # Compare key configurations
    configs_to_compare = [
        'MRP_PD_NoDisturbance',
        'MRP_PID_NoDisturbance',
        'Quaternion_PD_NoDisturbance',
        'Quaternion_PID_NoDisturbance'
    ]

    print("-" * 80)
    print(f"{'Config':<28} | {'Manual Conv':<12} | {'Tuned Conv':<12} | {'Manual SS':<10} | {'Tuned SS':<10} | {'Manual Eff':<10} | {'Tuned Eff':<10}")
    print("-" * 80)

    for config in configs_to_compare:
        if config in manual_results and config in tuned_results:
            man = manual_results[config]
            tun = tuned_results[config]
            controller = config.split('_')[1]
            manual_gain = manual_gains.get(controller, {})

            man_conv = f"{man['convergence_time_s']:.1f}s" if man['convergence_time_s'] != float('inf') else "∞"
            tun_conv = f"{tun['convergence_time_s']:.1f}s" if tun['convergence_time_s'] != float('inf') else "∞"
            man_ss = f"{man['steady_state_error_deg']:.3f}°"
            tun_ss = f"{tun['steady_state_error_deg']:.3f}°"
            man_eff = f"{man['control_effort_Nms']:.1f}"
            tun_eff = f"{tun['control_effort_Nms']:.1f}"

            print(f"{config:<28} | {man_conv:<12} | {tun_conv:<12} | {man_ss:<10} | {tun_ss:<10} | {man_eff:<10} | {tun_eff:<10}")
            print(f"  Manual gains: Kp={manual_gain.get('Kp', 'N/A')}, Kd={manual_gain.get('Kd', 'N/A')}, Ki={manual_gain.get('Ki', 'N/A')}")
            print(f"  Tuned gains:  Kp={tun['gains'].get('Kp', 'N/A')}, Kd={tun['gains'].get('Kd', 'N/A')}, Ki={tun['gains'].get('Ki', 'N/A')}")
            print("-" * 80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        results = run_tuned_simulations()
        save_tuned_results(results)
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_manual_vs_tuned()
    else:
        print("Usage:")
        print("  python run_tuned.py run      # Run simulations with tuned gains")
        print("  python run_tuned.py compare  # Compare manual vs tuned performance")