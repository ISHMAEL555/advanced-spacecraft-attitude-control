#!/usr/bin/env python
"""Reduced simulation runner - faster execution for demonstration."""

import numpy as np
import pickle
import json
from pathlib import Path

from configs.config import ConfigurationGenerator, SimulationConfig
from src.dynamics import SpacecraftDynamics
from src.representations import MRP, Quaternion, EulerAngles
from src.control import PDControl, PIDControl, LyapunovControl
from src.simulation import SpacecraftSimulator
from src.analysis import PerformanceMetrics, SimulationComparison


def create_control_law(config: SimulationConfig):
    """Instantiate control law from configuration."""
    name = config.control_name
    gains = config.control_gains

    if name == 'PD':
        return PDControl(**gains)
    elif name == 'PID':
        return PIDControl(**gains)
    elif name == 'Lyapunov':
        dynamics = SpacecraftDynamics()
        return LyapunovControl(dynamics.get_inertia(), **gains)
    else:
        raise ValueError(f"Unknown control law: {name}")


def create_attitude_representation(config: SimulationConfig):
    """Instantiate attitude representation from configuration."""
    name = config.representation_name

    if name == 'MRP':
        return MRP()
    elif name == 'Quaternion':
        return Quaternion()
    elif name == 'Euler':
        return EulerAngles()
    else:
        raise ValueError(f"Unknown representation: {name}")


def run_single_simulation(config: SimulationConfig) -> dict:
    """Run a single simulation configuration."""
    print(f"  Running {config}...", end=' ')

    # Create components
    dynamics = SpacecraftDynamics()
    attitude_rep = create_attitude_representation(config)
    control_law = create_control_law(config)

    # Create simulator
    simulator = SpacecraftSimulator(
        dynamics,
        attitude_rep,
        control_law,
        config.disturbance_torque
    )

    # Run simulation with fewer points for speed
    state = simulator.simulate(
        config.initial_attitude,
        config.initial_omega,
        config.desired_attitude,
        t_final=config.t_final,
        num_points=2000  # Reduced from 10000
    )

    # Compute metrics
    metrics = PerformanceMetrics.compute_all_metrics(state)
    metrics['config'] = str(config)
    metrics['state'] = state

    print("✓")
    return metrics


def main():
    """Run all 18 simulations."""
    print("\n" + "="*80)
    print("SPACECRAFT ATTITUDE CONTROL - FULL SIMULATION MATRIX (REDUCED)")
    print("="*80 + "\n")

    # Generate all configurations
    configs = ConfigurationGenerator.generate_matrix()
    ConfigurationGenerator.print_matrix(configs)

    print("\nRunning simulations...")
    print("-" * 80)

    results = {}
    for i, config in enumerate(configs, 1):
        print(f"[{i}/18]", end=' ')
        result = run_single_simulation(config)
        results[str(config)] = result

    print("\n" + "-" * 80)

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Save detailed results (pickle)
    with open(results_dir / 'simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved detailed results to results/simulation_results.pkl")

    # Create comparison metrics (JSON-serializable)
    comparison_data = {}
    for config_name, result in results.items():
        comparison_data[config_name] = {
            'convergence_time_s': float(result['convergence_time_s']),
            'steady_state_error_deg': float(result['steady_state_error_deg']),
            'control_effort_Nms': float(result['control_effort_Nms']),
            'saturation_duration_s': float(result['saturation_duration_s']),
            'final_error_deg': float(result['final_error_deg']),
        }

    with open(results_dir / 'metrics_summary.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Saved metrics summary to results/metrics_summary.json")

    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80 + "\n")

    comparison = SimulationComparison.create_comparison_table(
        {k: v['state'] for k, v in results.items()}
    )
    print(SimulationComparison.format_comparison_table(comparison))

    print("\n" + "="*80)
    print("All simulations complete! Results saved to results/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
