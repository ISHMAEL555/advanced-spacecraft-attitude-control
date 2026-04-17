#!/usr/bin/env python
"""Main simulation runner - executes all 18 configurations."""

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

    # Run simulation
    state = simulator.simulate(
        config.initial_attitude,
        config.initial_omega,
        config.desired_attitude,
        t_final=config.t_final,
        num_points=config.num_points
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
    print("SPACECRAFT ATTITUDE CONTROL - FULL SIMULATION MATRIX")
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

    # Monte Carlo analysis for one configuration
    print("\n" + "="*80)
    print("MONTE CARLO ROBUSTNESS ANALYSIS (MRP + PD)")
    print("="*80 + "\n")

    dynamics = SpacecraftDynamics()
    mrp = MRP()
    pd = PDControl()
    simulator = SpacecraftSimulator(dynamics, mrp, pd)

    initial_attitude, initial_omega, desired_attitude = (
        ConfigurationGenerator.default_initial_conditions()
    )

    mc_results = simulator.monte_carlo_analysis(
        initial_attitude,
        initial_omega,
        desired_attitude,
        num_runs=500,
        t_final=100.0,
        num_points=5000
    )

    mc_dir = results_dir / 'monte_carlo'
    mc_dir.mkdir(exist_ok=True)

    with open(mc_dir / 'monte_carlo_results.json', 'w') as f:
        json.dump({
            'convergence_mean': float(mc_results['convergence_mean']),
            'convergence_std': float(mc_results['convergence_std']),
            'convergence_3sigma': float(mc_results['convergence_3sigma_upper']),
            'ss_error_mean_deg': float(np.mean(mc_results['ss_errors']) * 180 / np.pi),
            'ss_error_std_deg': float(np.std(mc_results['ss_errors']) * 180 / np.pi),
        }, f, indent=2)

    print(f"Convergence time: {mc_results['convergence_mean']:.3f} ± {mc_results['convergence_std']:.3f} s")
    print(f"  3σ upper bound: {mc_results['convergence_3sigma_upper']:.3f} s")
    print(f"SS error: {np.mean(mc_results['ss_errors']) * 180 / np.pi:.4f} ± {np.std(mc_results['ss_errors']) * 180 / np.pi:.4f}°")
    print(f"✓ Saved Monte Carlo results to results/monte_carlo/\n")

    print("="*80)
    print("All simulations complete! Results saved to results/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
