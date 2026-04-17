"""Visualization script for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path


def plot_single_simulation(config_name, state, metrics):
    """Create comprehensive plots for a single simulation."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{config_name}', fontsize=14, fontweight='bold')

    t = np.array(state.t)
    attitude_error = np.array(state.attitude_error)
    omega = np.array(state.omega)
    u_sat = np.array(state.u_saturated)

    # Row 1: Attitude error and angular velocity
    ax = axes[0, 0]
    error_norm = np.linalg.norm(attitude_error, axis=1)
    ax.semilogy(t, error_norm, 'b-', linewidth=1.5, label='|error|')
    ax.axhline(y=0.5*np.pi/180, color='r', linestyle='--', label='Conv. threshold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [rad]')
    ax.set_title('Attitude Error')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    omega_norm = np.linalg.norm(omega, axis=1)
    ax.semilogy(t, omega_norm, 'g-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity')
    ax.grid(True, alpha=0.3)

    # Row 2: Control torque components
    ax = axes[1, 0]
    ax.plot(t, u_sat[:, 0], label='u_x', linewidth=1)
    ax.plot(t, u_sat[:, 1], label='u_y', linewidth=1)
    ax.plot(t, u_sat[:, 2], label='u_z', linewidth=1)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Sat. limit')
    ax.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque [N·m]')
    ax.set_title('Control Torque')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Control effort
    ax = axes[1, 1]
    u_norm = np.linalg.norm(u_sat, axis=1)
    ax.plot(t, u_norm, 'k-', linewidth=1.5)
    ax.fill_between(t, 0, u_norm, alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|u(t)| [N·m]')
    ax.set_title(f'Control Effort = {metrics["control_effort_Nms"]:.3f} N·m·s')
    ax.grid(True, alpha=0.3)

    # Row 3: Attitude error components
    ax = axes[2, 0]
    ax.plot(t, attitude_error[:, 0], label='σ_x', linewidth=1)
    ax.plot(t, attitude_error[:, 1], label='σ_y', linewidth=1)
    ax.plot(t, attitude_error[:, 2], label='σ_z', linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error Component [rad]')
    ax.set_title('Attitude Error Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: Metrics summary
    ax = axes[2, 1]
    ax.axis('off')
    metrics_text = f"""
    Convergence Time: {metrics['convergence_time_s']:.3f} s
    SS Error: {metrics['steady_state_error_deg']:.4f}°
    Control Effort: {metrics['control_effort_Nms']:.3f} N·m·s
    Saturation Time: {metrics['saturation_duration_s']:.3f} s
    Final Error: {metrics['final_error_deg']:.4f}°
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def create_comparison_plot(comparison_data):
    """Create comparison plot for all 18 configurations."""
    configs = list(comparison_data.keys())
    n_configs = len(configs)

    conv_times = [comparison_data[c]['convergence_time_s'] for c in configs]
    ss_errors = [comparison_data[c]['steady_state_error_deg'] for c in configs]
    efforts = [comparison_data[c]['control_effort_Nms'] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Performance Comparison Across 18 Configurations', fontsize=14, fontweight='bold')

    # Convergence time
    ax = axes[0]
    colors = ['C0' if 'PD' in c else 'C1' if 'PID' in c else 'C2' for c in configs]
    ax.bar(range(n_configs), conv_times, color=colors, alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Convergence Time [s]')
    ax.set_title('Convergence Time')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Steady-state error
    ax = axes[1]
    ax.bar(range(n_configs), ss_errors, color=colors, alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('SS Error [degrees]')
    ax.set_title('Steady-State Error')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Control effort
    ax = axes[2]
    ax.bar(range(n_configs), efforts, color=colors, alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Control Effort [N·m·s]')
    ax.set_title('Total Control Effort')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', alpha=0.7, label='PD'),
                       Patch(facecolor='C1', alpha=0.7, label='PID'),
                       Patch(facecolor='C2', alpha=0.7, label='Lyapunov')]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    return fig


def generate_visualizations():
    """Generate all visualizations from results."""
    results_dir = Path('results')

    print("Loading results...")
    with open(results_dir / 'simulation_results.pkl', 'rb') as f:
        results = pickle.load(f)

    with open(results_dir / 'metrics_summary.json', 'r') as f:
        metrics_summary = json.load(f)

    # Create output directory
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    print(f"Generating {len(results)} configuration plots...")
    for config_name, result in results.items():
        metrics = metrics_summary[config_name]
        state = result['state']
        fig = plot_single_simulation(config_name, state, metrics)
        fig.savefig(vis_dir / f'{config_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {config_name}")

    print("Generating comparison plot...")
    fig = create_comparison_plot(metrics_summary)
    fig.savefig(vis_dir / 'comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Visualizations saved to results/visualizations/")


if __name__ == '__main__':
    generate_visualizations()
