"""Visualization script for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path


def plot_single_simulation(config_name, state, metrics):
    """Create comprehensive plots for a single simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))  # Reduced to 2x2
    fig.suptitle(f'{config_name}', fontsize=14, fontweight='bold')

    t = np.array(state.t)
    attitude_error = np.array(state.attitude_error)
    omega = np.array(state.omega)
    u_sat = np.array(state.u_saturated)

    # Row 1: Attitude error and angular velocity norms
    ax = axes[0, 0]
    error_norm = np.clip(np.linalg.norm(attitude_error, axis=1), 1e-12, None)
    ax.semilogy(t, error_norm, 'b-', linewidth=1.5, label='|error|')
    ax.axhline(y=0.5*np.pi/180, color='r', linestyle='--', label='Conv. threshold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [rad]')
    ax.set_title('Attitude Error Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    omega_norm = np.clip(np.linalg.norm(omega, axis=1), 1e-12, None)
    ax.semilogy(t, omega_norm, 'g-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Norm')
    ax.grid(True, alpha=0.3)

    # Row 2: Control torque norm and metrics
    ax = axes[1, 0]
    u_norm = np.linalg.norm(u_sat, axis=1)
    ax.plot(t, u_norm, 'k-', linewidth=1.5)
    ax.fill_between(t, 0, u_norm, alpha=0.3)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Sat. limit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|u(t)| [N·m]')
    ax.set_title(f'Control Torque Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Metrics summary
    ax = axes[1, 1]
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


def create_results_table(metrics_summary):
    """Create a comprehensive results table for all configurations."""
    # Prepare data in deterministic order
    configs = sorted(metrics_summary.keys(), key=lambda c: (
        c.split('_')[0],
        ['PD', 'PID', 'Lyapunov'].index(c.split('_')[1]),
        c.split('_')[2]
    ))
    data = []
    for config in configs:
        rep, ctrl, dist = config.split('_')
        metrics = metrics_summary[config]
        row = [rep, ctrl, dist,
               f"{metrics['convergence_time_s']:.3f}",
               f"{metrics['steady_state_error_deg']:.4f}",
               f"{metrics['control_effort_Nms']:.3f}",
               f"{metrics['saturation_duration_s']:.3f}",
               f"{metrics['final_error_deg']:.4f}"]
        data.append(row)

    # Column headers
    columns = ['Rep.', 'Control', 'Dist.', 'Conv. Time [s]', 'SS Error [°]',
               'Control Effort [N·m·s]', 'Sat. Time [s]', 'Final Error [°]']

    # Create matplotlib table
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f0f0f0']*len(columns))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code by representation
    colors = {'MRP': '#e6f3ff', 'Quaternion': '#fff2e6', 'Euler': '#f0fff0'}
    for i, row in enumerate(data, 1):  # Start from 1 since 0 is header
        rep = row[0]
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(colors.get(rep, 'white'))

    plt.title('Simulation Results Summary', fontsize=14, fontweight='bold', pad=20)
    return fig


def create_grouped_comparison_plots(metrics_summary):
    """Create comparison plots grouped by representation and disturbance."""
    # Group data manually
    groups = {}
    for config, metrics in metrics_summary.items():
        rep, ctrl, dist = config.split('_')
        key = (rep, dist)
        if key not in groups:
            groups[key] = {}
        groups[key][ctrl] = metrics

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Grouped Performance Comparison', fontsize=14, fontweight='bold')

    metrics_keys = ['convergence_time_s', 'steady_state_error_deg', 'control_effort_Nms']
    metric_names = ['Convergence Time [s]', 'Steady-State Error [°]', 'Control Effort [N·m·s]']

    row = 0
    col = 0
    for (rep, dist), group_data in groups.items():
        ax = axes[row, col]
        controls = list(group_data.keys())
        x = np.arange(len(controls))

        for i, metric in enumerate(metrics_keys):
            values = [group_data[ctrl][metric] for ctrl in controls]
            ax.bar(x + i*0.25, values, width=0.25, label=metric_names[i], alpha=0.7)

        ax.set_title(f'{rep} - {dist}')
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(controls)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        col += 1
        if col == 2:
            col = 0
            row += 1

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

    print("Generating results table...")
    fig = create_results_table(metrics_summary)
    fig.savefig(vis_dir / 'results_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Generating grouped comparison plots...")
    fig = create_grouped_comparison_plots(metrics_summary)
    fig.savefig(vis_dir / 'grouped_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Visualizations saved to results/visualizations/")


if __name__ == '__main__':
    generate_visualizations()
