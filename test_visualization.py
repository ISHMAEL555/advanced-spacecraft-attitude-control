"""Quick preview visualization with minimal simulation."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import SpacecraftDynamics
from src.representations import MRP
from src.control import PDControl
from src.simulation import SpacecraftSimulator
from src.analysis import PerformanceMetrics

# Run one quick simulation
dynamics = SpacecraftDynamics()
mrp = MRP()
pd = PDControl(Kp=0.5, Kd=0.1)
simulator = SpacecraftSimulator(dynamics, mrp, pd)

# Initial conditions (in MRP format, not quaternion)
from src.representations import Quaternion
q_init = np.array([0, 0.1, 0.2, np.sqrt(1 - 0.1**2 - 0.2**2)])
sigma_init = Quaternion.to_mrp(q_init)  # Convert to MRP
omega_init = np.array([0.05, -0.02, 0.01])
q_desired = np.array([0, 0, 0, 1])
sigma_desired = Quaternion.to_mrp(q_desired)  # Convert to MRP

# Run with reduced points for speed
state = simulator.simulate(sigma_init, omega_init, sigma_desired, t_final=50.0, num_points=1000)
metrics = PerformanceMetrics.compute_all_metrics(state)

# Create preview plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('MRP + PD Control - Preview', fontsize=14, fontweight='bold')

t = np.array(state.t)
attitude_error = np.array(state.attitude_error)
omega = np.array(state.omega)
u_sat = np.array(state.u_saturated)

# Error norm
ax = axes[0, 0]
error_norm = np.linalg.norm(attitude_error, axis=1)
ax.semilogy(t, error_norm, 'b-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Attitude Error [rad]')
ax.set_title('Attitude Error')
ax.grid(True, alpha=0.3)

# Angular velocity
ax = axes[0, 1]
omega_norm = np.linalg.norm(omega, axis=1)
ax.semilogy(t, omega_norm, 'g-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Angular Velocity [rad/s]')
ax.set_title('Angular Velocity')
ax.grid(True, alpha=0.3)

# Control torque
ax = axes[1, 0]
ax.plot(t, u_sat[:, 0], label='u_x', linewidth=1.5)
ax.plot(t, u_sat[:, 1], label='u_y', linewidth=1.5)
ax.plot(t, u_sat[:, 2], label='u_z', linewidth=1.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Torque [N·m]')
ax.set_title('Control Torque Components')
ax.legend()
ax.grid(True, alpha=0.3)

# Metrics summary
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""
Convergence: {metrics['convergence_time_s']:.3f} s
SS Error: {metrics['steady_state_error_deg']:.4f}°
Control Effort: {metrics['control_effort_Nms']:.3f} N·m·s
Saturation: {metrics['saturation_duration_s']:.3f} s
Final Error: {metrics['final_error_deg']:.4f}°
"""
ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
preview_path = Path('results/preview_plot.png')
preview_path.parent.mkdir(exist_ok=True)
plt.savefig(preview_path, dpi=150, bbox_inches='tight')
print(f"✓ Preview plot saved to {preview_path}")
plt.close()
