"""Analysis and metrics computation."""

import numpy as np
from scipy.integrate import trapezoid
from typing import Dict, Tuple
from src.simulation import SimulationState


class PerformanceMetrics:
    """Compute performance metrics from simulation results."""

    @staticmethod
    def convergence_time(
        state: SimulationState,
        threshold: float = 1.0
    ) -> Tuple[float, int]:
        """
        Compute convergence time (time to reach < threshold degrees).

        Args:
            state: SimulationState with results
            threshold: Convergence threshold [degrees] (default: 1.0 degree)

        Returns:
            (convergence_time, index) or (inf, -1) if not converged
        """
        threshold_rad = threshold * np.pi / 180
        error_norms = np.linalg.norm(state.attitude_error, axis=1)

        conv_idx = np.where(error_norms < threshold_rad)[0]
        if len(conv_idx) > 0:
            return state.t[conv_idx[0]], conv_idx[0]
        else:
            return float('inf'), -1

    @staticmethod
    def steady_state_error(
        state: SimulationState,
        final_fraction: float = 0.1
    ) -> float:
        """
        Compute mean steady-state error over final fraction of simulation.

        Args:
            state: SimulationState with results
            final_fraction: Fraction of trajectory for ss error (default: last 10%)

        Returns:
            Mean error magnitude [rad]
        """
        ss_idx = int((1.0 - final_fraction) * len(state.t))
        error_norms = np.linalg.norm(state.attitude_error, axis=1)
        return np.mean(error_norms[ss_idx:])

    @staticmethod
    def angular_velocity_decay_time(
        state: SimulationState,
        threshold: float = 0.01
    ) -> Tuple[float, int]:
        """
        Compute time to reach |ω| < threshold [rad/s].

        Args:
            state: SimulationState with results
            threshold: Angular velocity threshold [rad/s]

        Returns:
            (decay_time, index) or (inf, -1) if not reached
        """
        omega_norms = np.linalg.norm(state.omega, axis=1)
        decay_idx = np.where(omega_norms < threshold)[0]

        if len(decay_idx) > 0:
            return state.t[decay_idx[0]], decay_idx[0]
        else:
            return float('inf'), -1

    @staticmethod
    def control_effort(state: SimulationState) -> float:
        """
        Compute total control effort: ∫ |u(t)| dt.

        Args:
            state: SimulationState with results

        Returns:
            Total control effort [N·m·s]
        """
        # Compute magnitude at each point
        u_norms = np.linalg.norm(state.u_saturated, axis=1)

        # Trapezoidal integration
        effort = trapezoid(u_norms, state.t)
        return effort

    @staticmethod
    def saturation_duration(state: SimulationState) -> float:
        """
        Compute total time spent in saturation.

        Args:
            state: SimulationState with results

        Returns:
            Total saturation time [s]
        """
        if len(state.saturation_times) == 0:
            return 0.0

        # Time spent saturated
        sat_times = np.array(state.saturation_times)
        if len(sat_times) == 0:
            return 0.0

        # Count unique time intervals
        # Simple approach: count number of time steps in saturation
        total_t = state.t[-1] - state.t[0]
        dt = total_t / (len(state.t) - 1) if len(state.t) > 1 else 0

        return len(sat_times) * dt

    @staticmethod
    def compute_all_metrics(state: SimulationState) -> Dict:
        """
        Compute all performance metrics.

        Args:
            state: SimulationState with results

        Returns:
            Dictionary of all metrics
        """
        conv_time, _ = PerformanceMetrics.convergence_time(state)
        decay_time, _ = PerformanceMetrics.angular_velocity_decay_time(state)
        ss_error = PerformanceMetrics.steady_state_error(state)
        effort = PerformanceMetrics.control_effort(state)
        sat_time = PerformanceMetrics.saturation_duration(state)

        return {
            'convergence_time_s': conv_time,
            'steady_state_error_rad': ss_error,
            'steady_state_error_deg': ss_error * 180 / np.pi,
            'angular_velocity_decay_time_s': decay_time,
            'control_effort_Nms': effort,
            'saturation_duration_s': sat_time,
            'final_error_rad': np.linalg.norm(state.attitude_error[-1]),
            'final_error_deg': np.linalg.norm(state.attitude_error[-1]) * 180 / np.pi,
            'final_omega_rad_s': np.linalg.norm(state.omega[-1]),
        }


class SimulationComparison:
    """Compare results across multiple simulation configurations."""

    @staticmethod
    def create_comparison_table(
        configs: Dict[str, SimulationState],
        metric_names: list = None
    ) -> Dict:
        """
        Create comparison table for multiple configurations.

        Args:
            configs: Dictionary {config_name: SimulationState}
            metric_names: List of metrics to include

        Returns:
            Dictionary with comparison data
        """
        if metric_names is None:
            metric_names = [
                'convergence_time_s',
                'steady_state_error_deg',
                'control_effort_Nms',
                'saturation_duration_s'
            ]

        results = {}
        for config_name, state in configs.items():
            metrics = PerformanceMetrics.compute_all_metrics(state)
            results[config_name] = {
                name: metrics.get(name, 'N/A')
                for name in metric_names
            }

        return results

    @staticmethod
    def format_comparison_table(comparison: Dict) -> str:
        """Format comparison results as a readable table."""
        if not comparison:
            return "No data to display"

        # Get all keys
        configs = list(comparison.keys())
        metrics = list(comparison[configs[0]].keys()) if configs else []

        # Header
        header = "Config".ljust(30) + " | " + " | ".join(m.ljust(20) for m in metrics)
        separator = "-" * (len(header))

        # Rows
        rows = [header, separator]
        for config in configs:
            values = comparison[config]
            row_str = config.ljust(30) + " | "
            row_values = []
            for metric in metrics:
                val = values.get(metric, 'N/A')
                if isinstance(val, float):
                    row_values.append(f"{val:.6f}".ljust(20))
                else:
                    row_values.append(str(val).ljust(20))
            row_str += " | ".join(row_values)
            rows.append(row_str)

        return "\n".join(rows)
