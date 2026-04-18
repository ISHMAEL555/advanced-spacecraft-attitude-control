"""Control law implementations: PD, PID, and Lyapunov-based control."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable
from scipy.optimize import minimize_scalar, differential_evolution


class GainTuner:
    """Automatic gain tuning for control laws."""

    @staticmethod
    def tune_pd_gains(
        dynamics: 'SpacecraftDynamics',
        attitude_rep: 'AttitudeRepresentation',
        initial_attitude: np.ndarray,
        initial_omega: np.ndarray,
        desired_attitude: np.ndarray,
        disturbance_torque: np.ndarray = None,
        method: str = 'optimization'
    ) -> Dict[str, float]:
        """
        Automatically tune PD gains using optimization.

        Args:
            dynamics: Spacecraft dynamics
            attitude_rep: Attitude representation
            initial_attitude: Initial attitude state
            initial_omega: Initial angular velocity
            desired_attitude: Desired attitude state
            disturbance_torque: External disturbance
            method: Tuning method ('optimization', 'ziegler_nichols', 'manual')

        Returns:
            Dictionary of tuned gains
        """
        if method == 'ziegler_nichols':
            return GainTuner._ziegler_nichols_pd(dynamics.I)
        elif method == 'optimization':
            return GainTuner._optimize_pd_gains(
                dynamics, attitude_rep, initial_attitude,
                initial_omega, desired_attitude, disturbance_torque
            )
        else:
            # Manual tuning based on inertia
            I_max = np.max(np.diag(dynamics.I))
            Kp = 50.0 * I_max / 10.0  # Scale with inertia
            Kd = 2.0 * np.sqrt(Kp * I_max)  # Critical damping
            return {'Kp': Kp, 'Kd': Kd, 'saturation_limit': 5.0}

    @staticmethod
    def _ziegler_nichols_pd(inertia_matrix: np.ndarray) -> Dict[str, float]:
        """Ziegler-Nichols tuning for PD control."""
        I_diag = np.diag(inertia_matrix)
        I_avg = np.mean(I_diag)

        # Ziegler-Nichols for position control
        Ku = 100.0  # Ultimate gain (estimated)
        Tu = 2.0 * np.pi * np.sqrt(I_avg / 10.0)  # Ultimate period

        Kp = 0.5 * Ku
        Kd = Kp * Tu / 8.0

        return {'Kp': Kp, 'Kd': Kd, 'saturation_limit': Ku / 10.0}

    @staticmethod
    def _optimize_pd_gains(
        dynamics: 'SpacecraftDynamics',
        attitude_rep: 'AttitudeRepresentation',
        initial_attitude: np.ndarray,
        initial_omega: np.ndarray,
        desired_attitude: np.ndarray,
        disturbance_torque: np.ndarray = None
    ) -> Dict[str, float]:
        """Optimize PD gains using numerical optimization."""

        def objective(gains_log):
            """Objective function to minimize."""
            Kp = 10 ** gains_log[0]  # Log scale for optimization
            Kd = 10 ** gains_log[1]

            # Create controller
            controller = PDControl(Kp=Kp, Kd=Kd)

            # Create simulator (avoid circular import by using the classes directly)
            from ..dynamics import SpacecraftDynamics
            from ..simulation import SpacecraftSimulator
            from ..analysis import PerformanceMetrics

            simulator = SpacecraftSimulator(dynamics, attitude_rep, controller, disturbance_torque)

            try:
                # Run simulation
                state = simulator.simulate(initial_attitude, initial_omega, desired_attitude,
                                         t_final=50.0, num_points=1000)

                # Compute metrics
                conv_time, _ = PerformanceMetrics.convergence_time(state, threshold=1.0)
                ss_error = PerformanceMetrics.steady_state_error(state)
                effort = PerformanceMetrics.control_effort(state)

                # Objective: minimize convergence time + steady-state error + control effort
                if conv_time == float('inf'):
                    return 1000.0  # Large penalty for no convergence

                objective_value = conv_time + 100.0 * ss_error * 180.0/np.pi + 0.001 * effort
                return objective_value

            except:
                return 1000.0  # Penalty for simulation failure

        # Optimize in log space
        bounds = [(-1, 3), (-1, 2)]  # Kp: 0.1-1000, Kd: 0.1-100

        try:
            result = differential_evolution(objective, bounds, maxiter=20, popsize=10,
                                          mutation=(0.5, 1.0), recombination=0.7)

            Kp_opt = 10 ** result.x[0]
            Kd_opt = 10 ** result.x[1]

            return {'Kp': Kp_opt, 'Kd': Kd_opt, 'saturation_limit': 5.0}

        except:
            # Fallback to manual tuning
            return GainTuner.tune_pd_gains(dynamics, attitude_rep, initial_attitude,
                                         initial_omega, desired_attitude, disturbance_torque, 'manual')


class ControlLaw(ABC):
    """Abstract base class for control laws."""

    @abstractmethod
    def compute_torque(
        self,
        error_attitude: np.ndarray,
        angular_velocity: np.ndarray,
        time: float = 0.0
    ) -> np.ndarray:
        """
        Compute control torque.

        Args:
            error_attitude: Attitude error
            angular_velocity: Angular velocity [rad/s]
            time: Current simulation time

        Returns:
            Control torque [N·m]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal states (e.g., integrators)."""
        pass


class PDControl(ControlLaw):
    """Proportional-Derivative (PD) control law."""

    def __init__(self, Kp: float = 50.0, Kd: float = 10.0, saturation_limit: float = 5.0):
        """
        Initialize PD controller.

        Args:
            Kp: Proportional gain
            Kd: Derivative gain
            saturation_limit: Actuator saturation limit [N·m]
        """
        self.Kp = Kp
        self.Kd = Kd
        self.sat_limit = saturation_limit

    def compute_torque(
        self,
        error_attitude: np.ndarray,
        angular_velocity: np.ndarray,
        time: float = 0.0
    ) -> np.ndarray:
        """
        PD control law: u = -Kp · e_att - Kd · ω
        """
        u = -self.Kp * error_attitude - self.Kd * angular_velocity
        return self._saturate(u)

    def _saturate(self, u: np.ndarray) -> np.ndarray:
        """Apply component-wise saturation."""
        return np.clip(u, -self.sat_limit, self.sat_limit)

    def reset(self):
        """No internal state to reset."""
        pass


class PIDControl(ControlLaw):
    """Proportional-Integral-Derivative (PID) control with anti-windup."""

    def __init__(
        self,
        Kp: float = 0.5,
        Kd: float = 0.1,
        Ki: float = 0.05,
        saturation_limit: float = 0.1
    ):
        """
        Initialize PID controller.

        Args:
            Kp: Proportional gain
            Kd: Derivative gain
            Ki: Integral gain
            saturation_limit: Actuator saturation limit [N·m]
        """
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.sat_limit = saturation_limit

        # Integral state
        self.integral_error = np.zeros(3)

    def compute_torque(
        self,
        error_attitude: np.ndarray,
        angular_velocity: np.ndarray,
        time: float = 0.0
    ) -> np.ndarray:
        """
        PID control law with back-calculation anti-windup:
        u = -Kp · e_att - Kd · ω - Ki · ∫e_att dt
        """
        # Control torque (before saturation)
        u_unsat = (
            -self.Kp * error_attitude
            - self.Kd * angular_velocity
            - self.Ki * self.integral_error
        )

        # Apply saturation
        u_sat = self._saturate(u_unsat)

        # Anti-windup back-calculation
        # ė_int = e_att + (u_sat - u_unsat) / Ki
        # This is done externally via update_integral_state()

        return u_sat

    def update_integral_state(
        self,
        error_attitude: np.ndarray,
        u_sat: np.ndarray,
        u_unsat: np.ndarray,
        dt: float
    ):
        """
        Update integral state with back-calculation anti-windup.

        Args:
            error_attitude: Current attitude error
            u_sat: Saturated control torque
            u_unsat: Unsaturated control torque
            dt: Time step
        """
        # Anti-windup: reduce integrator rate when saturated
        e_int_rate = error_attitude + (u_sat - u_unsat) / (self.Ki + 1e-10)
        self.integral_error += e_int_rate * dt

    def _saturate(self, u: np.ndarray) -> np.ndarray:
        """Apply component-wise saturation."""
        return np.clip(u, -self.sat_limit, self.sat_limit)

    def reset(self):
        """Reset integral state."""
        self.integral_error = np.zeros(3)


class LyapunovControl(ControlLaw):
    """Lyapunov-based nonlinear control law."""

    def __init__(
        self,
        inertia: np.ndarray,
        k1: float = 0.5,
        k2: float = 0.1,
        saturation_limit: float = 0.1
    ):
        """
        Initialize Lyapunov-based controller.

        Args:
            inertia: Spacecraft inertia matrix
            k1: MRP error feedback gain
            k2: Angular velocity feedback gain
            saturation_limit: Actuator saturation limit [N·m]
        """
        self.I = inertia
        self.k1 = k1
        self.k2 = k2
        self.sat_limit = saturation_limit

    def compute_torque(
        self,
        error_attitude: np.ndarray,
        angular_velocity: np.ndarray,
        time: float = 0.0
    ) -> np.ndarray:
        """
        Lyapunov-based control law derived from:
        V = (1/2) · ωᵀ · I · ω + k · Φ(e_att)

        For MRP: Φ(σ) = ln(1 + |σ|²)

        u = -k1 · σ / (1 + |σ|²) - k2 · ω
        """
        norm_sq = np.dot(error_attitude, error_attitude)

        # Stability-driven gain scheduling
        attitude_gain = self.k1 / (1.0 + norm_sq)

        u = -attitude_gain * error_attitude - self.k2 * angular_velocity

        # Apply saturation
        u_sat = self._saturate(u)

        return u_sat

    def _saturate(self, u: np.ndarray) -> np.ndarray:
        """Apply component-wise saturation."""
        return np.clip(u, -self.sat_limit, self.sat_limit)

    def reset(self):
        """No internal state to reset."""
        pass
