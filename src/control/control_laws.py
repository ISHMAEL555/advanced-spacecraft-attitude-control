"""Control law implementations: PD, PID, and Lyapunov-based control."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


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

    def __init__(self, Kp: float = 0.5, Kd: float = 0.1):
        """
        Initialize PD controller.

        Args:
            Kp: Proportional gain
            Kd: Derivative gain
        """
        self.Kp = Kp
        self.Kd = Kd

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
        return u

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
