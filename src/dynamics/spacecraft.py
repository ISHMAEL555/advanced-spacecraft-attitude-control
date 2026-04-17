"""Spacecraft dynamics model - Euler's rotational equations."""

import numpy as np
from typing import Tuple


class SpacecraftDynamics:
    """Rigid spacecraft dynamics under actuator constraints and external disturbances."""

    def __init__(self, inertia: np.ndarray = None):
        """
        Initialize spacecraft with inertia matrix.

        Args:
            inertia: 3x3 inertia matrix (default: diag(10, 20, 30) kg·m²)
        """
        if inertia is None:
            self.I = np.diag([10.0, 20.0, 30.0])
        else:
            self.I = np.array(inertia, dtype=float)

        # Precompute inverse for efficiency
        self.I_inv = np.linalg.inv(self.I)

    def angular_velocity_derivative(
        self,
        omega: np.ndarray,
        control_torque: np.ndarray,
        disturbance_torque: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute angular velocity derivative from Euler's rotational equation.

        I · ω̇ + ω × (I · ω) = u + L

        Args:
            omega: Angular velocity [rad/s] (3-element array)
            control_torque: Control torque u [N·m] (3-element array)
            disturbance_torque: External disturbance torque L [N·m]

        Returns:
            Angular velocity derivative ω̇ (3-element array)
        """
        if disturbance_torque is None:
            disturbance_torque = np.zeros(3)

        # Gyroscopic torque: ω × (I · ω)
        Iw = self.I @ omega
        gyroscopic = np.cross(omega, Iw)

        # Solve for angular acceleration
        # ω̇ = I⁻¹ · (u + L - ω × (I · ω))
        total_torque = control_torque + disturbance_torque - gyroscopic
        omega_dot = self.I_inv @ total_torque

        return omega_dot

    def get_inertia(self) -> np.ndarray:
        """Return the inertia matrix."""
        return self.I.copy()

    def get_inertia_inv(self) -> np.ndarray:
        """Return the inverse inertia matrix."""
        return self.I_inv.copy()
