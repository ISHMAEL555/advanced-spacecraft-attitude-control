"""Attitude representations: MRP, Quaternion, and Euler angles."""

import numpy as np
from abc import ABC, abstractmethod


class AttitudeRepresentation(ABC):
    """Abstract base class for attitude representations."""

    @abstractmethod
    def kinematics(self, state: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute attitude state derivative from angular velocity.

        Args:
            state: Current attitude state
            omega: Angular velocity [rad/s]

        Returns:
            State derivative
        """
        pass

    @abstractmethod
    def error_state(self, state_current: np.ndarray, state_desired: np.ndarray) -> np.ndarray:
        """
        Compute attitude error.

        Args:
            state_current: Current attitude
            state_desired: Desired attitude

        Returns:
            Attitude error
        """
        pass

    @abstractmethod
    def to_matrix(self, state: np.ndarray) -> np.ndarray:
        """Convert to 3x3 direction cosine matrix (DCM)."""
        pass


class MRP(AttitudeRepresentation):
    """Modified Rodriguez Parameters (MRP) representation."""

    @staticmethod
    def to_quaternion(sigma: np.ndarray) -> np.ndarray:
        """Convert MRP to quaternion (scalar-last convention)."""
        norm_sq = np.dot(sigma, sigma)

        # q = [(2*σ) / (1 + |σ|²), (1 - |σ|²) / (1 + |σ|²)]
        denom = 1.0 + norm_sq
        v = (2.0 * sigma) / denom
        w = (1.0 - norm_sq) / denom

        return np.array([v[0], v[1], v[2], w])

    @staticmethod
    def to_euler_angles(sigma: np.ndarray) -> np.ndarray:
        """Convert MRP to Euler angles (3-2-1 convention)."""
        # First convert to DCM
        mrp = MRP()
        C = mrp.to_matrix(sigma)

        # Extract Euler angles from DCM (3-2-1 convention)
        # C = Rz(ψ) * Ry(θ) * Rx(φ)
        theta = -np.arcsin(C[2, 0])
        phi = np.arctan2(C[2, 1], C[2, 2])
        psi = np.arctan2(C[1, 0], C[0, 0])

        return np.array([phi, theta, psi])

    @staticmethod
    def shadow_check(sigma: np.ndarray) -> np.ndarray:
        """
        Apply MRP shadow set switching for |σ| > 1.

        Args:
            sigma: MRP vector

        Returns:
            Corrected MRP vector
        """
        norm_sq = np.dot(sigma, sigma)
        if norm_sq > 1.0:
            return -sigma / norm_sq
        return sigma

    def kinematics(self, sigma: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        MRP kinematic equation: σ̇ = (1/4) · [σ×] · ω + (1/4) · (1 - |σ|²) · ω

        Where [σ×] is the skew-symmetric matrix.
        """
        norm_sq = np.dot(sigma, sigma)

        # Skew-symmetric matrix of sigma
        sigma_skew = np.array([
            [0, -sigma[2], sigma[1]],
            [sigma[2], 0, -sigma[0]],
            [-sigma[1], sigma[0], 0]
        ])

        sigma_dot = 0.25 * (sigma_skew @ omega + (1 - norm_sq) * omega)
        return sigma_dot

    def error_state(self, sigma_current: np.ndarray, sigma_desired: np.ndarray) -> np.ndarray:
        """
        Compute MRP error as the MRP of relative rotation.

        For MRP, when sigma_desired = [0,0,0] (identity), error = sigma_current.
        Otherwise, use the composition formula.
        """
        # Special case: desired is identity
        if np.allclose(sigma_desired, np.zeros(3)):
            return MRP.shadow_check(sigma_current)

        # General case: MRP composition for relative rotation
        sigma_d = sigma_desired
        sigma_c = sigma_current

        norm_d_sq = np.dot(sigma_d, sigma_d)
        norm_c_sq = np.dot(sigma_c, sigma_c)

        numerator = (
            sigma_d * norm_c_sq
            - sigma_c * norm_d_sq
            - np.cross(sigma_d, sigma_c)
        )

        denominator = 1.0 + np.dot(sigma_d, sigma_c)

        if abs(denominator) < 1e-10:
            return np.zeros(3)

        sigma_err = numerator / denominator
        return MRP.shadow_check(sigma_err)

    def to_matrix(self, sigma: np.ndarray) -> np.ndarray:
        """Convert MRP to direction cosine matrix."""
        norm_sq = np.dot(sigma, sigma)

        sigma_skew = np.array([
            [0, -sigma[2], sigma[1]],
            [sigma[2], 0, -sigma[0]],
            [-sigma[1], sigma[0], 0]
        ])

        # C = I + (8[σ×]² - 4(1-|σ|²)[σ×]) / (1+|σ|²)²
        numerator = 8 * sigma_skew @ sigma_skew - 4 * (1 - norm_sq) * sigma_skew
        denominator = (1 + norm_sq) ** 2

        C = np.eye(3) + numerator / denominator
        return C


class Quaternion(AttitudeRepresentation):
    """Quaternion representation (scalar-last convention: [x, y, z, w])."""

    @staticmethod
    def from_mrp(sigma: np.ndarray) -> np.ndarray:
        """Convert MRP to quaternion (scalar-last convention)."""
        norm_sq = np.dot(sigma, sigma)

        # q = [(2*σ) / (1 + |σ|²), (1 - |σ|²) / (1 + |σ|²)]
        denom = 1.0 + norm_sq
        v = (2.0 * sigma) / denom
        w = (1.0 - norm_sq) / denom

        return np.array([v[0], v[1], v[2], w])

    @staticmethod
    def to_mrp(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to MRP."""
        q = Quaternion.normalize(q)
        v, w = q[:3], q[3]

        # σ = v / (1 + w)
        if abs(1.0 + w) < 1e-10:
            return np.zeros(3)

        sigma = v / (1.0 + w)
        return sigma

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion."""
        return q / np.linalg.norm(q)

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Return quaternion conjugate."""
        q_conj = q.copy()
        q_conj[:3] = -q_conj[:3]
        return q_conj

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication q1 ⊗ q2 (scalar-last)."""
        v1, w1 = q1[:3], q1[3]
        v2, w2 = q2[:3], q2[3]

        v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        w = w1 * w2 - np.dot(v1, v2)

        return np.array([v[0], v[1], v[2], w])

    def kinematics(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Quaternion kinematic equation: q̇ = (1/2) · Ω(ω) · q

        Where Ω is a matrix function of angular velocity.

        Note: Quaternion normalization should be enforced periodically
        during integration to prevent numerical drift.
        """
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])

        omega_matrix = np.zeros((4, 4))
        omega_matrix[:3, :3] = omega_skew
        omega_matrix[:3, 3] = omega
        omega_matrix[3, :3] = -omega

        q_dot = 0.5 * omega_matrix @ q
        return q_dot

    def error_state(self, q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
        """
        Compute quaternion error with unwinding prevention.

        q_err = q_desired ⊗ q_current⁻¹
        """
        q_current_inv = self.conjugate(q_current)
        q_err = self.multiply(q_desired, q_current_inv)

        # Unwinding prevention: keep scalar part positive
        if q_err[3] < 0:
            q_err = -q_err

        return q_err

    def to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to direction cosine matrix."""
        q = self.normalize(q)
        x, y, z, w = q

        C = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        return C


class EulerAngles(AttitudeRepresentation):
    """Euler angles representation (3-2-1 convention)."""

    @staticmethod
    def from_mrp(sigma: np.ndarray) -> np.ndarray:
        """Convert MRP to Euler angles (3-2-1 convention)."""
        # First convert to DCM
        mrp = MRP()
        C = mrp.to_matrix(sigma)

        # Extract Euler angles from DCM (3-2-1 convention)
        # C = Rz(ψ) * Ry(θ) * Rx(φ)
        theta = -np.arcsin(C[2, 0])
        phi = np.arctan2(C[2, 1], C[2, 2])
        psi = np.arctan2(C[1, 0], C[0, 0])

        return np.array([phi, theta, psi])

    def kinematics(self, angles: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Euler angle rate from angular velocity (3-2-1 convention).

        [φ̇, θ̇, ψ̇]ᵀ = E · ω
        """
        phi, theta, psi = angles

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Avoid numerical blow-up near 3-2-1 singularity (theta -> ±90 deg)
        eps = 1e-6
        if abs(cos_theta) < eps:
            cos_theta = np.sign(cos_theta) * eps if cos_theta != 0.0 else eps

        tan_theta = sin_theta / cos_theta

        # E matrix
        E = np.array([
            [1, sin_phi * tan_theta, cos_phi * tan_theta],
            [0, cos_phi, -sin_phi],
            [0, sin_phi / cos_theta, cos_phi / cos_theta]
        ])

        angles_dot = E @ omega
        return angles_dot

    def error_state(self, angles_current: np.ndarray, angles_desired: np.ndarray) -> np.ndarray:
        """Compute Euler angle error (simple subtraction)."""
        return angles_desired - angles_current

    def to_matrix(self, angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (3-2-1) to direction cosine matrix."""
        phi, theta, psi = angles

        # Rotation matrices
        Rz_psi = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        Ry_theta = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        Rx_phi = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        # 3-2-1 convention: C = Rz(ψ) · Ry(θ) · Rx(φ)
        C = Rz_psi @ Ry_theta @ Rx_phi
        return C
