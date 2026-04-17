"""Unit tests for core components."""

import numpy as np
import pytest
from src.dynamics import SpacecraftDynamics
from src.representations import MRP, Quaternion, EulerAngles
from src.control import PDControl, PIDControl, LyapunovControl
from src.simulation import SpacecraftSimulator


class TestSpacecraftDynamics:
    """Test spacecraft dynamics model."""

    def test_initialization(self):
        """Test default initialization."""
        dynamics = SpacecraftDynamics()
        I = dynamics.get_inertia()
        assert I.shape == (3, 3)
        assert np.allclose(np.diag(I), [10.0, 20.0, 30.0])

    def test_angular_velocity_derivative_no_torque(self):
        """Test angular acceleration with zero torque."""
        dynamics = SpacecraftDynamics()
        omega = np.array([1.0, 0.0, 0.0])
        u = np.zeros(3)
        L = np.zeros(3)

        omega_dot = dynamics.angular_velocity_derivative(omega, u, L)

        # With I = diag(10, 20, 30), gyroscopic torque: ω × (I·ω)
        # ω × (I·ω) = [1,0,0] × [10,0,0] = [0,0,0]
        assert np.allclose(omega_dot, np.zeros(3))

    def test_angular_velocity_derivative_with_disturbance(self):
        """Test angular acceleration with external disturbance."""
        dynamics = SpacecraftDynamics()
        omega = np.zeros(3)
        u = np.zeros(3)
        L = np.array([0.1, 0.0, 0.0])

        omega_dot = dynamics.angular_velocity_derivative(omega, u, L)

        # I_inv @ L
        expected = dynamics.get_inertia_inv() @ L
        assert np.allclose(omega_dot, expected)


class TestMRP:
    """Test MRP representation."""

    def test_shadow_check_no_switch(self):
        """Test shadow check when |σ| < 1."""
        mrp = MRP()
        sigma = np.array([0.0, 0.5, 0.0])
        sigma_corrected = mrp.shadow_check(sigma)
        assert np.allclose(sigma_corrected, sigma)

    def test_shadow_check_switch(self):
        """Test shadow check when |σ| > 1."""
        mrp = MRP()
        sigma = np.array([0.0, 1.5, 0.0])
        sigma_corrected = mrp.shadow_check(sigma)

        # σ_s = -σ / |σ|² = -[0, 1.5, 0] / 2.25 = [0, -2/3, 0]
        expected = -sigma / np.dot(sigma, sigma)
        assert np.allclose(sigma_corrected, expected)

    def test_error_state_identity(self):
        """Test error computation with identical attitudes."""
        mrp = MRP()
        sigma = np.array([0.0, 0.5, 0.0])
        error = mrp.error_state(sigma, sigma)
        assert np.allclose(error, np.zeros(3))

    def test_to_matrix_identity(self):
        """Test DCM conversion for zero MRP."""
        mrp = MRP()
        sigma = np.zeros(3)
        C = mrp.to_matrix(sigma)
        assert np.allclose(C, np.eye(3))


class TestQuaternion:
    """Test Quaternion representation."""

    def test_normalize(self):
        """Test quaternion normalization."""
        q = np.array([1.0, 0.0, 0.0, 1.0])
        q_norm = Quaternion.normalize(q)
        assert np.isclose(np.linalg.norm(q_norm), 1.0)

    def test_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = Quaternion.conjugate(q)
        expected = np.array([-1.0, -2.0, -3.0, 4.0])
        assert np.allclose(q_conj, expected)

    def test_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q1 = np.array([0.0, 0.0, 0.0, 1.0])  # Identity
        q2 = np.array([1.0, 2.0, 3.0, 4.0])
        result = Quaternion.multiply(q1, q2)
        assert np.allclose(result, q2)

    def test_error_state_identity(self):
        """Test error computation with identical quaternions."""
        quat = Quaternion()
        q = np.array([0.0, 0.0, 0.0, 1.0])
        error = quat.error_state(q, q)
        # Should be identity (0, 0, 0, 1)
        assert np.allclose(error, q)


class TestPDControl:
    """Test PD control law."""

    def test_initialization(self):
        """Test PD controller initialization."""
        pd = PDControl(Kp=1.0, Kd=0.5)
        assert pd.Kp == 1.0
        assert pd.Kd == 0.5

    def test_zero_error_zero_velocity(self):
        """Test control torque with zero error and velocity."""
        pd = PDControl(Kp=1.0, Kd=0.5)
        error = np.zeros(3)
        omega = np.zeros(3)
        u = pd.compute_torque(error, omega)
        assert np.allclose(u, np.zeros(3))

    def test_control_response(self):
        """Test control torque response to error."""
        pd = PDControl(Kp=1.0, Kd=0.5)
        error = np.array([1.0, 0.0, 0.0])
        omega = np.array([0.5, 0.0, 0.0])
        u = pd.compute_torque(error, omega)

        expected = -1.0 * error - 0.5 * omega
        assert np.allclose(u, expected)


class TestPIDControl:
    """Test PID control law."""

    def test_initialization(self):
        """Test PID controller initialization."""
        pid = PIDControl(Kp=1.0, Kd=0.5, Ki=0.1)
        assert pid.Kp == 1.0
        assert pid.Ki == 0.1
        assert np.allclose(pid.integral_error, np.zeros(3))

    def test_anti_windup(self):
        """Test anti-windup integration."""
        pid = PIDControl(Kp=1.0, Kd=0.5, Ki=0.1, saturation_limit=0.1)
        error = np.array([1.0, 0.0, 0.0])
        omega = np.zeros(3)

        # Compute unsaturated control
        u_unsat = -1.0 * error - 0.5 * omega - 0.1 * pid.integral_error
        u_sat = pid.compute_torque(error, omega)

        # Update integral with anti-windup
        pid.update_integral_state(error, u_sat, u_unsat, dt=0.1)

        # Integral should be adjusted due to saturation
        assert np.any(pid.integral_error != 0)


class TestLyapunovControl:
    """Test Lyapunov-based control."""

    def test_initialization(self):
        """Test Lyapunov controller initialization."""
        I = np.diag([10.0, 20.0, 30.0])
        lyap = LyapunovControl(I, k1=0.5, k2=0.1)
        assert lyap.k1 == 0.5
        assert lyap.k2 == 0.1

    def test_gain_scheduling(self):
        """Test attitude-dependent gain scheduling."""
        I = np.diag([10.0, 20.0, 30.0])
        lyap = LyapunovControl(I, k1=1.0, k2=0.0)

        # Small error: high gain
        error_small = np.array([0.1, 0.0, 0.0])
        u_small = lyap.compute_torque(error_small, np.zeros(3))

        # Large error: lower gain (due to gain scheduling)
        error_large = np.array([0.5, 0.0, 0.0])
        u_large = lyap.compute_torque(error_large, np.zeros(3))

        # Magnitude should increase sublinearly
        assert np.linalg.norm(u_small) < np.linalg.norm(u_large)


class TestSimulator:
    """Test simulation framework."""

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        dynamics = SpacecraftDynamics()
        mrp = MRP()
        pd = PDControl()

        simulator = SpacecraftSimulator(dynamics, mrp, pd)
        assert simulator.dynamics == dynamics
        assert simulator.attitude_rep == mrp

    def test_single_simulation_runs(self):
        """Test that a full simulation completes."""
        dynamics = SpacecraftDynamics()
        mrp = MRP()
        pd = PDControl()

        simulator = SpacecraftSimulator(dynamics, mrp, pd)

        initial_attitude = np.array([0.0, 0.8, 0.0])
        initial_omega = np.array([0.0, 2.0, 0.0])
        desired_attitude = np.zeros(3)

        state = simulator.simulate(
            initial_attitude,
            initial_omega,
            desired_attitude,
            t_final=10.0,
            num_points=100
        )

        assert len(state.t) == 100
        assert state.attitude.shape == (100, 3)
        assert state.omega.shape == (100, 3)
        assert len(state.attitude_error) == 100

    def test_control_histories_are_populated(self):
        """Simulator should provide unsaturated and saturated control histories."""
        dynamics = SpacecraftDynamics()
        mrp = MRP()
        pid = PIDControl(Kp=0.5, Kd=0.1, Ki=0.05, saturation_limit=0.1)
        simulator = SpacecraftSimulator(dynamics, mrp, pid)

        state = simulator.simulate(
            np.array([0.0, 0.8, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.zeros(3),
            t_final=5.0,
            num_points=80
        )

        assert state.u_control.shape == (80, 3)
        assert state.u_saturated.shape == (80, 3)
        assert np.all(np.abs(state.u_saturated) <= 0.1 + 1e-8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
