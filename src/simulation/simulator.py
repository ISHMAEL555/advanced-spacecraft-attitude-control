"""Simulation framework for spacecraft attitude control."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Callable, List

from src.dynamics import SpacecraftDynamics
from src.representations import AttitudeRepresentation, MRP
from src.control import ControlLaw


class SimulationState:
    """Container for simulation state and trajectory data."""

    def __init__(self):
        self.t = []  # Time steps
        self.attitude = []  # Attitude states
        self.omega = []  # Angular velocities
        self.u_control = []  # Control torques
        self.u_saturated = []  # Saturated control torques
        self.attitude_error = []  # Attitude errors
        self.saturation_times = []  # Times when saturation occurs


class SpacecraftSimulator:
    """Main simulation engine for spacecraft attitude control."""

    def __init__(
        self,
        dynamics: SpacecraftDynamics,
        attitude_rep: AttitudeRepresentation,
        control_law: ControlLaw,
        disturbance_torque: np.ndarray = None
    ):
        """
        Initialize simulator.

        Args:
            dynamics: Spacecraft dynamics model
            attitude_rep: Attitude representation
            control_law: Control law to use
            disturbance_torque: External disturbance [N·m] (constant, body-fixed)
        """
        self.dynamics = dynamics
        self.attitude_rep = attitude_rep
        self.control_law = control_law
        self.disturbance = disturbance_torque if disturbance_torque is not None else np.zeros(3)

    def simulate(
        self,
        initial_attitude: np.ndarray,
        initial_omega: np.ndarray,
        desired_attitude: np.ndarray,
        t_final: float = 100.0,
        num_points: int = 10000,
        method: str = 'RK45'
    ) -> SimulationState:
        """
        Run full simulation.

        Args:
            initial_attitude: Initial attitude state (assumed to be in MRP)
            initial_omega: Initial angular velocity [rad/s]
            desired_attitude: Desired attitude state (assumed to be in MRP)
            t_final: Final simulation time [s]
            num_points: Number of output points
            method: ODE solver method

        Returns:
            SimulationState with complete trajectory
        """
        # Convert attitudes to the appropriate representation
        from src.representations import Quaternion, EulerAngles, MRP as MRPRep

        if isinstance(self.attitude_rep, Quaternion):
            attitude_init = Quaternion.from_mrp(initial_attitude)
            attitude_desired = Quaternion.from_mrp(desired_attitude)
        elif isinstance(self.attitude_rep, EulerAngles):
            attitude_init = EulerAngles.from_mrp(initial_attitude)
            attitude_desired = EulerAngles.from_mrp(desired_attitude)
        else:  # MRP
            attitude_init = initial_attitude
            attitude_desired = desired_attitude

        # Initial state vector: [attitude, omega]
        y0 = np.concatenate([attitude_init, initial_omega])
        t_eval = np.linspace(0, t_final, num_points)

        # ODE system
        last_t = [0.0]

        def dynamics_rhs(t: float, y: np.ndarray) -> np.ndarray:
            dt = max(0.0, t - last_t[0])
            last_t[0] = t

            if isinstance(self.attitude_rep, Quaternion):
                attitude = Quaternion.normalize(y[:4])
                omega = y[4:7]
            else:
                attitude = y[:3]
                omega = y[3:6]

            # Compute attitude error
            error_attitude_rep = self.attitude_rep.error_state(attitude, attitude_desired)

            # For quaternion control, convert error to MRP for control laws (3 DOF)
            if isinstance(self.attitude_rep, Quaternion):
                # Convert quaternion error to MRP for control
                error_attitude = Quaternion.to_mrp(error_attitude_rep)
            else:
                error_attitude = error_attitude_rep

            _, u = self._compute_control_torques(
                error_attitude, omega, t, dt=dt, update_integral=True
            )

            # Compute angular acceleration from dynamics
            omega_dot = self.dynamics.angular_velocity_derivative(
                omega, u, self.disturbance
            )

            # Compute attitude rate
            attitude_dot = self.attitude_rep.kinematics(attitude, omega)

            return np.concatenate([attitude_dot, omega_dot])

        # Solve ODE
        sol = solve_ivp(
            dynamics_rhs,
            (0, t_final),
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=False
        )

        # Extract and store results
        state = SimulationState()
        state.t = sol.t

        if isinstance(self.attitude_rep, Quaternion):
            state.attitude = np.array([Quaternion.normalize(q) for q in sol.y[0:4, :].T])
            state.omega = sol.y[4:7, :].T
        else:
            state.attitude = sol.y[0:3, :].T
            state.omega = sol.y[3:6, :].T

        # Compute errors and control torques for each time step
        state.u_control = []
        state.u_saturated = []
        state.attitude_error = []
        state.saturation_times = []

        self.control_law.reset()  # Reset for replay

        prev_t = sol.t[0] if len(sol.t) > 0 else 0.0
        for i, t_val in enumerate(sol.t):
            dt = max(0.0, t_val - prev_t)
            prev_t = t_val
            attitude = state.attitude[i]
            omega = state.omega[i]

            error_attitude_rep = self.attitude_rep.error_state(attitude, attitude_desired)

            # Convert quaternion errors to MRP for consistent metrics
            if isinstance(self.attitude_rep, Quaternion):
                error_attitude = Quaternion.to_mrp(error_attitude_rep)
            else:
                error_attitude = error_attitude_rep

            state.attitude_error.append(error_attitude)

            u_unsat, u_sat = self._compute_control_torques(
                error_attitude, omega, t_val, dt=dt, update_integral=True
            )
            state.u_control.append(u_unsat)
            state.u_saturated.append(u_sat)

            # Check for saturation
            if np.any(np.abs(u_sat) > 0.1 - 1e-6):
                state.saturation_times.append(t_val)

        state.u_control = np.array(state.u_control)
        state.u_saturated = np.array(state.u_saturated)
        state.attitude_error = np.array(state.attitude_error)

        return state

    def _compute_control_torques(
        self,
        error_attitude: np.ndarray,
        omega: np.ndarray,
        time: float,
        dt: float,
        update_integral: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute unsaturated and saturated control torques.

        Returns:
            (u_unsat, u_sat)
        """
        # PID: explicit unsaturated torque + anti-windup update
        if hasattr(self.control_law, 'update_integral_state'):
            u_unsat = (
                -self.control_law.Kp * error_attitude
                - self.control_law.Kd * omega
                - self.control_law.Ki * self.control_law.integral_error
            )
            u_sat = self.control_law._saturate(u_unsat)

            if update_integral and dt > 0.0:
                self.control_law.update_integral_state(error_attitude, u_sat, u_unsat, dt=dt)

            return u_unsat, u_sat

        # Lyapunov controller: recover unsaturated form for logging
        if hasattr(self.control_law, 'k1') and hasattr(self.control_law, 'k2'):
            norm_sq = np.dot(error_attitude, error_attitude)
            attitude_gain = self.control_law.k1 / (1.0 + norm_sq)
            u_unsat = -attitude_gain * error_attitude - self.control_law.k2 * omega
            u_sat = self.control_law._saturate(u_unsat)
            return u_unsat, u_sat

        # PD controller has no internal saturation in this implementation
        u = self.control_law.compute_torque(error_attitude, omega, time)
        return u, u

    def monte_carlo_analysis(
        self,
        initial_attitude: np.ndarray,
        initial_omega: np.ndarray,
        desired_attitude: np.ndarray,
        inertia_perturbation_range: Tuple[float, float] = (-0.10, 0.10),
        num_runs: int = 100,
        **kwargs
    ) -> Dict:
        """
        Perform Monte Carlo analysis with inertia perturbations.

        Args:
            initial_attitude: Initial attitude
            initial_omega: Initial angular velocity
            desired_attitude: Desired attitude
            inertia_perturbation_range: (min, max) relative perturbation
            num_runs: Number of Monte Carlo runs
            **kwargs: Additional arguments for simulate()

        Returns:
            Dictionary with convergence statistics
        """
        convergence_times = []
        final_errors = []
        ss_errors = []

        for run in range(num_runs):
            # Perturb inertia
            perturbation = np.random.uniform(
                inertia_perturbation_range[0],
                inertia_perturbation_range[1],
                3
            )
            I_perturbed = np.diag(np.diag(self.dynamics.I) * (1 + perturbation))

            # Create new dynamics with perturbed inertia
            temp_dynamics = SpacecraftDynamics(I_perturbed)
            temp_simulator = SpacecraftSimulator(
                temp_dynamics,
                self.attitude_rep,
                self.control_law,
                self.disturbance
            )

            # Run simulation
            state = temp_simulator.simulate(
                initial_attitude, initial_omega, desired_attitude, **kwargs
            )

            # Compute convergence time (to < 0.5°)
            error_norms = np.linalg.norm(state.attitude_error, axis=1)
            convergence_threshold = 0.5 * np.pi / 180  # 0.5 degrees in radians

            conv_idx = np.where(error_norms < convergence_threshold)[0]
            if len(conv_idx) > 0:
                conv_time = state.t[conv_idx[0]]
            else:
                conv_time = float('inf')

            convergence_times.append(conv_time)

            # Steady-state error (last 10% of simulation)
            ss_idx = int(0.9 * len(state.t))
            ss_error = np.mean(error_norms[ss_idx:])
            ss_errors.append(ss_error)

            final_errors.append(error_norms[-1])

        return {
            'convergence_times': np.array(convergence_times),
            'convergence_mean': np.mean(convergence_times),
            'convergence_std': np.std(convergence_times),
            'convergence_3sigma_upper': np.mean(convergence_times) + 3 * np.std(convergence_times),
            'ss_errors': np.array(ss_errors),
            'ss_error_mean': np.mean(ss_errors),
            'ss_error_std': np.std(ss_errors),
            'final_errors': np.array(final_errors),
        }
