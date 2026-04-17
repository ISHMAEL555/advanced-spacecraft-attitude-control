"""Simulation framework for spacecraft attitude control."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple

from src.dynamics import SpacecraftDynamics
from src.representations import AttitudeRepresentation
from src.control import ControlLaw, PIDControl, LyapunovControl


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

        is_pid = isinstance(self.control_law, PIDControl)

        # Initial state vector: [attitude, omega, integral_error?]
        if is_pid:
            self.control_law.reset()
            y0 = np.concatenate([attitude_init, initial_omega, self.control_law.integral_error])
        else:
            y0 = np.concatenate([attitude_init, initial_omega])
        t_eval = np.linspace(0, t_final, num_points)

        # ODE system
        def dynamics_rhs(t: float, y: np.ndarray) -> np.ndarray:
            if isinstance(self.attitude_rep, Quaternion):
                attitude = Quaternion.normalize(y[:4])
                omega = y[4:7]
                idx_after_omega = 7
            else:
                attitude = y[:3]
                omega = y[3:6]
                idx_after_omega = 6

            integral_error = y[idx_after_omega:idx_after_omega + 3] if is_pid else None

            # Compute attitude error
            error_attitude_rep = self.attitude_rep.error_state(attitude, attitude_desired)

            # For quaternion control, convert error to MRP for control laws (3 DOF)
            if isinstance(self.attitude_rep, Quaternion):
                # Convert quaternion error to MRP for control
                error_attitude = Quaternion.to_mrp(error_attitude_rep)
            else:
                error_attitude = error_attitude_rep

            if is_pid:
                u_unsat = (
                    -self.control_law.Kp * error_attitude
                    - self.control_law.Kd * omega
                    - self.control_law.Ki * integral_error
                )
                u = np.clip(u_unsat, -self.control_law.sat_limit, self.control_law.sat_limit)
                integral_dot = error_attitude + (u - u_unsat) / (self.control_law.Ki + 1e-10)
            else:
                u = self.control_law.compute_torque(error_attitude, omega, t)
                integral_dot = None

            # Compute angular acceleration from dynamics
            omega_dot = self.dynamics.angular_velocity_derivative(
                omega, u, self.disturbance
            )

            # Compute attitude rate
            attitude_dot = self.attitude_rep.kinematics(attitude, omega)

            if is_pid:
                return np.concatenate([attitude_dot, omega_dot, integral_dot])
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
            idx_after_omega = 7
        else:
            state.attitude = sol.y[0:3, :].T
            state.omega = sol.y[3:6, :].T
            idx_after_omega = 6

        integral_traj = sol.y[idx_after_omega:idx_after_omega + 3, :].T if is_pid else None

        # Compute errors and control torques for each time step
        state.u_control = []
        state.u_saturated = []
        state.attitude_error = []
        state.saturation_times = []

        for i, t_val in enumerate(sol.t):
            attitude = state.attitude[i]
            omega = state.omega[i]

            error_attitude_rep = self.attitude_rep.error_state(attitude, attitude_desired)

            # Convert quaternion errors to MRP for consistent metrics
            if isinstance(self.attitude_rep, Quaternion):
                error_attitude = Quaternion.to_mrp(error_attitude_rep)
            else:
                error_attitude = error_attitude_rep

            state.attitude_error.append(error_attitude)

            if is_pid:
                integral_error = integral_traj[i]
                u_unsat = (
                    -self.control_law.Kp * error_attitude
                    - self.control_law.Kd * omega
                    - self.control_law.Ki * integral_error
                )
                u_sat = np.clip(u_unsat, -self.control_law.sat_limit, self.control_law.sat_limit)
            elif isinstance(self.control_law, LyapunovControl):
                norm_sq = np.dot(error_attitude, error_attitude)
                attitude_gain = self.control_law.k1 / (1.0 + norm_sq)
                u_unsat = -attitude_gain * error_attitude - self.control_law.k2 * omega
                u_sat = np.clip(u_unsat, -self.control_law.sat_limit, self.control_law.sat_limit)
            else:
                u_unsat = self.control_law.compute_torque(error_attitude, omega, t_val)
                u_sat = u_unsat

            state.u_control.append(u_unsat)
            state.u_saturated.append(u_sat)

            # Check for saturation
            saturation_limit = getattr(self.control_law, 'sat_limit', 0.1)
            if np.any(np.abs(u_sat) >= saturation_limit - 1e-6):
                state.saturation_times.append(t_val)

        state.u_control = np.array(state.u_control)
        state.u_saturated = np.array(state.u_saturated)
        state.attitude_error = np.array(state.attitude_error)

        return state

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
