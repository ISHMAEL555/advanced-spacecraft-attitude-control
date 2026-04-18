"""Configuration system for generating simulation matrix."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for a single simulation."""

    representation_name: str  # 'MRP', 'Quaternion', 'Euler'
    control_name: str  # 'PD', 'PID', 'Lyapunov'
    disturbance_case: str  # 'NoDisturbance', 'Constant'
    initial_attitude: np.ndarray
    initial_omega: np.ndarray
    desired_attitude: np.ndarray
    disturbance_torque: np.ndarray
    control_gains: Dict
    t_final: float = 100.0
    num_points: int = 10000

    def __str__(self) -> str:
        return f"{self.representation_name}_{self.control_name}_{self.disturbance_case}"


class ConfigurationGenerator:
    """Generate simulation matrix configurations."""

    @staticmethod
    def default_initial_conditions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return default initial conditions from README.

        Returns:
            (initial_attitude, initial_omega, desired_attitude)
        """
        # MRP initial condition
        initial_mrp = np.array([0.0, 0.8, 0.0])
        initial_omega = np.array([0.0, 2.0, 0.0])

        # Desired attitude (all zeros for rest-to-rest)
        desired_mrp = np.array([0.0, 0.0, 0.0])

        return initial_mrp, initial_omega, desired_mrp

    @staticmethod
    def disturbance_cases() -> Dict[str, np.ndarray]:
        """
        Return disturbance torque cases.

        Returns:
            Dictionary of {case_name: torque_vector}
        """
        return {
            'NoDisturbance': np.array([0.0, 0.0, 0.0]),
            'Constant': np.array([1.0, 2.0, -1.0])
        }

    @staticmethod
    def control_gains() -> Dict[str, Dict]:
        """
        Return default control gains for each law.

        Returns:
            Dictionary of {law_name: gains_dict}
        """
        return {
            'PD': {
                'Kp': 50.0,
                'Kd': 10.0,
                'saturation_limit': 5.0
            },
            'PID': {
                'Kp': 50.0,
                'Kd': 10.0,
                'Ki': 0.1,
                'saturation_limit': 5.0
            },
            'Lyapunov': {
                'k1': 50.0,
                'k2': 10.0,
                'saturation_limit': 5.0
            }
        }

    @staticmethod
    def generate_matrix(
        representations: List[str] = None,
        control_laws: List[str] = None,
        disturbance_cases: List[str] = None
    ) -> List[SimulationConfig]:
        """
        Generate full simulation matrix.

        Args:
            representations: List of representation names
            control_laws: List of control law names
            disturbance_cases: List of disturbance case names

        Returns:
            List of SimulationConfig objects (18 total)
        """
        if representations is None:
            representations = ['MRP', 'Quaternion', 'Euler']

        if control_laws is None:
            control_laws = ['PD', 'PID', 'Lyapunov']

        if disturbance_cases is None:
            disturbance_cases = ['NoDisturbance', 'Constant']

        # Get defaults
        initial_attitude, initial_omega, desired_attitude = (
            ConfigurationGenerator.default_initial_conditions()
        )
        disturbance_dict = ConfigurationGenerator.disturbance_cases()
        gains_dict = ConfigurationGenerator.control_gains()

        configs = []

        for rep in representations:
            for control in control_laws:
                for dist_case in disturbance_cases:
                    config = SimulationConfig(
                        representation_name=rep,
                        control_name=control,
                        disturbance_case=dist_case,
                        initial_attitude=initial_attitude.copy(),
                        initial_omega=initial_omega.copy(),
                        desired_attitude=desired_attitude.copy(),
                        disturbance_torque=disturbance_dict[dist_case].copy(),
                        control_gains=gains_dict[control].copy(),
                        t_final=100.0,
                        num_points=10000
                    )
                    configs.append(config)

        return configs

    @staticmethod
    def print_matrix(configs: List[SimulationConfig]):
        """Print all configurations in matrix form."""
        print("\n" + "="*80)
        print("SIMULATION MATRIX (18 configurations)")
        print("="*80)

        # Group by representation
        by_rep = {}
        for config in configs:
            if config.representation_name not in by_rep:
                by_rep[config.representation_name] = {}
            if config.disturbance_case not in by_rep[config.representation_name]:
                by_rep[config.representation_name][config.disturbance_case] = []
            by_rep[config.representation_name][config.disturbance_case].append(
                config.control_name
            )

        for rep in sorted(by_rep.keys()):
            print(f"\n{rep}:")
            for dist in sorted(by_rep[rep].keys()):
                controls = by_rep[rep][dist]
                print(f"  {dist}: {', '.join(controls)}")

        print("\n" + "="*80)
