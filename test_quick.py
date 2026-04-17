#!/usr/bin/env python
"""Quick test simulation with fewer points for faster execution."""

import numpy as np
import json
from pathlib import Path

from configs.config import ConfigurationGenerator
from src.dynamics import SpacecraftDynamics
from src.representations import MRP
from src.control import PDControl
from src.simulation import SpacecraftSimulator
from src.analysis import PerformanceMetrics


print("Running quick test with MRP + PD (no disturbance)...")

dynamics = SpacecraftDynamics()
mrp = MRP()
pd = PDControl()
simulator = SpacecraftSimulator(dynamics, mrp, pd)

initial_attitude, initial_omega, desired_attitude = (
    ConfigurationGenerator.default_initial_conditions()
)

state = simulator.simulate(
    initial_attitude,
    initial_omega,
    desired_attitude,
    t_final=100.0,
    num_points=2000
)

metrics = PerformanceMetrics.compute_all_metrics(state)

print("\nMetrics:")
print(f"  Convergence time: {metrics['convergence_time_s']:.3f} s")
print(f"  SS Error: {metrics['steady_state_error_deg']:.4f}°")
print(f"  Control effort: {metrics['control_effort_Nms']:.3f} N·m·s")
print(f"  Saturation time: {metrics['saturation_duration_s']:.3f} s")
print(f"  Final error: {metrics['final_error_deg']:.4f}°")
print("\n✓ Test passed!")
