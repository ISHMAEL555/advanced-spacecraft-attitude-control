# 🛰️ Nonlinear Spacecraft Attitude Control under Constraints

A modular Python framework for the **design, implementation, and rigorous comparative analysis** of nonlinear attitude control laws for a rigid spacecraft under actuator saturation and persistent external disturbances.

---

## 🚀 Overview

This project investigates the **rest-to-rest attitude stabilization problem** across:

* **3 attitude representations**
* **3 control architectures**
* **2 disturbance regimes**

→ **Total: 18 simulation configurations**

Each configuration is evaluated against a consistent set of performance metrics.

The emphasis is on:

* Physical correctness
* Numerical robustness
* Honest comparative analysis

This is not just about showing convergence, but understanding:

> how the system behaves, how fast it converges, and under what conditions it fails.

---

## 🧱 Spacecraft Model

### **Inertia Matrix**

```
I = diag(10, 20, 30)   [kg·m²]
```

### **Initial Conditions**

```
MRP:              σ₀ = (0, 0.8, 0)
Angular velocity: ω₀ = (0, 2, 0)   [rad/s]
```

### **Control Objective**

```
σ → 0,   ω → 0   (rest-to-rest stabilization)
```

---

## ⚙️ Dynamics

Euler’s rotational equation of motion:

```
I · ω̇ + ω × (I · ω) = u + L
```

where:

* `u` → control torque
* `L` → external disturbance torque

Kinematics are propagated using representation-specific equations.

---

## 🧭 Attitude Representations

| Representation    | Singularity         | Notes                                |
| ----------------- | ------------------- | ------------------------------------ |
| **MRP**           | |σ| = 1             | Shadow switching for large rotations |
| **Quaternion**    | None (double cover) | Requires unwinding prevention        |
| **Euler (3-2-1)** | Pitch = ±90°        | Included to demonstrate limitations  |

---

### 🔹 MRP Shadow Set Switching

At every integration step, check:

```
|σ| > 1
```

If triggered:

```
σ_s = -σ / |σ|²
```

This prevents divergence during large-angle maneuvers.

---

### 🔹 Quaternion Unwinding Prevention

Error quaternion:

```
q_err = q_d ⊗ q⁻¹
```

Sign correction:

```python
if q_err[0] < 0:
    q_err = -q_err
```

Prevents unnecessary large-angle rotations.

---

## 🎛️ Control Architectures

### 🔹 PD Control

```
u = -Kp · e_att - Kd · ω
```

* Stable without disturbance
* Non-zero steady-state error with disturbance

---

### 🔹 PD + Integral (PID)

```
u = -Kp · e_att - Kd · ω - Ki · ∫e_att dt
```

* Eliminates steady-state error
* Requires **anti-windup compensation**

Back-calculation:

```
e_int_dot = e_att + (u_sat - u_unsat) / Ki
```

---

### 🔹 Lyapunov-Based Control

Lyapunov candidate:

```
V = (1/2) · ωᵀ · I · ω + k · Φ(e_att)
```

Control law derived from:

```
V̇ ≤ 0
```

* Stability-driven design
* Saturation-aware behavior

---

## 🔒 Actuator Constraints

```
|u_i| ≤ 0.1   [N·m]
```

| Regime        | Description                   |
| ------------- | ----------------------------- |
| ✅ Unsaturated | Baseline validation           |
| ⚠️ Saturated  | Realistic constraint handling |

---

## 🌪️ Disturbance Scenarios

### **Case A — No Disturbance**

```
L = (0, 0, 0)
```

### **Case B — Constant Disturbance**

```
L = (1, 2, -1)   [N·m]
```

Represents persistent body-fixed disturbances (e.g., propellant leakage).

---

## 🧪 Simulation Matrix

| Dimension               | Options                |
| ----------------------- | ---------------------- |
| Attitude Representation | MRP, Quaternion, Euler |
| Control Law             | PD, PID, Lyapunov      |
| Disturbance Case        | Case A, Case B         |

**Total simulations: 18**

---

## 📊 Performance Metrics

| Metric                 | Definition                 |
| ---------------------- | -------------------------- |
| Convergence time       | Time to reach < 0.5°       |
| Steady-state error     | Mean error over final 10 s |
| Angular velocity decay | Time to reach |ω| < 0.01   |
| Control effort         | ∫ |u(t)| dt                |
| Saturation duration    | Time spent in saturation   |

---

## 📈 Results

*Populated after simulation runs are complete.*

| Representation | Controller | Disturbance | Conv. Time | SS Error | Effort |
| -------------- | ---------- | ----------- | ---------- | -------- | ------ |
| MRP            | PD         | None        | —          | —        | —      |
| MRP            | PID        | Constant    | —          | —        | —      |
| Quaternion     | PD         | None        | —          | —        | —      |
| Euler          | Lyapunov   | Constant    | —          | —        | —      |

📁 Full outputs available in `results/`

### Example visual outputs

After running:

```bash
python run_simulations.py
python visualize.py
```

you can inspect generated plots such as:

- `results/visualizations/Quaternion_PID_NoDisturbance.png`
- `results/visualizations/MRP_PD_Constant.png`
- `results/visualizations/comparison_all.png`

---

## ✅ Validation

### Recent robustness updates

The simulator now includes a few implementation-level safeguards that are useful to know when interpreting results:

* **PID integral state is integrated in the ODE state**, so simulation runs do not mutate controller instance memory between runs.
* **Saturation events are logged only when clamping actually occurs** (`u_sat != u_unsat`).
* **Quaternion trajectories are normalized** during simulation and result extraction to limit numerical drift.
* **Euler 3-2-1 kinematics use a small denominator guard** near pitch singularity.

These changes improve reproducibility and prevent false saturation reporting in controller comparisons.

---

### 1. Cross-Representation Consistency

All representations must produce identical **ω(t)**.

---

### 2. Euler Singularity Demonstration

Simulation near pitch = 90° highlights representation limitations.

---

## 🎲 Monte Carlo Robustness

```
I_perturbed = diag(I) · (1 + δ)
δ ~ Uniform(-0.10, +0.10)
N = 500
```

Outputs:

* Convergence rate
* Mean and 3σ bounds

📁 Results: `results/monte_carlo/`

---

## 🗂️ Repository Structure

```
attitude-control-simulator/
│
├── src/
│   ├── dynamics/
│   ├── representations/
│   ├── control/
│   ├── simulation/
│   └── analysis/
│
├── configs/
├── results/
├── notebooks/
└── tests/
```

---

## 🧠 Key Engineering Observations

* MRP switching is essential for correctness
* Quaternion unwinding significantly affects control effort
* Anti-windup is critical under saturation
* Lyapunov control maintains stability under constraints
* Euler angles expose singularity limitations

---

## 🔮 Future Work

* Reaction wheel modeling
* State estimation (EKF)
* Sensor noise and bias
* Discrete-time control
* Hardware-in-the-loop validation

---

## 🛠️ Requirements

```
Python 3.10+
numpy
scipy
matplotlib
```

---

## 📄 License

MIT License

---

<p align="center">
  <i>Focused on physically consistent modeling and rigorous control analysis.</i>
</p>
