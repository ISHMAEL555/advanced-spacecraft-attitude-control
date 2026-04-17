# 🛰️ Nonlinear Spacecraft Attitude Control under Constraints

A modular Python framework for the **design, implementation, and comparative analysis** of nonlinear attitude control laws for a rigid spacecraft under actuator saturation and persistent disturbances.

---

## 🚀 Overview

This project investigates the **rest-to-rest attitude stabilization problem** using multiple attitude representations and nonlinear control strategies.

The focus is on evaluating controller performance under realistic conditions:

* Nonlinear rigid body dynamics
* Actuator saturation constraints
* Constant external disturbances
* Representation-dependent kinematics

The framework emphasizes **physical consistency, numerical robustness, and structured comparison**.

---

## 🧱 Spacecraft Model

**Inertia Matrix**

```
I = diag(10, 20, 30)   [kg·m²]
```

**Initial Conditions**

* MRP: `sigma0 = (0, 0.8, 0)`
* Angular velocity: `omega0 = (0, 2, 0) rad/s`

**Control Objective**

```
sigma → 0
omega → 0
```

---

## ⚙️ Dynamics

The rotational dynamics are governed by:

```
I * omega_dot + omega × (I * omega) = u + L
```

Where:

* `omega` → body angular velocity
* `u` → control torque
* `L` → external disturbance torque

---

## 🧭 Attitude Representations

The following representations are implemented and compared:

| Representation           | Characteristics                                   |
| ------------------------ | ------------------------------------------------- |
| **MRPs**                 | Compact, efficient, requires shadow set switching |
| **Quaternions**          | Singular-free, numerically robust                 |
| **Euler Angles (3-2-1)** | Intuitive, but prone to singularities             |

---

## 🎛️ Control Architectures

### 🔹 Proportional-Derivative (PD)

```
u = -Kp * e_att - Kd * omega
```

* Stable in the absence of disturbances
* Exhibits steady-state error under constant disturbance

---

### 🔹 PD with Integral Action

```
u = -Kp * e_att - Kd * omega - Ki * integral(e_att)
```

* Eliminates steady-state error under constant disturbances
* Introduces slower convergence and potential overshoot
* Requires anti-windup under actuator limits

---

### 🔹 Lyapunov-Based Control

```
V = 0.5 * omega^T * I * omega + k * Phi(e_att)
```

* Ensures stability through Lyapunov design
* Accounts for actuator saturation
* Provides bounded, robust response

---

## 🔒 Actuator Constraints

```
|u_i| ≤ 0.1   [N·m],   i = 1,2,3
```

Two regimes are considered:

* ✅ **Unsaturated operation** (baseline validation)
* ⚠️ **Saturated operation** (realistic constraint handling)

---

## 🌪️ Disturbance Scenarios

### Case A — No Disturbance

```
L = (0, 0, 0)
```

### Case B — Constant Disturbance

```
L = (1, 2, -1)   [N·m]
```

Represents persistent body-fixed disturbances (e.g., propellant leakage).

---

## 🧪 Simulation Coverage

All combinations are evaluated:

* 3 Attitude Representations
* 3 Control Laws
* 2 Disturbance Cases

**Total Simulations: 18**

---

## 📊 Performance Metrics

Each configuration is evaluated using:

* ⏱️ Convergence time
* 🎯 Steady-state error
* 📉 Angular velocity decay
* ⚡ Control effort (∫|u| dt)
* 🚧 Saturation behavior

---

## 📈 Expected Results

| Controller | Disturbance | Expected Behavior                 |
| ---------- | ----------- | --------------------------------- |
| PD         | No          | Stable, fast convergence          |
| PD         | Yes         | Non-zero steady-state error       |
| PID        | Yes         | Near-zero steady-state error      |
| Lyapunov   | Yes         | Stable, saturation-aware response |

---

## 🗂️ Repository Structure

```
attitude-control-simulator/
│
├── src/
│   ├── dynamics/
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

## 🧠 Key Insights

* MRPs are efficient but require careful switching
* Quaternions provide the most robust numerical performance
* Euler angles highlight singularity limitations
* Integral control is essential for disturbance rejection
* Actuator saturation significantly affects closed-loop behavior

---

## 🔮 Future Work

* Reaction wheel actuator modeling
* State estimation (e.g., Extended Kalman Filter)
* Sensor noise and bias modeling
* Discrete-time controller implementation
* Monte Carlo robustness analysis

---

## 🛠️ Requirements

* Python 3.10+
* NumPy
* SciPy
* Matplotlib

---

## 📄 License

MIT License

---

<p align="center">
  <i>Focused on physically consistent modeling and realistic control behavior.</i>
</p>
