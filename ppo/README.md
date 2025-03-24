# PPO-based 2D Path Planning for UGV

This project implements an autonomous path palnning system for an Unmanned Ground Vehicle (UGV) using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The UGV learns to navigate from a start point to a moving target point in a continuous 2D environment.

## 🚗 Overview

The task is framed as a reinforcement learning problem where the agent (UGV) must learn an optimal policy to reach the target efficiently and safely. The state and action spaces are continuous, and the agent receives rewards based on its distance to the target.

## 📌 Problem Setup

- **Agent**: UGV (Ackermann kinematic model)
- **Environment**: 2D continuous space with moving target
- **Objective**: Reach the target location from a given start point 
- **Algorithm**: Proximal Policy Optimization (PPO)

## 🧠 Observation Space

The observation space of the system is represented by the UGV's position (X, Y), its orientation ($\theta$) and the target position ($$\ X_t \$$ , $$\ Y_t \$$).
  - $X, Y$ : UGV Position on a 2D grid.
  - $\theta$ : UGV Orientation (direction) of the UGV, with possible values: [0, 90, 180, 270] degrees.
  - $$\ X_t \$$ , $$\ Y_t \$$ : Target Position

## 🎮 Action Space

The action space consists of two continuous control inputs:

- **Steering angle** (in radians)
- **Velocity** (in meters/second)

The **neural network (policy)** outputs actions in the normalized range:
- Steering angle ∈ **[-1, 1]**
- Velocity ∈ **[0, 1]**

```python
self.action_space = spaces.Box(
    low=np.array([-1, 0]),
    high=np.array([1, 1]),
    dtype=np.float32
)
```

## 📐 Control Bounds

These normalized outputs from the policy network are scaled to the actual control limits before being applied in the environment.

```python
self.v_max = 0.6
self.v_min = 0.0

self.steer_max = np.deg2rad(25)   # ≈ 0.4363 rad
self.steer_min = np.deg2rad(-25)  # ≈ -0.4363 rad
```

## 🔄 Action Scaling Function

To convert normalized actions to their real-world equivalents, the following function is used:

```python
def scale_action(self, action):
    """Scales action from [-1,1] and [0,1] to real-world bounds."""
    steer_angle = self.steer_min + (action[0] + 1) * 0.5 * (self.steer_max - self.steer_min)
    v = self.v_min + (action[1] + 1) * 0.5 * (self.v_max - self.v_min)
    return steer_angle, v
```

## 🧮 Scaling Equations

Let the normalized action from the policy be:

- $\ a_0$ in [-1, 1] — **Steering angle** ( $\delta \$)
- $\ a_1$ in [0, 1] — **Velocity** $\( v $\)

Then the scaled real-world values are computed as:

### ✅ Steering Angle ( $\delta \$)
$
\delta = \delta_{\text{min}} + \frac{(a_0 + 1)}{2} \cdot (\delta_{\text{max}} - \delta_{\text{min}})
$

### ✅ Velocity
$\[
v = v_{\text{min}} + \frac{(a_1 + 1)}{2} \cdot (v_{\text{max}} - v_{\text{min}})
$\]

This ensures that the policy outputs remain bounded while allowing fine control over the UGV in the continuous environment.

## 🏁 Reward Function

The reward function is designed to encourage the UGV to:

- Move closer to the target
- Avoid collisions
- Minimize time taken to reach the target
- Penalize erratic or unsafe maneuvers

## ⚙️ Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Gym (custom environment)
- [Optional] PyBullet (if simulation is physics-based)

Install dependencies:

```bash
pip install -r requirements.txt

