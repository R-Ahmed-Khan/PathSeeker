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

  ```python
# State space bounds
self.x_min, self.x_max = -2, 2
self.y_min, self.y_max = -2, 2
self.theta_min, self.theta_max = np.deg2rad(-180), np.deg2rad(180)

# Observation space: [x, y, theta, x_target, y_target]
self.observation_space = spaces.Box(
    low=np.array([self.x_min, self.y_min, self.theta_min, self.x_min, self.y_min]),
    high=np.array([self.x_max, self.y_max, self.theta_max, self.x_max, self.y_max]),
    dtype=np.float32
)
```

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

## 🔄 Action Scaling

### 🧮 Scaling Equations

Let the normalized action from the policy be:

- $\ a_0$ in [-1, 1] — **Steering angle** ( $\delta \$)
- $\ a_1$ in [0, 1] — **Velocity** $\( v $\)

Then the scaled real-world values are computed as:

- Steering Angle ( $\delta \$)
  
$$
\
\delta = \delta_{\text{min}} + \frac{(a_0 + 1)}{2} \cdot (\delta_{\text{max}} - \delta_{\text{min}})
\
$$

- Velocity
  
$$
\
v = v_{\text{min}} + \frac{(a_1 + 1)}{2} \cdot (v_{\text{max}} - v_{\text{min}})
\
$$

To convert normalized actions to their real-world equivalents, the following function is used:

### ✅ Python Implementation

```python
def scale_action(self, action):
    """Scales action from [-1,1] and [0,1] to real-world bounds."""
    steer_angle = self.steer_min + (action[0] + 1) * 0.5 * (self.steer_max - self.steer_min)
    v = self.v_min + (action[1] + 1) * 0.5 * (self.v_max - self.v_min)
    return steer_angle, v

def get_next_observation(self, observation, action):    
    steer_angle, v = self.scale_action(action)
    ...
```

This ensures that the policy outputs remain bounded while allowing fine control over the UGV in the continuous environment.

## 🎯 Reward Function

The total reward $\( r $\) at each time step is defined as the sum of:

- **Proximity reward** $\( r_p $\) — encourages the agent to get closer to the target.
- **Heading alignment reward** $\( r_\theta $\) — encourages the agent to align its heading toward the target.

$$
r = r_p + r_\theta
$$

Where:

🔹 Proximity Reward

Let $$\ \mathbf{p} = (x, y) \$$ be the agent's current position and $$\ \mathbf{g} = (x_t, y_t) \$$ the target position:

$$
r_p = \frac{1}{\lVert \mathbf{p} - \mathbf{g} \rVert + 0.01}
$$

🔹 Heading Alignment Reward

Let $\theta$ be the agent's current heading, and $\theta_{\text{t}}$ be the direction to the target:

$$
\theta_{\text{t}} = \arctan2(y_t - y, \ x_t - x)
$$

The alignment error is:

$$
\theta_{\text{error}} = \text{wrap}(\theta_{\text{t}} - \theta)
$$

Then:

$$
r_\theta = 10 \cdot \cos(\delta)
$$

> ✅ **Note:** The `wrap` function ensures the angle $\( \delta $\) is within $\[-\pi, \pi]$ for proper angular difference handling.

---

### 🧠 Summary

The reward encourages the agent to:

- Minimize distance to the target.
- Align its heading with the direction toward the target.


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

