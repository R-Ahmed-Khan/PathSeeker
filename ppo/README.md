# PPO-based 2D Path Planning for UGV

This project implements an autonomous path palnning system for an Unmanned Ground Vehicle (UGV) using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The UGV learns to navigate from a start point to a moving target point in a continuous 2D environment.

<div align="center">
  
  ![ackermann](https://github.com/user-attachments/assets/d83b73fa-57d1-42be-82d2-23c6331b77dc)
  
</div>

## 🚗 Overview

The task is framed as a reinforcement learning problem where the agent (UGV) must learn an optimal policy to reach the target efficiently and safely. The state and action spaces are continuous, and the agent receives rewards based on its distance to the target.

## 📌 Problem Setup

- **Agent**: UGV (Ackermann kinematic model)
- **Environment**: 2D continuous space with moving target
- **Objective**: Reach the target location from a given start point 
- **Algorithm**: Proximal Policy Optimization (PPO)

## 🧩 Observation Space

The observation space of the system is represented by the UGV's position (X, Y), its orientation ($\theta$) and the target position ($$\ X_t \$$ , $$\ Y_t \$$).
  - $X, Y$ : UGV Position on a 2D grid.
  - $\theta$ : UGV Orientation (heading)
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
self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
```

### 📐 Control Bounds

These normalized outputs from the policy network are scaled to the actual control limits before being applied in the environment.

```python
self.v_max = 0.6
self.v_min = 0.0

self.steer_max = np.deg2rad(25) 
self.steer_min = np.deg2rad(-25) 
```

### 🔄 Action Scaling

#### 🧮 Scaling Equations

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

#### ✅ Python Implementation

```python
def scale_action(self, action):
    """Scales action from [-1, 1] to the actual range."""
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
r_\theta = 10 \cdot \cos\theta_{\text{error}}
$$

> **Note:** The `wrap` function ensures the angle $\theta_{\text{error}}$ is within $\[-\pi, \pi]$ for proper angular difference handling.

### 🛑 Termination and Truncation Conditions

The episode can end in two ways:

1️⃣ Termination Condition

The episode **terminates successfully** when the agent reaches close enough to the target:

$$
\lVert \mathbf{p} - \mathbf{g} \rVert < 0.2
$$

Where:
- $\mathbf{p} = (x, y)$ is the agent's current position.
- $\mathbf{g} = (x_t, y_t)$ is the target position.

This condition ensures that the episode ends when the target is reached.

2️⃣ Truncation Condition

The episode is **truncated (forcefully stopped)** if any of the following occur:

- The agent exceeds the allowed episode length:

$$
\text{step count} < \text{episode length}
$$

- The agent moves **out of the environment bounds**:

$$
x < x_{\text{min}} \quad \text{or} \quad x > x_{\text{max}}
$$

$$
y < y_{\text{min}} \quad \text{or} \quad y > y_{\text{max}}
$$

These checks prevent the agent from running indefinitely or exiting the valid operating area.


### 🛠️ Implementation Logic

```python
termination = self.distance_target < 0.2

truncation = (
    step_count > self.episode_length 
    or current_position[0] < self.x_min
    or current_position[0] > self.x_max
    or current_position[1] < self.y_min
    or current_position[1] > self.y_max
)
```

## 🎛️ Hyperparameters

We have used the following learning parameters:

- Rnning Device: CPU
- Time steps: 1500000
- Memory length: 600
- Batch size: 200
- No. of epochs: 10
- Learning rate ($\alpha$): 0.0001
- Policy clip parameter ($\epsilon$): 0.2
- Discount factor ($\gamma$): 0.99
- GAE lambda ($\lambda$): 0.95

## 📜 PPO Algorithm

### 🔧 Initialization
- Set hyperparameters:
  - `gamma` ← discount factor
  - `alpha` ← learning rate
  - `gae_lambda` ← GAE parameter
  - `policy_clip` ← PPO clipping threshold
  - `batch_size` ← number of transitions per mini-batch
  - `n_epochs` ← number of learning epochs

- Initialize:
  - Actor network π_θ
  - Critic network V_ϕ
  - Optimizers for both networks
  - Memory buffer `PPOMemory`

---

### 🚀 Interaction with Environment

**For each time step or episode:**
1. Observe current state `s`
2. Get action `a`, log probability `log_prob`, and value `v` from policy:
   - `dist ← π_θ(s)`
   - `a_raw ~ dist.sample()`
   - `a ← tanh(a_raw)`  # squash to [-1, 1]
   - `log_prob ← log_prob(a_raw) - tanh_correction`
   - `v ← V_ϕ(s)`
3. Execute action `a` in the environment
4. Observe reward `r`, next state `s'`, and done flag
5. Store `(s, a, log_prob, v, r, done)` in memory

---

### 🧮 Learning Phase

**When ready to learn:**

🔁 Generate Batches
- Extract `states, actions, log_probs, values, rewards, dones`
- Create shuffled mini-batches

📈 Compute GAE (Generalized Advantage Estimation)

```python
for t in range(T - 1):
    A[t] = 0
    discount = 1
    for k in range(t, T - 1):
        delta = r_k + γ * V_{k+1} * (1 - done_k) - V_k
        A[t] += discount * delta
        discount *= γ * λ
```

### 🔁 PPO Update Loop

```python
for batch in batches:
    # Get current policy outputs
    dist_new = π_θ(states)
    V_new = V_ϕ(states)

    # Compute new log probabilities
    new_log_probs = dist_new.log_prob(actions)
    ratio = exp(new_log_probs - old_log_probs)

    # PPO clipped surrogate objective
    surr1 = ratio * advantage
    surr2 = clip(ratio, 1 - ε, 1 + ε) * advantage
    actor_loss = -mean(min(surr1, surr2))

    # Critic loss
    returns = advantage + values
    critic_loss = mean((returns - V_new)^2)

    # Total loss and optimization
    total_loss = actor_loss + 0.5 * critic_loss
    Backpropagate and update actor & critic
```

## 💻 Installation

### ⚙️ Requirement 

This repository was made on python 3.12.4 64-bit.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

PathSeeker/ppo/ 

        ├── analysis/ 

           ├── 

           ├── 

           └──  
  
        ├── models/ 

           ├── policy_ppo 
  
           └── value_ppo 
  
        ├── run/ 

           ├── learn_ppo.py 
  
           └── test_ppo.py 

        ├── src/ 

           ├── environment.py 
  
           ├── ppo.py 
  
           └── utils.py 

        ├── requirements.txt 
   
        └── README.md

