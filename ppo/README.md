# PPO-based 2D Path Planning for UGV

This project implements an autonomous path palnning system for an Unmanned Ground Vehicle (UGV) using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The UGV learns to navigate from a start point to a moving target point in a continuous 2D environment.

<div align="center">
  
  ![ackermann](https://github.com/user-attachments/assets/d83b73fa-57d1-42be-82d2-23c6331b77dc)
  
</div>

## 🗺️ Overview

The task is framed as a reinforcement learning problem where the agent (UGV) must learn an optimal policy to reach the target. The state and action spaces are continuous, and the agent receives rewards based on its distance to the target and its alignment with the target.

### 🚗 Ackermann Kinematic Model

$$
\
x_{t+1} = x_t + v \cdot \cos(\theta_t) \cdot \Delta t,
\
$$

$$
\
y_{t+1} = y_t + v \cdot \sin(\theta_t) \cdot \Delta t,
\
$$

$$
\
\theta_{t+1} = \theta_{t} + \frac{v}{L} \cdot \tan(\delta) \cdot \Delta t,
\
$$

where x, y are UGV's position and $\theta$ is its heading, L is the distance between front and axle wheels, v is the velocity, and $\delta$ is the steering angle.

## 📌 Problem Setup

- **Agent**: UGV (Ackermann kinematic model)
- **Environment**: 2D continuous space with moving target
- **Objective**: Reach the target location from a given start point (Target can also be moving)
- **Algorithm**: Proximal Policy Optimization (PPO)

## 🧭 Observation Space

The observation space of the system is represented by the UGV's position (X, Y), its orientation ($\theta$) and the target position ($$\ X_t \$$ , $$\ Y_t \$$).
  - $X, Y$ : UGV Position on a 2D space.
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

This ensures that the policy outputs remain bounded while allowing fine control over the UGV in the continuous environment. Moreover, this methodology also allows to scale the control bounds as required.

## 🏆 Reward Function

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

### 🛠️ Implementation Logic
```python
def reward(self, observation, step_count):
    current_position = observation[:2]
    theta = observation[2]
    target_point = observation[3:]
    
    self.distance_target = np.linalg.norm(current_position - target_point)
    r_p =  1 / (self.distance_target + 0.01) 
    
    desired_theta = np.arctan2(target_point[1] - current_position[1], target_point[0] - current_position[0])
    alignment_error = self.wrap_angle(desired_theta - theta)
    r_theta = 10*np.cos(alignment_error)  
    
    reward = r_p + r_theta 
    return reward
```

### 🛑 Termination and Truncation Conditions

The episode can end in two ways:

1️⃣ Termination Condition

The episode **terminates successfully** when the agent reaches close enough to the target:

$$
\lVert \mathbf{p} - \mathbf{g} \rVert < 0.2
$$

This condition ensures that the episode ends when the target is reached.

2️⃣ Truncation Condition

The episode is **truncated** if any of the following occur:

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

- Time steps (time_steps): 1500000
- Memory length (memory): 600
- Batch size (batch_size): 200
- No. of epochs (epochs): 10
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
  - `total_timesteps` ← total number of steps

- Initialize these 3 classes:
  - Actor network $\pi_{\theta}$
  - Critic network $V_{\phi}$
  - Memory class `PPOBase`

---

### 🚀 Interaction with Environment

**For each time step:**
1. Observe current state `s`
2. Get action `a`, log probability `log_prob`, and value `v` from policy:
   - `dist ← π_θ(s)`
   - `a_raw ~ dist.sample()`
   - `a ← tanh(a_raw)`
   - `log_prob ← log_prob(a_raw) - tanh_correction`
   - `v ← V_ϕ(s)`
     
   ```python
   def choose_action(self, observation):
      state = T.tensor([observation], dtype=T.float).to(self.actor.device)
      dist = self.actor(state)
      value = self.critic(state)
      raw_action = dist.sample()
      action = T.tanh(raw_action)  # squash to [-1,1]
      log_prob = dist.log_prob(raw_action).sum(dim=-1)  # log prob in raw space
      log_prob -= T.log(1 - action.pow(2) + 1e-6).sum(dim=-1)  # Tanh correction
      return action.cpu().detach().numpy()[0], log_prob.item(), value.item()
   ```
   
3. Execute action `a` in the environment
4. Observe reward `r`, next state `s'`, and done flag

   ```python
   def get_next_observation(self, observation, action):
      steer_angle, v = self.scale_action(action)
      x, y, theta, target_x, target_y = observation
      next_x = x + v * np.cos(theta) * self.dt
      next_y = y + v * np.sin(theta) * self.dt
      next_theta = self.wrap_angle(theta + (v / self.L) * np.tan(steer_angle) * self.dt)
      return np.array([next_x, next_y, next_theta, target_x, target_y], dtype=np.float32)
   ```
   
6. Store `(s, a, log_prob, v, r, done)` in memory

---

### 🧮 Learning Phase

**When ready to learn:**

🔁 Generate Batches
- Extract `states, actions, log_probs, values, rewards, dones`
- Create shuffled mini-batches

📈 Compute GAE (Generalized Advantage Estimation)

The equations are taken from  [[1](https://arxiv.org/abs/1707.06347)]:

$$
\
\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\
$$

$$
\
\hat{A_{t}} = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T - t + 1} \delta_{T - 1}
\
$$

Our implementation:

$$
\
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\
$$

$$
\
\hat{A_{t}} = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k
\
$$

```python
for t in range(len(reward_arr) - 1):
    discount = 1
    a_t = 0
    for k in range(t, len(reward_arr) - 1):
        a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                           (1 - int(dones_arr[k])) - values[k])
        discount *= self.gamma * self.gae_lambda
    advantage[t] = a_t
```

### 🔁 PPO Update Loop

$$
\
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
\
$$

where, $r_t(\theta)$ is the probability ratio:

$$
\
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\
$$

```python
dist = self.actor(states)
critic_value = self.critic(states).squeeze()

new_probs = dist.log_prob(actions).sum(dim=-1)
prob_ratio = (new_probs - old_probs).exp()

weighted_probs = advantage[batch] * prob_ratio
weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                 1 + self.policy_clip) * advantage[batch]
```

## 💻 Installation

### ⚙️ Steps for Setup:

This repository was made on python 3.12.4 64-bit.

1. Clone the Repository:
   ```bash
   git clone https://github.com/R-Ahmed-Khan/PathSeeker.git
   ```

2. Requirements:
   ```bash
   cd PathSeeker/ppo/
   pip install -r requirements.txt
   ```

### 🧪 Training and Testing Policy

1. To train the policy, run
   ```bash
   python3 run/learn_ppo.py
   ```

   If you want to change any hyperparameter for training other than the default values, run
   ```bash
   python3 run/learn_ppo.py --<hyperparameter> <value>
   ```
    
   The hyperparameters are `time_steps`, `memory`, `batch_size`, `epochs`, `alpha`, `epsilon`, `gamma`, `gae_lamda`. You can find more details about the  
   hyperparameters in `run/learn_ppo.py` lines 13 - 21.

2. To test the policy with static target, run
   ```bash
   python3 run/test_ppo.py --no_moving_target
   ```

3. To test the policy with moving target, run
   ```bash
   python3 run/test_ppo.py --moving_target
   ```

## 📈 Experimental Results

### Learning Curve

<div align="center">
  
![reward_curve](https://github.com/user-attachments/assets/6bd6b937-d537-47c4-b72b-b8aef7d132eb)

<div>
  
### Policy Loss

<div align="center">
  
![actor_loss_curve](https://github.com/user-attachments/assets/a7f3d043-aa9f-499d-b085-3d0d9bad34f7)

<div>
  
### Critic Loss



## 🌀 Simulation

### Static Target

<div align="center">
  
![simulation](https://github.com/user-attachments/assets/5827f87c-7636-4600-886a-ea6e3186e502)

<div>


### Moving Target

<div align="center">
  
  ![simulation](https://github.com/user-attachments/assets/c412f1f3-393d-4f14-a9a8-aeaec9d2ce44)
  
</div>

## ChatGPT Prompts

<p align="center">
  <img src="https://github.com/user-attachments/assets/64ef2b97-5f40-47bb-916d-d72871db5c2d" alt="Image 1" width="45%"/>
  <img src="https://github.com/user-attachments/assets/63512144-8771-434b-8cdd-c28126972a21" alt="Image 2" width="45%"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/df7cc985-923a-44ec-af6c-7e66e90c26cc" alt="Image 1" width="45%"/>
  <img src="https://github.com/user-attachments/assets/d1c87807-601a-438c-aceb-d724347de59b" alt="Image 2" width="45%"/>
</p>


## 📚 References

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).  
**Proximal Policy Optimization Algorithms**. arXiv preprint [[1](https://arxiv.org/abs/1707.06347)].


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

