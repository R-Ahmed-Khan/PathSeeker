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

- **Input to Neural Network**:  
  - Steering angle ∈ [-1, 1]  
  - Velocity ∈ [0, 1]

\subsection*{Action Space}

The UGV is modeled using an Ackermann steering system and is controlled via a continuous 2D action space. The action vector consists of:

\begin{itemize}
    \item Steering angle $a_{\text{steer}} \in [-1, 1]$
    \item Velocity $a_{\text{vel}} \in [0, 1]$
\end{itemize}

The action space is defined as:
\[
\texttt{spaces.Box}\left(\texttt{low} = \begin{bmatrix} -1 \\ 0 \end{bmatrix}, 
\texttt{high} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \texttt{dtype=np.float32} \right)
\]

These normalized outputs from the policy network are scaled to the actual control limits before being applied in the environment.

\subsubsection*{Control Bounds}

\begin{align*}
    \theta_{\text{min}} &= -25^\circ = \texttt{np.deg2rad(-25)} \\
    \theta_{\text{max}} &= 25^\circ = \texttt{np.deg2rad(25)} \\
    v_{\text{min}} &= 0.0 \\
    v_{\text{max}} &= 0.6
\end{align*}

\subsubsection*{Action Scaling}

The actual steering angle $\theta$ and velocity $v$ are computed from the normalized action $\mathbf{a} = [a_{\text{steer}}, a_{\text{vel}}]$ as follows:

\begin{align*}
    \theta &= \theta_{\text{min}} + \frac{a_{\text{steer}} + 1}{2} \cdot (\theta_{\text{max}} - \theta_{\text{min}}) \\
    v &= v_{\text{min}} + \frac{a_{\text{vel}} + 1}{2} \cdot (v_{\text{max}} - v_{\text{min}})
\end{align*}

\subsubsection*{Implementation Snippet}

\begin{verbatim}
def scale_action(self, action):
    steer_angle = self.steer_min + (action[0] + 1) * 0.5 * (self.steer_max - self.steer_min)
    v = self.v_min + (action[1] + 1) * 0.5 * (self.v_max - self.v_min)
    return steer_angle, v
\end{verbatim}

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

