# UGV Navigation using PPO

This project implements an autonomous navigation system for an Unmanned Ground Vehicle (UGV) using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The UGV learns to navigate from a start point to a target point while avoiding obstacles in a continuous 2D environment.

## 🚗 Overview

The task is framed as a reinforcement learning problem where the agent (UGV) must learn an optimal policy to reach the target efficiently and safely. The state and action spaces are continuous, and the agent receives rewards based on its distance to the target and collision status.

## 📌 Problem Setup

- **Agent**: UGV (Ackermann steering model)
- **Environment**: 2D continuous space with static obstacles
- **Objective**: Reach the target location from a given start point while avoiding collisions
- **Algorithm**: Proximal Policy Optimization (PPO)

## 🧠 Observation Space

The observation space includes:

- Current position and orientation of the UGV
- Position of the target
- Distance and relative angle to the target
- Distance to nearest obstacle(s) if applicable

## 🎮 Action Space

- Steering angle (continuous)
- Acceleration or velocity (continuous)

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

