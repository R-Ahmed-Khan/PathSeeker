# UGV Navigation using Reinforcement Learning

This repository contains the implementation of a UGV (Unmanned Ground Vehicle) navigation task, where the goal is to move from a given start position to a target position while avoiding obstacles in the environment.

## 🧠 Problem Statement

The UGV must learn to navigate from a start location to a target location without colliding with obstacles. The task is approached using reinforcement learning techniques and is implemented in a custom environment.

## 🛠️ Approaches Used

Three different reinforcement learning algorithms are used to solve this task:

1. **Tabular Q-Learning**  
   A basic value-based method using a discrete state-action representation.

2. **REINFORCE (Policy Gradient)**  
   A Monte Carlo policy gradient method that directly optimizes the policy.

3. **TRPO (Trust Region Policy Optimization)**  
   A more advanced policy optimization method that ensures stable and efficient updates.

## 📁 Folder Structure

Each approach has its own directory in the repository:

- `tabular_rl/`: Contains implementation and results for the tabular Q-learning approach.
- `reinforce/`: Contains implementation and results for the REINFORCE algorithm.
- `trpo/`: Contains implementation and results for the TRPO approach.

Each folder includes the necessary scripts, logs, and data specific to the respective algorithm.

## 🚀 Getting Started

Clone the repo and navigate to the folder corresponding to the approach you wish to run. Each folder has its own setup and instructions (if required).

```
git clone https://github.com/R-Ahmed-Khan/PathSeeker.git
```

For tabular RL, type in the terminal
```
cd PathSeeker/tabular_rl
```

For Reinforce, type in the terminal
```
cd PathSeeker/reinforce
```

For TRPO, type in the terminal
```
cd PathSeeker/trpo
```
