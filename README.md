
# Tabular RL-Based 2D Path Planning for UGV

This repository contains the implementation of a Tabular Reinforcement Learning (RL) approach for solving the problem of 2D path planning of an Unmanned Ground Vehicle (UGV) that tracks a moving target while avoiding fixed obstacles. The problem is addressed in a grid-based environment, where the UGV navigates to the target while also taking into account its heading and learning to avoid obstacles based on Q-learning algorithm.

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [State and Action Space](#state-and-action-space)
3. [Q-learning Algorithm](#q-learning-algorithm)
4. [Installation](#installation)
5. [Experimental Results](#experimental-results)

## Problem Overview
The task is to design a UGV that can move through a 2D environment, tracking a moving target while avoiding fixed obstacles. The environment is represented as a 2D grid, where the UGV learns to plan its path using tabular RL. The UGV's objective is to reach the target position while avoiding collisions with obstacles.

- Objective: Train the UGV to move towards a dynamically changing target while avoiding obstacles placed in the grid.

## State and Action Space

- State Space: The state of the system is represented by the UGV's position (X, Y) and its orientation (Theta).
  - X, Y: Position on a 2D grid.
  - Theta: Orientation (direction) of the UGV, with possible values: [0, 90, 180, 270, 360] degrees.
  
- Action Space: The UGV can take the following actions:
  - Move Forward: Move one grid cell forward.
  - Move Backward: Move one grid cell backward.
  - Turn Clockwise: Rotate 90 degrees clockwise.
  - Turn Counterclockwise: Rotate 90 degrees counterclockwise.

## Q-learning Algorithm

The Q-learning update rule is given by the following equation:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]
$$

Where:
- $$\ Q(S_t, A_t) \$$ is the action-value function for the state $$\ S_t \$$ and action $$\ A_t \$$,
- $$\ \alpha \$$ is the learning rate (controls how much new information overrides the old),
- $$\ R_{t+1} \$$ is the reward received after performing action $$\ A_t \$$ in state $$\ S_t \$$,
- $$\ \gamma \$$ is the discount factor (how much future rewards are valued over immediate rewards),
- $$\ \max_{a'} Q(S_{t+1}, a') \$$ is the maximum predicted reward for the next state $$\ S_{t+1} \$$ over all possible actions $$\ a' \$$.

This equation is used to iteratively update the Q-values (action-value function) as the agent interacts with the environment.




## Installation

### Requirements:
To run this project, you'll need Python 3.x and a few dependencies. You can set up the environment using pip or conda.

### Steps for Setup:

1. Clone the Repository:
   ```bash
   git clone https://github.com/R-Ahmed-Khan/HetroRL.git
   cd HetroRL/

## Experimental Results

The policy was tested on following parameters:

- Start Position (x,y,theta) = (0, 4, 0)
- Target Positions (x,y) = [(0, 1), (3, 1), (1,1),(2, 1), (1, 0), (3, 0)]

It can be seen from the results that reward converges, hence the UGV tracks the moving target point while avoiding the obstacle.

### Reward History and Temporal Difference Error

<div style="display: flex; justify-content: space-between;">

  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/user-attachments/assets/f7ddcc66-3442-4cf0-8735-70cc48883a01" alt="reward_history" width="300"/>
    <p>Reward History</p>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/user-attachments/assets/eaa02f00-82a7-4a6d-8dc1-35dba1c01d5d" alt="td_error" width="300"/>
    <p>Temporal Difference Error</p>
  </div>

</div>




### UGV Path

<div align="center">
  
  ![tabular_rl_2](https://github.com/user-attachments/assets/71a53c6e-e6f6-41bc-928c-5c263d13146c)

</div>

