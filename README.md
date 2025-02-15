# Tabular RL-Based 2D Path Planning for UGV

This repository contains the implementation of a Tabular Reinforcement Learning (RL) approach for solving the problem of 2D path planning of an Unmanned Ground Vehicle (UGV) that tracks a moving target while avoiding fixed obstacles. The problem is addressed in a grid-based environment, where the UGV navigates to the target while also tacking into account its heading and learning to avoid obstacles based on Q-learning algorithm.

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [State and Action Space](#state-and-action-space)
3. [Installation](#installation)
4. [Experimental Results](#experimental-results)

## Problem Overview
The task is to design a UGV that can move through a 2D environment, tracking a moving target while avoiding fixed obstacles. The environment is represented as a 2D grid, where the UGV learns to plan its path using tabular RL. The UGV's objective is to reach the target position while avoiding collisions with obstacles.

- Objective: Train the UGV to move towards a dynamically changing target while avoiding obstacles placed in the grid.

## State and Action Space

- State Space: The state of the system is represented by the UGV's position (X, Y) and its orientation (Theta).
  - X, Y: Position on a 2D grid.
  - Theta: Orientation (direction) of the UGV, with possible values: [0, 90, 270, 360] degrees.
  
- Action Space: The UGV can take the following actions:
  - Move Forward: Move one grid cell forward.
  - Move Backward: Move one grid cell backward.
  - Turn Clockwise: Rotate 90 degrees clockwise.
  - Turn Counterclockwise: Rotate 90 degrees counterclockwise.



## Installation

### Requirements:
To run this project, you'll need Python 3.x and a few dependencies. You can set up the environment using pip or conda.

### Steps for Setup:

1. Clone the Repository:
   ```bash
   git clone https://github.com/R-Ahmed-Khan/HetroRL.git
   cd HetroRL/
