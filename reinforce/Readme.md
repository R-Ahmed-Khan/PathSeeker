
# Reinforce for Cartpole-Pendulum System

This repository contains the implementation of a Reinforce (with and without baselines).

## Table of Contents
1. [Presentation](#presentation)
2. [Problem Overview](#problem-overview)
3. [State and Action Space](#state-and-action-space)
4. [Reinforce](#reinforce)
5. [Installation](#installation)
6. [Hyperparameters](#hyperparameters)
7. [Experimental Results](#experimental-results)
8. [Directory Structure](#directory-structure)

## Presentation

You can view the presentation here: [PathSeeker](https://docs.google.com/presentation/d/1Wafn8a_oZkaHDxx68Jwqe9TH-ihOuR3aBR_MfxjrF1Q/edit#slide=id.g332ab2782df_0_1)


## Problem Overview

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

## State and Action Space

### Observations

- **Cart Position:**  
  Range: -4.8 < x < 4.8
- **Cart Velocity:**  
  Range: -∞ < v < ∞
- **Pole Angle:**  
  Range: -24° < θ < 24°
- **Pole Angular Velocity:**  
  Range: -∞ < ω < ∞

### Actions

- **Push Cart to Left:**  
  Force: -10 Newtons
- **Push Cart to Right:**  
  Force: 10 Newtons

## Reward Function

- Each timestep that the episode continues (i.e., the pole is still considered balanced and the cart has not gone out of bounds), the agent receives a reward of **+1**.
- There are no additional bonuses or penalties beyond these basic rules. Hence, the total return (sum of rewards) in one episode is equal to the number of timesteps the agent keeps the pole balanced before an end condition is triggered.

The reward function for the cartpole is defined as follows:

The agent receives a reward of **1** if the cart's position and the pole's angle remain within safe bounds; otherwise, the reward is **0** and the episode terminates. Specifically, the reward is **1** if the absolute value of the cart position is less than or equal to 2.4 and the absolute value of the pole's angle is less than or equal to 0.2095 radians. This condition ensures that the pole remains balanced and the cart does not stray too far from the center.

$$
\text{Reward} =
\begin{cases} 
1, & \text{if } |x| \leq 2.4 \text{ and } |\theta| \leq 0.2095 \text{ radians} \\
0, & \text{otherwise (episode terminates)}
\end{cases}
$$


## Reinforce 


## Installation

### System Dependencies:

- Before installing Python dependencies, you need to install the following system packages:
  ```bash
  sudo apt-get install imagemagick
  sudo apt-get install python3-dev

### Steps for Setup:

1. Clone the Repository:
   ```bash
   git clone https://github.com/R-Ahmed-Khan/PathSeeker.git

2. Requirements:
   ```bash
   cd PathSeeker/reinforce/
   pip install -r requirements.txt

### Training and Testing Policy

1. **Reinforce without baseline:**
   ```bash
   cd PathSeeker/reinforce/without_baseline
   python3 run reinforce.py
   python3 test_policy.py

2. **Reinforce with baseline:**
   ```bash
   cd PathSeeker/reinforce/with_baseline
   python3 run reinforce.py
   python3 test_policy.py

## Hyperparameters

### Learning Parameters

We have used the following learning parameters:

- Adam Optimizer
- Learning rate: 0.0005
- Discount Factor: 0.99
- Total Steps: 400000

## Experimental Results

### Cumulative Rewards

#### Cumulative Reward without Baseline
![reward_without_baseline](https://github.com/user-attachments/assets/bcef5f47-c2ef-4bed-82ce-66df41193260)

#### Cumulative Reward with Baseline

![reward_with_baseline](https://github.com/user-attachments/assets/20d59c8a-8566-4642-94af-9817d5777d5b)

### CartPole Response without Baseline

https://github.com/user-attachments/assets/98396b37-5689-4e31-9e4c-58aa0d276286

### CartPole Response with Baseline

https://github.com/user-attachments/assets/482dcc67-4ad0-46fc-be4f-fec6e2830b8d


## Directory Structure

    PathSeeker/reinforce/ 

        ├── without_baseline/ 

           ├── vanilla_reinforce.py

           ├── test_policy.py

           └── policy_net.pth 
  
        ├── with_baseline/ 

           ├── reinforce_scalar_baseline.py

           ├── test_policy.py

           └── policy_net.pth 

        ├── requirements.txt 
   
        └── README.md
