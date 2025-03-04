
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

- Learning rate: 0.0005
- Discount Factor: 0.99
- Episodes: 400000

## Experimental Results

### CartPole Response without Baseline


### CartPole Response with Baseline



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
