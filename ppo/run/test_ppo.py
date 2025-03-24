import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from environment import CustomEnv 
from ppo import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--time_steps', type=int, default=100, help='total number of time steps')
parser.add_argument('--memory', type=int, default=600, help='memory length')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=7, help='number of epochs')
parser.add_argument('--alpha', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.2, help='policy clip parameter')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--gae_lamda', type=float, default=0.95, help='GAE Lambda')
args = parser.parse_args()

env = CustomEnv()
agent = Agent(n_actions=env.action_space.shape[0], 
            input_dims=env.observation_space.shape,
            device=torch.device(args.dvc),
            gamma=args.gamma, 
            alpha=args.alpha, 
            gae_lambda=args.gae_lamda,
            policy_clip=args.epsilon,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            )
agent.load_models()
env.step_count = 0 
observation, _ = env.reset()
initial_state = [observation[0], observation[1], observation[2], observation[3], observation[4]]

trajectory = []
theta_vals = []  
done = False
step_count = 0
animate = True
while not done:
    action, _, _ = agent.choose_action(observation)  
    observation, reward, terminated, truncated, info = env.step(action)  
    trajectory.append(observation[:2]) 
    theta_vals.append(observation[2])  
    plt.clf() 
    x_vals, y_vals = zip(*trajectory)
    
    if animate is True:
        plt.plot(x_vals, y_vals, label="Robot Path")
        plt.scatter(initial_state[0], initial_state[1], color="red", label="Start")  
        plt.scatter(initial_state[3], initial_state[4], color="green", label="Target") 
        arrow_interval = max(1, len(x_vals) // 10)  
        dx = np.cos(theta_vals) * 0.2  
        dy = np.sin(theta_vals) * 0.2
        plt.quiver(x_vals, y_vals, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')
        plt.xlim(env.x_min, env.x_max)
        plt.ylim(env.y_min, env.y_max)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.pause(0.01)  
    print(env.step_count)
    done = terminated or truncated
    
if animate is False:
    plt.plot(x_vals, y_vals, label="Robot Path")
    plt.scatter(initial_state[0], initial_state[1], color="red", label="Start") 
    plt.scatter(initial_state[3], initial_state[4], color="green", label="Target")  
    dx = np.cos(theta_vals) * 0.2 
    dy = np.sin(theta_vals) * 0.2
    plt.quiver(x_vals, y_vals, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.xlim(env.x_min, env.x_max)
    plt.ylim(env.y_min, env.y_max)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
print('terminated ', terminated)
print('truncated ', truncated)
plt.show()
