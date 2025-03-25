import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from environment import CustomEnv 
from ppo import Agent

# ---------- Argument Parser ----------
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
parser.add_argument('--moving_target', action='store_true', help='Enable moving target')
parser.add_argument('--no_moving_target', dest='moving_target', action='store_false', help='Disable moving target')
parser.set_defaults(moving_target=True)
args = parser.parse_args()

# ---------- Environment and Agent Setup ----------
env = CustomEnv()
agent = Agent(
    n_actions=env.action_space.shape[0], 
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

# ---------- Target Movement ----------
target_move = args.moving_target
print(target_move)
if target_move:
    observation[-2] = 0
    observation[-1] = 0
    observation[2] = np.arctan2(observation[-1] - observation[1], observation[-2] - observation[0]) + np.deg2rad(np.random.uniform(-30, 30)) 
    x_vel = np.random.uniform(-0.01, 0.01)
    y_vel = np.random.uniform(-0.01, 0.01)

initial_state = [observation[0], observation[1], observation[2], observation[3], observation[4]]
trajectory = []
theta_vals = []
target_positions = []

done = False

# ---------- Set Up for Animation ----------
fig, ax = plt.subplots()
frames = []

# ---------- Simulation Loop ----------
while not done:
    action, _, _ = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    trajectory.append(observation[:2])
    theta_vals.append(observation[2])
    target_positions.append((observation[-2], observation[-1]))

    if target_move:
        observation[-2] += x_vel
        observation[-1] += y_vel

    done = terminated or truncated
    print(env.step_count)

# ---------- Plot Update Function ----------
def update_plot(i):
    ax.clear()
    x_vals, y_vals = zip(*trajectory[:i+1])
    dx = np.cos(theta_vals[i]) * 0.2
    dy = np.sin(theta_vals[i]) * 0.2
    target_x, target_y = target_positions[i]

    ax.plot(x_vals, y_vals, label="Robot Path")
    ax.scatter(initial_state[0], initial_state[1], color="red", label="Start")
    ax.scatter(target_x, target_y, color="green", label="Target", s=75)
    ax.quiver(x_vals[-1], y_vals[-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid()

# ---------- Generate and Save GIF ----------
ani = animation.FuncAnimation(fig, update_plot, frames=len(trajectory), interval=10)
save_path = os.path.join(os.getcwd(), 'analysis/simulation.gif')
ani.save(save_path, writer=PillowWriter(fps=10))
print("Animation saved as 'simulation.gif'")