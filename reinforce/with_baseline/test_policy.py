import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

class PolicyNet(nn.Module):
    def __init__(self, nvec_s: int, nvec_u: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(nvec_s, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nvec_u)

    def forward(self, x):
        """
        Forward pass for inference: Selects the most probable action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Logits for action probabilities
        action = torch.argmax(x, dim=-1)  # Choose action with highest probability
        return action  # Return only the action (no log_prob or entropy)

env = gym.make("CartPole-v1",
               render_mode='human'
               )

# Load the trained policy network
policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n)  # 9 possible actions
policy_net.load_state_dict(torch.load("policy_net.pth"))
policy_net.eval()  # Set the model to evaluation mode
device = torch.device("cpu")

n_episodes = 100
for _ in range(n_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = policy_net(torch.from_numpy(obs).float().to(device)).item()
        obs, reward, terminated,  truncated, info = env.step(action)
        env.render()
