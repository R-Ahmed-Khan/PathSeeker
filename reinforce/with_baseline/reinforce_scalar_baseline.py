import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class PolicyNet(nn.Module):
    def __init__(self, nvec_s: int, nvec_u: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(nvec_s, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nvec_u)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        entropy = dist.entropy()
        log_prob = dist.log_prob(action)
        return action, log_prob, entropy

# -----------------------------
# REINFORCE agent with a simple
# global scalar baseline
# -----------------------------
class REINFORCE_ScalarBaseline:
    def __init__(self, env: gym.Env, lr=1e-3, gamma=0.99, n_steps=400000):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = AdamW(self.policy_net.parameters(), lr=lr)

        # Global baseline (running average of returns) - just a float
        self.global_baseline = 0.0
        # How fast to update the baseline
        self.baseline_alpha = 0.01

        self.total_steps = 0
        self.episodes = 0
        self.total_rewards = 0

        self.writer = SummaryWriter("runs/REINFORCE_ScalarBaseline")

    def rollout(self):
        """
        Run one episode, storing log_probs, rewards, entropies.
        """
        state, info = self.env.reset()
        done = False

        self.log_probs = []
        self.rewards = []
        self.entropies = []

        while not done:
            state_t = torch.from_numpy(state).float().to(self.device)
            action, log_prob, entropy = self.policy_net(state_t)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.entropies.append(entropy)

            state = next_state
            self.total_rewards += reward
            self.total_steps += 1
            self.pbar.update(1)

        self.writer.add_scalar("Cumulative Reward", self.total_rewards, self.total_steps)
        self.episodes += 1

    def compute_returns(self):
        """
        Monte Carlo returns: G_t = r_t + gamma*r_{t+1} + ...
        """
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def learn(self):
        """
        1. Compute returns
        2. Update global baseline
        3. Compute advantages = returns - baseline
        4. Update policy
        """
        returns = self.compute_returns()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Update global baseline using each state's return (incremental update)
        for G_t in returns:
            self.global_baseline += self.baseline_alpha * (G_t.item() - self.global_baseline)

        # Advantages
        advantages = returns - self.global_baseline

        # Policy loss
        policy_loss = -torch.mean(advantages.detach() * log_probs)

        # Entropy bonus (optional)
        entropy_loss = -torch.mean(entropies)
        policy_loss = policy_loss + 0.001 * entropy_loss

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return policy_loss.item()

    def train(self):
        import math
        self.pbar = tqdm(total=self.n_steps, desc="Training")

        while self.total_steps < self.n_steps:
            self.rollout()
            loss = self.learn()

            if self.episodes % 100 == 0:
                avg_reward = self.total_rewards / max(1, self.episodes)
                self.writer.add_scalar("EpisodeReward", avg_reward, self.total_steps)
                self.writer.add_scalar("PolicyLoss", loss, self.total_steps)
                self.pbar.set_description(f"Reward: {avg_reward:.3f}")

                # reset counters
                self.episodes = 0
                self.total_rewards = 0.0

        torch.save(self.policy_net.state_dict(), "policy_net.pth")
        print("Models saved successfully!")
        self.writer.close()
        self.pbar.close()

env = gym.make("CartPole-v1")
agent = REINFORCE_ScalarBaseline(env, lr=0.0005, gamma=0.99, n_steps=400000)
agent.train()


