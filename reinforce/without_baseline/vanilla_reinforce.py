import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, nvec_s: int, nvec_u: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(nvec_s, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nvec_u)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        dist = torch.distributions.Categorical(logits=x)
        action = dist.sample()
        entropy = dist.entropy()
        log_prob = dist.log_prob(action)
        return action, log_prob, entropy
    
class Reinforce:
    def __init__(self, env:gym.Env, lr, gamma, n_steps):

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer_policy = AdamW(self.policy_net.parameters(), lr=lr)

        self.total_steps = 0

        # stats
        self.episodes = 0
        self.total_rewards = 0
        self.mean_episode_reward = 0

    def rollout(self):

        state, info = self.env.reset()
        terminated = False
        truncated = False
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        while True:

            action, log_prob, entropy = self.policy_net(torch.from_numpy(state).float().to(self.device))
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())

            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.entropies.append(entropy)

            state = next_state

            self.total_rewards += reward
            self.total_steps += 1
            self.pbar.update(1)
            if terminated or truncated:
                self.writer.add_scalar("Cumulative Reward", self.total_rewards, self.total_steps)
                self.episodes += 1

                if self.episodes % 100 ==0:

                    self.mean_episode_reward = self.total_rewards / self.episodes
                    self.pbar.set_description(f"Reward: {self.mean_episode_reward :.3f}")
                    self.writer.add_scalar("Reward", self.mean_episode_reward, self.total_steps)
                    self.episodes =0
                    self.total_rewards = 0

                break
                
    def calculate_returns(self):    
        next_returns = 0
        returns = np.zeros_like(self.rewards, dtype=np.float32)
        for i in reversed(range(len(self.rewards))):
            next_returns = self.rewards[i] + self.gamma * next_returns
            returns[i] = next_returns   

        return torch.tensor(returns, dtype = torch.float32).to(self.device)
    
    def learn(self):

        self.log_probs = torch.stack(self.log_probs)
        self.entropies = torch.stack(self.entropies) 

        returns = self.calculate_returns()

        advantages = returns.squeeze() 

        policy_loss = -torch.mean(advantages.detach() * self.log_probs)

        entropy_loss = -torch.mean(self.entropies)
        policy_loss = policy_loss + 0.001 * entropy_loss
        self.writer.add_scalar("Policy Loss", policy_loss, self.total_steps)


        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer_policy.step()

    def train(self):
        self.writer = SummaryWriter(log_dir="runs/reinforce_logs/REINFORCE_NO_BASELINE")

        self.pbar = tqdm(total=self.n_steps, position=0, leave=True)

        while self.total_steps < self.n_steps:

            self.rollout()
            self.learn()
        
        torch.save(self.policy_net.state_dict(), "policy_net.pth")
        print("Models saved successfully!")

env = gym.make("CartPole-v1")

agent = Reinforce(env, 0.0005, 0.99, 400000)
agent.train()
