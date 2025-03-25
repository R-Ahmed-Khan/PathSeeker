import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class PPOBase:

    """
    #######################################
    Initializer function to create buffers.
    #######################################
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    
    """
    ##############################################################
    Function to generate shuffled batches for stochastic training.
    ##############################################################
    """
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    
    """
    ######################################
    Function to append single transitions.
    ######################################
    """
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    
    """
    ##########################
    Function to clear buffers.
    ##########################
    """
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    
    """
    ###########################################################
    Initializer for Actor for continous state and action space.
    ###########################################################
    """
    def __init__(self, n_actions, input_dims, alpha, device, fc1_dims=256, fc2_dims=256, base_path=os.getcwd()):
        super(Actor, self).__init__()
        self.policy_file_path = os.path.join(base_path, 'models/policy_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )
        self.log_std = nn.Parameter(T.zeros(n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    
    """
    ################################################
    Function to get action probability distribution.
    ################################################
    """
    def forward(self, state):
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist

    
    """
    #######################
    Function to save model.
    #######################
    """
    def save_checkpoint(self):
        T.save(self.state_dict(), self.policy_file_path)

    
    """
    #######################
    Function to load model.
    #######################
    """
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.policy_file_path))

class Critic(nn.Module):
    
    """
    #########################################
    Initializer for Critic for value network.
    #########################################
    """
    def __init__(self, input_dims, alpha, device, fc1_dims=256, fc2_dims=256, base_path=os.getcwd()):
        super(Critic, self).__init__()
        self.value_file_path = os.path.join(base_path, 'models/value_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.value_file_path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.value_file_path))

class Agent:
    
    """
    ######################################################
    Initializer for PPO hyperparameters, Actor and Critic.
    ######################################################
    """
    def __init__(self, n_actions, input_dims, device, gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.memory = PPOBase(batch_size)
        self.actor = Actor(n_actions, input_dims, alpha, device)
        self.critic = Critic(input_dims, alpha, device)
        

    
    """
    #########################################
    Fuction to store transitions for batches.
    #########################################
    """
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    
    """
    #############################################################################
    Fuction to get action from distribution as well as log probability and value.
    #############################################################################
    """
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        raw_action = dist.sample()
        action = T.tanh(raw_action)  
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  
        log_prob -= T.log(1 - action.pow(2) + 1e-6).sum(dim=-1)  
        return action.cpu().detach().numpy()[0], log_prob.item(), value.item()

    
    """
    #####################
    Fuction for learning.
    #####################
    """
    def learn(self):
        actor_loss_history = []
        critic_loss_history = []
        for _ in range(self.n_epochs):
            
            # ---generate batches
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # ---GAE advantage estimate (Canonical PPO Paper)
            for t in range(len(reward_arr) - 1):
                lamda = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += lamda * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    lamda *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            #
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                
                # ---policy ratio
                new_probs = dist.log_prob(actions).sum(dim=-1)
                prob_ratio = (new_probs - old_probs).exp()

                # ---surrogate function (policy loss)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                # ---critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                actor_loss_history.append(actor_loss.item())
                critic_loss_history.append(critic_loss.item())

        self.memory.clear_memory()
        return actor_loss_history, critic_loss_history