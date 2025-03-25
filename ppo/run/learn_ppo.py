import os
import sys
import numpy as np
import argparse
import torch 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from environment import CustomEnv
from ppo import Agent
from utils import*

parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--time_steps', type=int, default=1500000, help='total number of time steps')
parser.add_argument('--memory', type=int, default=600, help='memory length')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--alpha', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.2, help='policy clip parameter')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--gae_lamda', type=float, default=0.95, help='GAE Lambda')
args = parser.parse_args()

class Learn_PPO():

    def __init__(self):
        self.device = torch.device(args.dvc)
        self.timesteps = args.time_steps
        self.N = args.memory
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.gae_lamda = args.gae_lamda

    def moving_average(self,data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    def learn(self):
        env = CustomEnv()
        
        agent = Agent(n_actions=env.action_space.shape[0], 
                      input_dims=env.observation_space.shape,
                      device=self.device,
                      gamma=self.gamma, 
                      alpha=self.alpha, 
                      gae_lambda=self.gae_lamda,
                      policy_clip=self.epsilon,
                      batch_size=self.batch_size,
                      n_epochs=self.n_epochs,
                      )
        
        best_score = -np.inf
        score_history = []
        timesteps_history = []
        actor_loss_history = []
        actor_loss_timesteps = []
        critic_loss_history = []
        critic_loss_timesteps = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0  
        while n_steps < self.timesteps:
            observation, _ = env.reset()
            done = False
            score = 0
            while not done and n_steps < self.timesteps:  
                action, prob, val = agent.choose_action(observation)
                observation_, reward, terminate, truncate, _ = env.step(action)
                done = terminate or truncate
                n_steps += 1  
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % self.N == 0: 
                    actor_losses, critic_losses = agent.learn()
                    actor_loss_history.extend(actor_losses)
                    actor_loss_timesteps.extend([n_steps] * len(actor_losses))
                    critic_loss_history.extend(critic_losses)
                    critic_loss_timesteps.extend([n_steps] * len(actor_losses))
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            timesteps_history.append(n_steps) 
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            print(f'Timesteps: {n_steps}, Score: {score:.1f}, Avg Score: {avg_score:.1f}, Learning Steps: {learn_iters}')

        plot_learning_curve_reward(timesteps_history, score_history)
        plot_learning_curve_actor_loss(actor_loss_timesteps, actor_loss_history)
        plot_learning_curve_critic_loss(critic_loss_timesteps, actor_loss_history)
        smoothed_actor_loss = self.moving_average(actor_loss_history, window_size=10)
        smoothed_actor_timesteps = actor_loss_timesteps[:len(smoothed_actor_loss)]
        plot_learning_curve_actor_loss_smoothed(smoothed_actor_timesteps, smoothed_actor_loss)
        smoothed_critic_loss = self.moving_average(critic_loss_history, window_size=10)
        smoothed_critic_timesteps = critic_loss_timesteps[:len(smoothed_critic_loss)]
        plot_learning_curve_critic_loss_smoothed(smoothed_critic_timesteps, smoothed_critic_loss)
            

def main():
    delete()
    ppo = Learn_PPO()
    ppo.learn()

if __name__ == '__main__':
    main()
