import sys
import os
import numpy as np
import csv

from environment import Environment
from utils import Utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis')))
from plot_learning_data import main_plot

class Q_Learning():

    def __init__(self, grid_size, obstacles, epsilon, gamma, alpha, episodes):
        self.env = Environment(grid_size)
        self.ugv_inputs = self.env.actions()
        self.eps_steps = 100000
        self.q_table = np.zeros([self.env.no_of_observations, self.env.no_of_actions])
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        base_path = os.getcwd()
        artifacts_path = os.path.join(base_path, 'artifacts')
        self.reward_history_file_path = os.path.join(artifacts_path, 'reward_history.csv')

    def train_q_learning(self):
        """ Train the robot using Q-learning for a number of episodes """
        reward_history = []
        temp_diff_error = []
        for episode in range(self.episodes):
            state = self.env.reset(self.grid_size)
            done = False
            total_reward = 0
            ep_temp_diff = 0

            step = 0

            while not done:
                ugv_position, target_position = Utils.decode_state(state, self.grid_size)
                action = self.env.choose_action(state, self.ugv_inputs, self.epsilon, self.q_table)
                next_state = self.env.get_next_state(state, action, self.grid_size)
                reward = self.env.get_reward(state, next_state, self.obstacles, self.grid_size)
                old_q_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state, :])
                self.q_table[state, action] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max)
                temp_diff = (reward + self.gamma * next_max) - old_q_value
                state = next_state
                ugv_position, target_position = Utils.decode_state(state, self.grid_size)
                step += 1
                total_reward += reward
                ep_temp_diff += temp_diff
                if ugv_position[:2] == target_position[:2] or ugv_position[:2] in self.obstacles:
                    done = True
                    print(f"Episode {episode+1} ended with reward: {total_reward}")
                    break
                elif step >= self.eps_steps:
                    print(f"Episode {episode+1} ended as steps reached")
                    break
            reward_history.append(total_reward)
            temp_diff_error.append(ep_temp_diff)

        return reward_history, temp_diff_error, self.q_table
    
    def storing_learning_data(self, reward_history, temp_diff_error, q_table):
        base_path = os.getcwd()
        artifacts_path = os.path.join(base_path, 'artifacts')
        reward_history_file_path = os.path.join(artifacts_path, 'reward_history.csv')
        with open(reward_history_file_path, 'a') as file:
            for value in reward_history:
                file.write(f"{value}\n")

        temp_diff_error_file_path = os.path.join(artifacts_path, 'temp_diff_error.csv')
        with open(temp_diff_error_file_path, 'a') as file:
            for value in temp_diff_error:
                file.write(f"{value}\n")

        q_table_file_path = os.path.join(artifacts_path, 'q_table.csv')
        with open(q_table_file_path, 'a') as file:
            for row in q_table:
                for i, value in enumerate(row):
                    if i < len(row)-1:
                        file.write(f"{value}, ")
                    else:
                        file.write(f"{value}")
                file.write("\n")

        obstacles_file_path = os.path.join(artifacts_path, 'obstacles.csv')
        with open(obstacles_file_path, 'a') as file:
            for obstacle in self.obstacles:
                file.write(f"{obstacle[0]}, {obstacle[1]}\n")

        grid_size_file_path = os.path.join(artifacts_path, 'grid_size.csv')
        with open(grid_size_file_path, 'a') as file:
            file.write(f"{self.grid_size}\n")



def main_q_learning(env_args, learn_args):
    learn = Q_Learning(env_args['grid_size'], env_args['obstacles'], learn_args['epsilon'], learn_args['gamma'], learn_args['alpha'],  learn_args['episodes'])
    reward_history, temp_diff_error, q_table = learn.train_q_learning()
    learn.storing_learning_data(reward_history, temp_diff_error, q_table)
    main_plot()
