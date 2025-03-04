import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

class Plotting():

    def __init__(self):
        base_path = os.getcwd()
        self.artifacts_path = os.path.join(base_path, 'artifacts')

    def plot_reward_history(self):
        self.reward_history_file_path = os.path.join(self.artifacts_path, 'reward_history.csv')
        data = pd.read_csv(self.reward_history_file_path, header=None)
        values = data[0].tolist()
        alpha = 0.01
        smoothed_rewards = []
        smoothed_value = 0

        for reward in values:
            smoothed_value = alpha * reward + (1 - alpha) * smoothed_value
            smoothed_rewards.append(smoothed_value)

        plt.figure(figsize=(8, 6))
        plt.plot(smoothed_rewards, color='blue')

        plt.title('Rewards vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid()
        image_path = os.path.join(self.artifacts_path, 'rewards_history.png')
        print(image_path)
        plt.savefig(image_path)
        plt.show()

    def plot_temporal_error(self):
        self.temp_diff_error_file_path = os.path.join(self.artifacts_path, 'temp_diff_error.csv')
        data = pd.read_csv(self.temp_diff_error_file_path, header=None)
        values = data[0].tolist()
        alpha = 1
        smoothed_temp_error = []
        smoothed_value = 0

        for reward in values:
            smoothed_value = alpha * reward + (1 - alpha) * smoothed_value
            smoothed_temp_error.append(smoothed_value)

        plt.figure(figsize=(8, 6))
        plt.plot(smoothed_temp_error, color='blue')

        plt.title('Temporal Difference Error vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Temporal Difference Error')
        plt.grid()
        image_path = os.path.join(self.artifacts_path, 'temporal_difference_error.png')
        print(image_path)
        plt.savefig(image_path)
        plt.show()


def main_plot():
    plot = Plotting()
    plot.plot_reward_history()
    plot.plot_temporal_error()

# if __name__ == "__main__":
#     main_plot()


