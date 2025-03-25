import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

analysis_folder_path = os.path.join(os.getcwd(), 'analysis')
models_folder_path = os.path.join(os.getcwd(), 'models')

def delete():
    if os.path.exists(analysis_folder_path):
        shutil.rmtree(analysis_folder_path)  
    os.makedirs(analysis_folder_path)
    if os.path.exists(models_folder_path):
        shutil.rmtree(models_folder_path)  
    os.makedirs(models_folder_path)

def plot_learning_curve(x, scores):
    reward_figure_path = os.path.join(analysis_folder_path, 'reward_curve')
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(reward_figure_path)
    
def plot_learning_curve_reward(x, scores):
    reward_figure_path = os.path.join(analysis_folder_path, 'reward_curve')
    plt.figure()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Mean Reward over 100 episodes')
    plt.grid()
    plt.savefig(reward_figure_path)

def plot_learning_curve_actor_loss(x, losses):
    loss_figure_path = os.path.join(analysis_folder_path, 'actor_loss_curve')
    plt.figure()
    plt.plot(x, losses)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Policy Loss')
    plt.grid()
    plt.savefig(loss_figure_path)

def plot_learning_curve_actor_loss_smoothed(x, losses):
    loss_figure_path = os.path.join(analysis_folder_path, 'actor_loss_curve_smoothed')
    plt.figure()
    plt.plot(x, losses)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Policy Loss')
    plt.grid()
    plt.savefig(loss_figure_path)

def plot_learning_curve_critic_loss(x, losses):
    loss_figure_path = os.path.join(analysis_folder_path, 'critic_loss_curve')
    plt.figure()
    plt.plot(x, losses)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Critic Loss')
    plt.grid()
    plt.savefig(loss_figure_path)

def plot_learning_curve_critic_loss_smoothed(x, losses):
    loss_figure_path = os.path.join(analysis_folder_path, 'critic_loss_curve_smoothed')
    plt.figure()
    plt.plot(x, losses)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Critic Loss')
    plt.grid()
    plt.savefig(loss_figure_path)