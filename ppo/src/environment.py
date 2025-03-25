import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.L = 0.1 
        self.dt = 0.1  
        self.episode_length = 400
        self.v_max = 0.6 ; self.v_min = 0
        self.steer_max = np.deg2rad(35); self.steer_min = np.deg2rad(-35)
        self.x_min, self.x_max = -2, 2
        self.y_min, self.y_max = -2, 2
        self.theta_min, self.theta_max = np.deg2rad(-180), np.deg2rad(180)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.theta_min, self.x_min, self.y_min]),
            high=np.array([self.x_max, self.y_max, self.theta_max, self.x_max, self.y_max]),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        target_x = np.random.uniform(self.x_min, self.x_max)
        target_y = np.random.uniform(self.y_min, self.y_max)
        theta = np.arctan2(target_y - y, target_x - x) + np.deg2rad(np.random.uniform(-30, 30)) 
        self.observation = np.array([x, y, theta, target_x, target_y], dtype=np.float32)
        self.step_count = 0
        return self.observation, {}
    
    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    def scale_action(self, action):
        steer_angle = self.steer_min + (action[0] + 1) * 0.5 * (self.steer_max - self.steer_min)
        v = self.v_min + (action[1] + 1) * 0.5 * (self.v_max - self.v_min)
        return steer_angle, v
    
    def get_next_observation(self, observation, action):
        steer_angle, v = self.scale_action(action)
        x, y, theta, target_x, target_y = observation
        next_x = x + v * np.cos(theta) * self.dt
        next_y = y + v * np.sin(theta) * self.dt
        next_theta = self.wrap_angle(theta + (v / self.L) * np.tan(steer_angle) * self.dt)
        return np.array([next_x, next_y, next_theta, target_x, target_y], dtype=np.float32)
    
    def reward(self, observation, step_count):
        current_position = observation[:2]
        theta = observation[2]
        target_point = observation[3:]
        self.distance_target = np.linalg.norm(current_position - target_point)
        r_p =  1 / (self.distance_target + 0.01) 
        desired_theta = np.arctan2(target_point[1] - current_position[1], target_point[0] - current_position[0])
        alignment_error = self.wrap_angle(desired_theta - theta)
        r_theta = 10*np.cos(alignment_error) 
        termination = self.distance_target < 0.2
        truncation = (step_count > self.episode_length 
                      or current_position[0] < self.x_min
                      or current_position[0] > self.x_max
                      or current_position[1] < self.y_min
                      or current_position[1] > self.y_max
                    )
        reward = r_p + r_theta 
        return reward, termination, truncation
    
    def step(self, action):
        self.observation = self.get_next_observation(self.observation, action)
        self.step_count += 1
        reward, terminated, truncated = self.reward(self.observation, self.step_count)
        terminated = bool(terminated)
        truncated = bool(truncated)
        info = {}
        return self.observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass 
