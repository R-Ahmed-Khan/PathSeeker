from utils import Utils
import random
import numpy as np

class Environment():
    def __init__(self, grid_size):
        self.no_of_actions = 4
        self.ugv_grid_positions = grid_size*grid_size*self.no_of_actions
        self.target_positions = grid_size*grid_size
        self.no_of_observations = self.ugv_grid_positions*self.target_positions
        pass

    def reset(self, grid_size):
        while True:
            start = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.choice([0, 90, 180, 270]))
            goal = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            if start != goal:
                break
            print(start)
        state = Utils.encode_state(start, goal, grid_size)
        return state
        
    def actions(self):
        ugv_inputs = [(1, 0), (-1, 0), (0, 90), (0, -90)]
        return ugv_inputs
    
    def choose_action(self, state, actions, epsilon, q_table):
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(len(actions)))
        else:
            return np.argmax(q_table[state, :])
        
    def is_valid_move(self, x, y, grid_size):
        """ Check if the robot is within bounds """
        return 0 <= x < grid_size and 0 <= y < grid_size
        
    def get_next_state(self, state, action, grid_size):
        """ Move the robot in 3D space based on the action """
        ugv_position, target_position = Utils.decode_state(state, grid_size)
        x, y, theta = ugv_position
        temp_actions = self.actions()
        if temp_actions[action][1] == 0:
            dxy = temp_actions[action][0]
            if theta == 90 or theta == 270:
                next_ugv_position = (x, y + dxy, theta)
            if theta == 0 or theta == 180:
                next_ugv_position = (x + dxy, y, theta)
        elif temp_actions[action][0] == 0:
            dtheta = temp_actions[action][1]
            next_theta = theta + dtheta
            if next_theta == 360:
                next_theta = 0
            elif next_theta == -90:
                next_theta = 270
            elif next_theta == -180:
                next_theta = 180
            elif next_theta == -270:
                next_theta = 90
            next_ugv_position = (x, y, next_theta)
            
        if self.is_valid_move(next_ugv_position[0], next_ugv_position[1], grid_size):
            state = Utils.encode_state(next_ugv_position, target_position, grid_size)
            return state
        else:
            return state

    def get_reward(self, state, next_state, obstacles, grid_size):
        next_ugv_position, target_position = Utils.decode_state(next_state, grid_size)
        ugv_position, target_position = Utils.decode_state(state, grid_size)

        distance_to_target = np.linalg.norm(np.array(next_ugv_position[:2]) - np.array(target_position))

        # Calculate the direction the UGV is facing

        delta_x = next_ugv_position[0] - ugv_position[0]
        delta_y = next_ugv_position[1] - ugv_position[1]

        # Calculate the angle to the target in radians
        angle_to_target = np.arctan2(delta_y, delta_x) * 180 / np.pi  # Convert to degrees

        # Normalize the angle to be one of 0°, 90°, 270°, or 360°
        if -45 <= angle_to_target < 45:
            angle_to_target = 0  # East
        elif 45 <= angle_to_target < 135:
            angle_to_target = 90  # North
        elif 135 <= angle_to_target < 225:
            angle_to_target = 180  # South 
        elif 225 <= angle_to_target < 315:
            angle_to_target = 270  # West
        else:
            angle_to_target = 360  # Wrap around for East again

        orientation_diff = abs(ugv_position[2] - angle_to_target)

        if next_ugv_position[:2] == target_position:
            return 200
        elif next_ugv_position[:2] in obstacles:
            return -1000
        else:
            return - orientation_diff - distance_to_target
    
