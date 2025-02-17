import numpy as np
import pandas as pd
import os
from utils import Utils
import ast 
from environment import Environment
import matplotlib.pyplot as plt
import ast
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis')))
from animate import main_animate

class Policy():
    def __init__(self):
        base_path = os.getcwd()
        artifacts_path = os.path.join(base_path, 'artifacts')
        self.q_table_file_path = os.path.join(artifacts_path, 'q_table.csv')
        self.obstacle_file_path = os.path.join(artifacts_path, 'obstacles.csv')
        self.grid_file_path = os.path.join(artifacts_path, 'grid_size.csv')

    def get_q_table(self):
        data = []
        with open(self.q_table_file_path, 'r') as file:
            for line in file:
                row = [float(value.replace(',', '')) for value in line.split()]
                data.append(row)
        self.q_table = np.array(data)

    def get_obstacles(self):
        data = []
        with open(self.obstacle_file_path, 'r') as file:
            for line in file:
                row = [float(value.replace(',', '')) for value in line.split()]
                data.append(row)
        self.obstacles = [tuple(inner_list) for inner_list in data]

    def get_grid(self):
        with open(self.grid_file_path, 'r') as file:
            for line in file:
                self.grid_size = float(line)

    def get_environment(self):
        self.env = Environment(self.grid_size)

    def test_policy(self, ugv_position, target_position):
        target_position= ast.literal_eval(target_position)
        """ Test the learned policy by following the Q-table and visualize the path """
        idx = 0
        path = [ugv_position]  
        visited_states = set()  

        while ugv_position[:2] != target_position[-1]:  
            current_target = target_position[idx]

            state = Utils.encode_state(ugv_position, current_target, self.grid_size)

            ugv_position, target_position_c = Utils.decode_state(state, self.grid_size)

            action = np.argmax(self.q_table[state, :])

            next_state = self.env.get_next_state(state, action, self.grid_size)

            if next_state == state:
                ugv_position, target_position_c = Utils.decode_state(state, self.grid_size)
                print(f"Agent is stuck at {ugv_position}")
                break

            state = next_state
            visited_states.add(state)
            ugv_position, target_position_c = Utils.decode_state(state, self.grid_size)
            path.append(ugv_position)
            idx += 1
            if idx >= len(target_position):
                idx = len(target_position) - 1

        return path
    
def main_policy(args):
    policy = Policy()
    policy.get_q_table()
    policy.get_obstacles()
    policy.get_grid()
    policy.get_environment()
    path = policy.test_policy(args['start'], args['targets'])
    print('Path Taken by the UGV: ')
    print(path)
    main_animate(path, args['targets'])