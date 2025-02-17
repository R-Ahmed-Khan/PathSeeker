import argparse
import sys
import os
import ast
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from q_learning import main_q_learning 

def parse_all_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_args', type=str, required=True, help='Environment arguments (grid_size, obstacles)')
    parser.add_argument('--learn_args', type=str, required=True, help='Learning arguments (epsilon, gamma, alpha, episodes)')
    args = parser.parse_args()

    env_args = {}
    learn_args = {}

    for item in args.env_args.split('|'):
        item = item.strip()  
        if '=' in item: 
            key, value = item.split('=', 1)  
            if key == 'obstacles':
                env_args[key] = ast.literal_eval(value)
            else:
                env_args[key] = int(value) if value.isdigit() else value

    for item in args.learn_args.split('|'):
        item = item.strip()  
        if '=' in item:  
            key, value = item.split('=', 1)  
            learn_args[key] = float(value) if '.' in value else int(value)

    return env_args, learn_args

def delete():
    base_path = os.getcwd()
    artifacts_folder_path = os.path.join(base_path, 'artifacts')
    if os.path.exists(artifacts_folder_path):
        shutil.rmtree(artifacts_folder_path)  
    os.makedirs(artifacts_folder_path)  

def main():
    delete()

    env_args, learn_args = parse_all_args()

    main_q_learning(env_args, learn_args) 

if __name__ == "__main__":
    main()
