import argparse
import sys
import os
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from policy import main_policy

def parse_policy_args():
    # Set up argument parser
    parser = argparse.ArgumentParser()

    # Command line argument for policy_args
    parser.add_argument('--policy_args', type=str, required=True, help='Policy arguments (start, target)')

    # Parse command line arguments
    args = parser.parse_args()

    # Initialize the dictionary
    policy_args = {}

    for item in args.policy_args.split('|'):
        item = item.strip()  
        if '=' in item: 
            key, value = item.split('=', 1)  
            if key == 'start':
                policy_args[key] = ast.literal_eval(value)  
            elif key == 'target':
                policy_args[key] = ast.literal_eval(value)  
            else:
                policy_args[key] = value  
    return policy_args

def main():

    policy_args = parse_policy_args()

    main_policy(policy_args)

if __name__ == "__main__":
    main()


