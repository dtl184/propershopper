#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env_files.env import SupermarketEnv
from env_files.utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class 
from agents.planning_agent import PlanningAgent
import pickle
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target',
        type=str,
        help='target location of this task',
        default='cart'
    )
    args = parser.parse_args()


    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = PlanningAgent(name=args.target, target=args.target)
    agent.build_map() 

    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    state = json.loads(state) 

    print(int(state['observation']['players'][0]['position'][0]/0.2), int(state['observation']['players'][0]['position'][1]/0.2))

    done = False 
    while not done:
        action, done = agent.select_action(state)
        sock_game.send(str.encode("0 " + action_commands[action]))
        state = recv_socket_data(sock_game)
        state = json.loads(state)

    # Close socket connection
    sock_game.close()

