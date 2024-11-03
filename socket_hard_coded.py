#Author: Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

# from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
from hard_agent import HardcodedAgent
import pickle
import pandas as pd


if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = HardcodedAgent(action_space)
    agent.qtable = pd.read_json('hardcode_qtable.json')

    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 1
    episode_length = 400
    has_basket = False 
    has_garlic = False
    has_milk = False
    has_banana = False
    checked_out = False
    cur_task = 0
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(state) 

            if action_index == 6 and not has_basket:
                print('got basket')
                has_basket = True 
                agent.toggle_state('basket') 

            if action_index == 6 and cur_task == 1:
                print('got garlic')
                has_garlic = True 
                agent.toggle_state('garlic') 

            if action_index == 6 and cur_task == 2:
                print('got milk')
                has_milk = True 
                agent.toggle_state('milk') 

            if action_index == 6 and cur_task == 3:
                print('got banana')
                has_banana = True 
                agent.toggle_state('banana')
            action = "0 " + action_commands[action_index] 

            if action_index == 6:
                # action_commands[6] = 'SELECT'
                cur_task += 1

            sock_game.send(str.encode(action))  # send action to env 

            next_state = recv_socket_data(sock_game)  # get observation from env 
            if action_index == 6:
                sock_game.send(str.encode(action)) 
                next_state = recv_socket_data(sock_game) 
            if action_index == 6 and cur_task == 4:
                print('checked out')
                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
            next_state = json.loads(next_state)
            if action_index == 6:
                print(next_state['observation']['baskets'])
            if has_garlic and action_index == 6 and cur_task == 2:
                print(next_state['observation']['players'][0])
                has_garlic = True 

            # Update state
            state = next_state
            agent.qtable.to_json('hardcode_qtable.json')

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

