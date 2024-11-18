#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
import numpy as np
from env import SupermarketEnv
from time import sleep
from utils import recv_socket_data

from navigation_agent import NavigationAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd
import argparse
from termcolor import colored


obj_pos_dict = {
    'init': [1.2, 15.6], # default starting position
    'cart': [1, 18.5],
    'basket': [3.5, 18.5],
    'exit': [-0.8, 15.6],

    # milk
    'milk': [5.5, 1.5],
    'chocolate milk': [9.5, 1.5],
    'strawberry milk': [13.5, 1.5], 

    # fruit
    'apple': [5.5, 5.5],
    'orange': [7.5, 5.5],
    'banana': [9.5, 5.5],
    'strawberry': [11.5, 5.5],
    'raspberry': [13.5, 5.5],

    # meat
    'sausage': [5.5, 9.5],
    'steak': [7.5, 9.5],
    'chicken': [11.5, 9.5],
    'ham': [13.5, 9.5], 

    # cheese
    'brie cheese': [5.5, 13.5],
    'swiss cheese': [7.5, 13.5],
    'cheese wheel': [9.5, 13.5], 

    # veggie 
    'garlic': [5.5, 17.5], 
    'leek': [7.5, 17.5], 
    'red bell pepper': [9.5, 17.5], 
    'carrot': [11.5, 17.5],
    'lettuce': [13.5, 17.5], 

    # something else 
    'avocado': [5.5, 21.5],
    'broccoli': [7.5, 21.5],
    'cucumber': [9.5, 21.5],
    'yellow bell pepper': [11.5, 21.5], 
    'onion': [13.5, 21.5], 

    'checkout': [1, 4.5]
} 

def get_init_pos(start_obj):
    return obj_pos_dict[start_obj] + np.random.uniform(-0.5, 0.5, 2)

cur_ep_known_places = np.zeros([300, 300])


def euclidean_distance(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def calculate_reward(previous_state, current_state, agent, target_object='cart'):
    anxiety = -0.005 

    target_location = obj_pos_dict[target_object]

    curr_player = current_state['observation']['players'][0]
    prev_player = previous_state['observation']['players'][0] 

    prev_dist = euclidean_distance(prev_player['position'], target_location) 
    curr_dist = euclidean_distance(curr_player['position'], target_location) 

    if curr_dist < 0.5: # Consider success 
        print(colored('Success!', 'green'))
        return 1000

    return (prev_dist-curr_dist)*5+anxiety -20*len(current_state['violations'])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--start',
        type=str,
        default='init'
    )

    parser.add_argument(
        '--target' ,
        type=str, 
        default='cart'
    ) 

    parser.add_argument(
        '--port',
        type=int,
        default=9000
    )

    parser.add_argument(
        '--policy_filename',
        type=str,
        default='navigation_qtable.json'
    )

    parser.add_argument(
        '--granularity',
        type=float,
        default=0.4
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--continue_training',
        type=bool,
        default=False
    )

    args = parser.parse_args()

    policy_filename = args.policy_filename
    target_location = args.target
    granularity = args.granularity
    epsilon = args.epsilon

    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = NavigationAgent(name='navigate ' + args.target, epsilon=0.9, granularity=granularity) 
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 2000
    episode_length = 500
    history = []
    for i in range(1, training_time+1):
        sock_game.send(str.encode("0 RESET"))  # reset the game

        first_cart = True
        state = recv_socket_data(sock_game)
        state = json.loads(state)

        # reinitialize agent position
        state['observation']['players'][0]['position'] = get_init_pos(args.start) 
        foo = state['observation']['players'][0]['position']
        sock_game.send(str.encode("0 RESET"))  # reset the game
        cnt = 0
        cur_ep_return = 0.
        print(f'Episode {i}') 
        cur_ep_known_places = np.zeros([300, 300])
        print(f"Starting at {state['observation']['players'][0]['position']}")
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]
            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            # during the first iteration of an episode, next_state has the results of RESET and the first action.
            # this strips off the results of RESET because json.loads expects only one json object
            if cnt == 1:
                next_state = next_state.decode('utf-8').split('\n')[1]
                next_state.strip()
            next_state = json.loads(next_state) 
            if next_state['observation']['players'][0]['position'][0] < 0 and abs(next_state['observation']['players'][0]['position'][1] - obj_pos_dict['exit'][1]) < 1:
                next_state['gameOver'] = True

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state, agent, args.target) 
            cur_ep_return += reward
            agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state

            if cnt > episode_length:
                history.append(0)
                break
            if reward == 1000:
                history.append(1)
                break
        # Additional code for end of episode if needed
        history = history[-50:]
        if i % 100 == 0:
            print(f"Success rate becomes: {np.mean(history)}") 
            agent.save_qtables()
        # sock_game.close()
        # print('Socket connection closed')

    # Close socket connection
    sock_game.close()

