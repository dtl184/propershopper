#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class 
from GoalConditionedAgent import GoalConditionedAgent
import pickle
import pandas as pd
import argparse


cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

first_cart = True 

obj_pos_dict = {
    'cart': [1.5, 18.5],
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

def distance_to_cart(state):
    agent_position = state['observation']['players'][0]['position']
    if agent_position[0] > 1.5:
        cart_distances = [euclidean_distance(agent_position, cart_pos_right)]
    else:
        cart_distances = [euclidean_distance(agent_position, cart_pos_left)]
    return min(cart_distances)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def calculate_reward(previous_state, current_state, target_location=(1, 17.5)):
    norm_penalty = -5
    if current_state['violations'] is not None:
        print(f"violated norm: {current_state['violations']}")
        return norm_penalty
    prev_player = previous_state['observation']['players'][0]
    curr_player = current_state['observation']['players'][0]

    prev_dist = euclidean_distance(prev_player['position'], target_location) 
    curr_dist = euclidean_distance(curr_player['position'], target_location)

    return prev_dist-curr_dist
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target',
        type=tuple,
        help='target location of this task',
        default='cart'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=9000
    )
    args = parser.parse_args()

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = GoalConditionedAgent(obj_pos_dict)
    # agent.load_qtables()
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 1
    episode_length = 1000
    for i in range(1, training_time+1):
        # sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # sock_game.connect((HOST, PORT))
        sock_game.send(str.encode("0 RESET"))  # reset the game
        first_cart = True
        state = recv_socket_data(sock_game)
        state = json.loads(state) 
        agent.parse_shopping_list(state)
        print(agent.plan)
        cnt = 0
        cur_ep_return = 0.
        print(f'Episode {i}') 
        print(f"Starting at {state['observation']['players'][0]['position']}")
        while not state['gameOver']:
            cnt += 1
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]
            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state) 
            if next_state['observation']['players'][0]['position'][0] < 0 and abs(next_state['observation']['players'][0]['position'][1] - exit_pos[1]) < 1:
                next_state['gameOver'] = True

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function 
            if reward == -25:
                next_state['gameOver'] = True # Force endding episode for losing the cart
            cur_ep_return += reward
            agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state
            agent.qtable.to_json('qtable.json')

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed
        print(f'Current episode reward: {cur_ep_return}')
    sock_game.close()

