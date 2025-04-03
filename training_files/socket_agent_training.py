#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from env_files.utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd
import argparse


cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

first_cart = True

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
        #print(f'violated norm: {current_state['violations']}')
        return norm_penalty
    prev_player = previous_state['observation']['players'][0]
    curr_player = current_state['observation']['players'][0]

    prev_dist = euclidean_distance(prev_player['position'], target_location) 
    curr_dist = euclidean_distance(curr_player['position'], target_location)

    return prev_dist-curr_dist


# def calculate_reward(previous_state, current_state):
#     # design your own reward function here
#     # You should design a function to calculate the reward for the agent to guide the agent to do the desired task
#     # print(previous_state['command_result'])
#     prev_dist_to_cart = distance_to_cart(previous_state)
#     cur_dist_to_cart = distance_to_cart(current_state)

#     prev_dist_to_exit = euclidean_distance(previous_state['observation']['players'][0]['position'], exit_pos)
#     cur_dist_to_exit = euclidean_distance(current_state['observation']['players'][0]['position'], exit_pos)

#     get_cart_rwd = 50
#     leave_rwd = 100
#     drop_cart_penalty = -25
#     no_op_penalty = -0.005

#     # if current_state['command_result']['command'] == 'NOP':
#     #     return no_op_penalty 

#     # print(current_state['observation']['players'][0]['position'])
#     print(current_state['violations'])
    
#     if current_state['gameOver']:
#         # print('Left shop')
#         # return leave_rwd
#         if current_state['observation']['players'][0]['curr_cart'] != -1:
#             print('Task finished')
#             return leave_rwd 
#         else:
#             print('Exit without a cart') 
#             return -10 
#     # else:
#     #     return 0
    
#     global first_cart 

#     if first_cart:
#         if current_state['observation']['players'][0]['curr_cart'] != -1:
#             first_cart = False 
#             print('Picking up cart task finished')
#             return get_cart_rwd
#         else:
#             return -(cur_dist_to_cart - prev_dist_to_cart )
#     else:
#         if current_state['observation']['players'][0]['curr_cart'] != -1:
#             return -(cur_dist_to_exit - prev_dist_to_exit) 
#         elif previous_state['observation']['players'][0]['curr_cart'] != -1:
#             print('Dropped cart')
#             return drop_cart_penalty 
#         else:
#             return prev_dist_to_cart - cur_dist_to_cart
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target',
        type=tuple,
        help='target location of this task',
        default='cart'
    )
    args = parser.parse_args()

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space, epsilon=0.01)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    # agent.qtable = pd.read_json('qtable.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 100
    episode_length = 1000
    for i in range(1, training_time+1):
        # sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # sock_game.connect((HOST, PORT))
        sock_game.send(str.encode("0 RESET"))  # reset the game
        first_cart = True
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        cur_ep_return = 0.
        print(f'Episode {i}') 
        print(f"Starting at {state['observation']['players'][0]['position']}")
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]
            # if cnt  < 10:
            #     action = '0 ' + 'WEST'
            # else:
            #     action = '0 ' + 'INTERACT'

            # print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            # print(next_state)
            # if len(next_state) == 0:
            #     next_state = state
            #     next_state['gameOver'] = True
            # else:
            next_state = json.loads(next_state) 
            if next_state['observation']['players'][0]['position'][0] < 0 and abs(next_state['observation']['players'][0]['position'][1] - exit_pos[1]) < 1:
                next_state['gameOver'] = True

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function 
            if reward == -25:
                next_state['gameOver'] = True # Force endding episode for losing the cart
            cur_ep_return += reward
            # print("------------------------------------------")
            # print(reward, action_commands[action_index])
            # print("------------------------------------------")
            # Update Q-table
            agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state
            # print(state['gameOver'])
            agent.qtable.to_json('qtable.json')

            if cnt > episode_length:
                # sock_game.close()
                break
        # Additional code for end of episode if needed
        print(f'Current episode reward: {cur_ep_return}')
        # sock_game.close()
        # print('Socket connection closed')

    # Close socket connection
    sock_game.close()

