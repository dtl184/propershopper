import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class 
from planner import HyperPlanner
from get_obj_agent import GetObjAgent
from constants import *
import pickle
import pandas as pd
import argparse
from termcolor import colored

class IrlAgent:
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.9, mini_epsilon=0.05, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               
    
    def feature_vector(self, state):
        # Return the feature vector for the current state
        return state['observation']['players'][0]['position']


    def irl(self, feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):

        n_states, d_states = feature_matrix.shape

        # Initialise weights.
        alpha = rn.uniform(size=(d_states,))

        # Calculate the feature expectations \tilde{phi}.
        feature_expectations = find_feature_expectations(feature_matrix,
                                                        trajectories)

        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            r = feature_matrix.dot(alpha)
            expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                            transition_probability, trajectories)
            grad = feature_expectations - feature_matrix.T.dot(expected_svf)

            alpha += learning_rate * grad








def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=9000
    )
    args = parser.parse_args()

    planner = HyperPlanner(obj_list)
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT)) 

    training_time = 100
    episode_length = 800
    for i in range(1, training_time+1):
        # start = False 
        planner.reset()

        sock_game.send(str.encode("0 RESET"))  # reset the game

        state = recv_socket_data(sock_game)
        state = json.loads(state)
        planner.parse_shopping_list(state)
        agent = planner.get_agent()

        print(f'Current Plan: {planner.plan}')
        init_inventory()
        while not planner.plan_finished(): 
            done = False
            cnt = 0
            current_task = planner.get_task() 
            print(f'Current Task: {current_task}')
            while not done:
                cnt += 1
                action_index, finish = agent.choose_action(state)
                action = "0 " + agent.action_commands[action_index] 
                # print(action)
                sock_game.send(str.encode(action))  # send action to env 

                next_state = recv_socket_data(sock_game)  # get observation from env
                # fill with random state if state is empty
                next_state = json.loads(next_state) 
                if action == '0 INTERACT' and current_task.split()[0] == 'get':
                    sock_game.send(str.encode(action))  # remove dialogue box 
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                    if current_task.split()[1] in special_food_list:
                        sock_game.send(str.encode(action))  # remove dialogue box 
                        next_state = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(next_state) 

                if action == '0 INTERACT' and current_task.split()[0] == 'pay':
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                    sock_game.send(str.encode('0 INTERACT'))  # remove dialogue box 
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                # if next_state['observation']['players'][0]['position'][0] < 0 and abs(next_state['observation']['players'][0]['position'][1] - exit_pos[1]) < 1:
                #     next_state['gameOver'] = True

                # Define the reward based on the state and next_state
                reward = calculate_reward(state, next_state, target=current_task.split()[1], task=current_task.split()[0])  # You need to define this function 
                if reward == 100 or finish:
                    print("Success!") 
                    done = True 

                agent.learning(action_index, reward, state, next_state)

                # Update state
                state = next_state

                if cnt > episode_length:
                    print("\nepisode count exceeded\n")
                    break
            agent.save_qtables()
            if not done:
                break # mission failed. Restart everything 
            else:
                planner.update() 
                agent = planner.get_agent() 
                if agent is None:
                    print(colored('Whole task succeeded', 'green'))
    sock_game.close()

