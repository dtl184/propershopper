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
from helper import project_collision
import numpy as np
import pandas as pd
import json  
from irl.maxent import irl, find_feature_expectations, find_expected_svf, find_policy  


class IRLAgent:
    def __init__(self, n_states, trajectories, discount=0.9, epochs=100, learning_rate=0.1):
        self.action_space = ['NORTH', 'SOUTH', 'WEST', 'EAST']
        self.n_states = n_states
        self.feature_matrix = self.generate_feature_matrix()
        self.trajectories = trajectories
        self.x_min = 1
        self.x_max = 19
        self.y_min = 2
        self.y_max = 24
        self.transition_probability = None
        self.discount = discount              
        self.epochs = epochs                  
        self.learning_rate = learning_rate    
        self.reward = None  # initially empty since we want to learn this                 
        self.policy = None
    
    def set_reward(self, r):
        self.reward = r




    def generate_transition_matrix(self, state):
        x_min, x_max = 1, 19
        y_min, y_max = 2, 24

        grid_width = x_max - x_min + 1
        grid_height = y_max - y_min + 1

        n_actions = len(self.action_space)
        transition_matrix = np.zeros((self.n_states, n_actions, self.n_states))

        # Populate the transition matrix
        for i in range(self.n_states):
            xi, yi = self.inverse_trans(i)  # Current state's coordinates

            for j, action in enumerate(self.action_space):
                if action == 'NORTH':
                    dx, dy = 0, -1
                elif action == 'SOUTH':
                    dx, dy = 0, 1
                elif action == 'EAST':
                    dx, dy = 1, 0
                elif action == 'WEST':
                    dx, dy = -1, 0

                # Calculate the intended next position
                intended_x, intended_y = xi + dx, yi + dy

                # Prepare the object structure for collision checking
                obj = {
                    "position": [xi, yi],  
                    "width": 1,  
                    "height": 1  
                }

                  # Convert action string to Direction enum

                # Check if movement is within valid bounds
                if x_min <= intended_x <= x_max and y_min <= intended_y <= y_max:
                    # Check for projected collision
                    if project_collision(obj, state, j):
                        next_state = i  # Stay in the same place if collision is detected
                    else:
                        next_state = self.trans(state)
                else:
                    next_state = i  # Stay in the same place if movement is out of bounds

                # Assign transition probability (only allow if there's no collision)
                val = 1.0 if next_state != i else 0.0
                transition_matrix[i, j, next_state] = val

        self.transition_probability = transition_matrix


    def feature_vector(self, i):

        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def generate_feature_matrix(self):

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n)
            features.append(f)
        return np.array(features)                    
    
    def learn_reward(self):
        self.reward = irl(
            feature_matrix=self.feature_matrix,
            n_actions=len(self.action_space),
            discount=self.discount,
            transition_probability=self.transition_probability,
            trajectories=self.trajectories,
            epochs=self.epochs,
            learning_rate=self.learning_rate
        )
    
    def update_policy(self):
        
        self.policy = find_policy(
            n_states=self.feature_matrix.shape[0],
            n_actions=len(self.action_space),
            transition_probability=self.transition_probability,
            reward=self.reward,
            discount=self.discount
        )
    
    def choose_action(self, state):

        action_rewards = []
        
        for action in self.action_space:
            if action == 'NORTH':
                dx, dy = 0, -1
            elif action == 'SOUTH':
                dx, dy = 0, 1
            elif action == 'EAST':
                dx, dy = 1, 0
            elif action == 'WEST':
                dx, dy = -1, 0
        
            next_x = state[0] + dx
            next_y = state[1] + dy

            next_state_index = self.trans(next_x, next_y)
            action_rewards.append(self.reward[next_state_index])

        act = action_rewards.index(max(action_rewards))

        if act == 3:
            return 2
        elif act == 2:
            return 3
        elif act == 1:
            return 0
        else:
            return 1
        

    def trans(self, state):
        x, y = state['observation']['players'][0]['position']

        total_x_values = self.x_max - self.x_min + 1
        x_index = round(x) - self.x_min
        y_index = round(y) - self.y_min
        return y_index * total_x_values + x_index

    def inverse_trans(self, state_index):
        total_x_values = self.x_max - self.x_min + 1
        y_index = state_index // total_x_values
        x_index = state_index % total_x_values
        x = x_index + self.x_min
        y = y_index + self.y_min
        return x, y
