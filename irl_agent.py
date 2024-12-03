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

import numpy as np
import pandas as pd
import json  
from irl.maxent import irl, find_feature_expectations, find_expected_svf, find_policy  

class TransitionProbability:
    def __init__(self, n_states, n_actions, x_min, x_max, y_min, y_max):
        self.n_states = n_states
        self.n_actions = n_actions
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.granularity = 0.1  # Assumes one decimal place precision

        # Compute total x values based on grid dimensions
        self.total_x_values = round((x_max - x_min) / self.granularity) + 1

    def state_to_coords(self, state):
        """Convert a state index to (x, y) coordinates."""
        y_index = state // self.total_x_values
        x_index = state % self.total_x_values
        x = self.x_min + x_index * self.granularity
        y = self.y_min + y_index * self.granularity
        return x, y

    def coords_to_state(self, x, y):
        """Convert (x, y) coordinates to a state index."""
        x_index = round((x - self.x_min) / self.granularity)
        y_index = round((y - self.y_min) / self.granularity)
        return y_index * self.total_x_values + x_index

    def __getitem__(self, indices):
        """Dynamically compute the transition probability."""
        i, j, k = indices
        x, y = self.state_to_coords(i)

        # Define actions (dx, dy): [NORTH, SOUTH, EAST, WEST]
        actions = [(0, -0.1), (0, 0.1), (0.1, 0), (-0.1, 0)]
        dx, dy = actions[j]

        # Compute the next state coordinates
        next_x, next_y = x + dx, y + dy

        # Check bounds
        if not (self.x_min <= next_x <= self.x_max and self.y_min <= next_y <= self.y_max):
            return 0  # Out of bounds -> no transition

        # Compute the expected next state
        expected_state = self.coords_to_state(next_x, next_y)
        return 1 if expected_state == k else 0


class IRLAgent:
    def __init__(self, n_states, trajectories, discount=0.9, epochs=100, learning_rate=0.5):
        self.action_space = ['NORTH', 'SOUTH', 'WEST', 'EAST']
        self.n_states = n_states
        self.feature_matrix = self.generate_feature_matrix()
        self.trajectories = trajectories
        self.transition_probability = self.generate_transition_matrix() #TransitionProbability(
#     n_states=437,  # Example number of states for 0.1 precision
#     n_actions=4,  # Number of actions
#     x_min=0.5,
#     x_max=19,
#     y_min=2.0,
#     y_max=24
# )   
        self.discount = discount              
        self.epochs = epochs                  
        self.learning_rate = learning_rate    
        self.reward = None  # initially empty since we want to learn this                 
        self.policy = None
    
    def set_reward(self, r):
        self.reward = r


    def generate_transition_matrix(self):

        x_min, x_max = 1, 19
        y_min, y_max = 2, 24

        grid_width = x_max - x_min + 1
        grid_height = y_max - y_min + 1
        # assert self.n_states == grid_width * grid_height, "n_states must match grid dimensions."

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

                if x_min <= intended_x <= x_max and y_min <= intended_y <= y_max:
                    # Valid move: compute next state
                    next_state = self.trans(intended_x, intended_y)
                else:
                    # Invalid move: stay in the same state
                    next_state = i

                # Set the transition probability
                transition_matrix[i, j, next_state] = 1.0

        return transition_matrix

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
        
    
    # def trans(self, x, y):
    #     x_min, y_min = 0.5, 2.1  # Minimum x and y values
    #     x_max, y_max = 19.0, 24.0  # Maximum x and y values
    #     granularity = 0.1  # Step size for precision

    #     # Compute the indices for x and y
    #     x_index = round((x - x_min) / granularity)
    #     y_index = round((y - y_min) / granularity)

    #     # Total number of x values
    #     total_x_values = round((x_max - x_min) / granularity) + 1

    #     # Calculate the unique index
    #     return y_index * total_x_values + x_index

    def trans(self, x, y, x_min=1, y_min=2, x_max=19, y_max=24):

        grid_width = x_max - x_min + 1
        return (y - y_min) * grid_width + (x - x_min)

    def inverse_trans(self, state_index, x_min=1, y_min=2, x_max=19, y_max=24):
        """
        Converts a unique integer state index back into the (x, y) coordinates.

        Parameters:
            state_index (int): The unique state index.
            x_min (int): The minimum x value. Default is 0.
            y_min (int): The minimum y value. Default is 0.
            x_max (int): The maximum x value. Default is 19.
            y_max (int): The maximum y value. Default is 24.

        Returns:
            tuple: The (x, y) coordinates.
        """
        grid_width = x_max - x_min + 1
        y = state_index // grid_width + y_min
        x = state_index % grid_width + x_min
        return x, y
