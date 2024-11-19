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

class IRLAgent:
    def __init__(self, feature_matrix, action_space, transition_probability, discount=0.9, epochs=100, learning_rate=0.1):
        self.feature_matrix = feature_matrix  
        self.action_space = action_space      
        self.transition_probability = transition_probability  
        self.discount = discount              
        self.epochs = epochs                  
        self.learning_rate = learning_rate    
        self.reward = None  # initially empty since we want to learn this                 
        self.policy = None                    
    
    def learn_reward(self, trajectories):
        self.reward = irl(
            feature_matrix=self.feature_matrix,
            n_actions=self.action_space,
            discount=self.discount,
            transition_probability=self.transition_probability,
            trajectories=trajectories,
            epochs=self.epochs,
            learning_rate=self.learning_rate
        )
    
    def update_policy(self):
        
        self.policy = find_policy(
            n_states=self.feature_matrix.shape[0],
            n_actions=self.action_space,
            transition_probability=self.transition_probability,
            reward=self.reward,
            discount=self.discount
        )
    
    def choose_action(self, state):
        
        # Convert state into its corresponding index for the policy
        state_index = self.trans(state)
        return np.argmax(self.policy[state_index])
    
    def trans(self, state, granularity=0.5):
        obs = []
        # x range: (0, 20)
        # y range: (0, 25)
        obs.append(int((state['observation']['players'][0]['position'][0] + 1) / granularity))
        obs.append(int((state['observation']['players'][0]['position'][1] + 1) / granularity))
        obs.append(int(state['observation']['players'][0]['direction']))
        obs.append(state['observation']['players'][0]['curr_cart'] != -1)
        obs = obs[0] + obs[1] * 40 + obs[2] * 1600 + obs[3] * 6400
        return obs
