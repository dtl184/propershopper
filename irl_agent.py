import json
import random
import socket
import matplotlib.pyplot as plt
import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data
import matplotlib.image as mpimg
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




    def generate_transition_matrix(self):
        """
        Generates a transition matrix where each valid state (1 in valid_states)
        can transition to adjacent valid states with probability 1.
        """
        x_min, x_max = 1, 19
        y_min, y_max = 2, 24

        n_actions = len(self.action_space)
        transition_matrix = np.zeros((self.n_states, n_actions, self.n_states))

        # Load valid states
        try:
            with open("valid_states.txt", "r") as f:
                valid_states = np.array(json.load(f))  # 1D list indexed by state index
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error: valid_states.txt not found or corrupted.")
            return

        # Iterate over all states
        for i in range(self.n_states):
            # If the state is invalid, no transitions allowed
            if valid_states[i] == 0:
                continue

            xi, yi = self.inverse_trans(i)  # Convert state index to coordinates

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
                next_x, next_y = xi + dx, yi + dy

                # Check if the next state is within bounds
                if x_min <= next_x <= x_max and y_min <= next_y <= y_max:
                    next_state = self.coord_trans(next_x, next_y)  # Get the next state index
                    
                    # Check if next state is valid in valid_states
                    if valid_states[next_state] == 1:
                        transition_matrix[i, j, next_state] = 1.0  # Assign transition probability

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
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))

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
        
            next_x = x + dx
            next_y = y + dy

            next_state_index = self.coord_trans(next_x, next_y)
            action_rewards.append(-1 * self.reward[next_state_index])

        act = action_rewards.index(max(action_rewards))

        if act == 3:
            return 2
        elif act == 2:
            return 3
        elif act == 1:
            return 0
        else:
            return 1

    def coord_trans(self, x, y):
        total_x_values = self.x_max - self.x_min + 1
        x_index = round(x) - self.x_min
        y_index = round(y) - self.y_min
        return y_index * total_x_values + x_index

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
    


    def visualize_valid_states(
        self, filename="valid_states.txt", image_path="map.png", grid_shape=(19, 23), 
        x_offset=13, y_offset=52
    ):
        """
        Loads valid_states.txt, overlays a 19x23 grid on map.png, 
        colors active grid squares, and displays the index in each grid cell.
        Saves the visualization as 'valid_states_grid.png'.
        """

        # Step 1: Load the valid states file
        try:
            with open(filename, "r") as f:
                state_list = json.load(f)  # Load list of 0s and 1s
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error: valid_states.txt not found or corrupted.")
            return

        # Step 2: Ensure data size matches expected grid size
        if len(state_list) != grid_shape[0] * grid_shape[1]:
            print("Error: Data size does not match expected grid dimensions.")
            return

        # Step 3: Reshape into 2D grid
        state_grid = np.array(state_list).reshape(grid_shape)

        # Step 4: Load the background image
        try:
            img = mpimg.imread(image_path)
            img_height, img_width, _ = img.shape
        except FileNotFoundError:
            print("Error: The file map.png was not found.")
            return

        # Step 5: Compute grid offsets and extents
        x_extent_min = x_offset / img_width * grid_shape[0]  # Adjust for x offset
        x_extent_max = (img_width - x_offset) / img_width * grid_shape[0]
        y_extent_min = y_offset / img_height * grid_shape[1]  # Adjust for y offset
        y_extent_max = (img_height - x_offset) / img_height * grid_shape[1]

        # Step 6: Set up figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[0, grid_shape[0], grid_shape[1], 0], aspect='auto', zorder=0)  # Keep top-left origin

        # Step 7: Draw the grid, color visited cells, and add index numbers
        for x in range(grid_shape[0]):  # 19 columns
            for y in range(grid_shape[1]):  # 23 rows
                x_pos = x_extent_min + ((x + self.x_min - 1) / grid_shape[0]) * (x_extent_max - x_extent_min)
                y_pos = y_extent_min + ((y + self.y_min - 2) / grid_shape[1]) * (y_extent_max - y_extent_min)

                # Calculate the 1D index for the grid cell
                index = self.coord_trans(x + self.x_min, y + self.y_min)

                # Color the grid square
                if state_list[index] == 1:  
                    color = "white"  # Visited
                else:
                    color = "black"  # Unvisited
                ax.add_patch(plt.Rectangle((x_pos, y_pos), 1, 1, color=color, alpha=0.7, zorder=1))

                # Add the index number in each grid cell
                ax.text(x_pos + 0.5, y_pos + 0.5, str(index), color="red", fontsize=8, 
                        ha="center", va="center", zorder=3, fontweight="bold")

        # Step 8: Draw grid lines within the correct extents
        for x in range(grid_shape[0] + 1):  # Vertical grid lines (19)
            x_pos = x_extent_min + ((x + self.x_min - 1) / grid_shape[0]) * (x_extent_max - x_extent_min)
            ax.axvline(x_pos, color='gray', linewidth=0.5, zorder=2)

        for y in range(grid_shape[1] + 1):  # Horizontal grid lines (23)
            y_pos = y_extent_min + ((y + self.y_min - 2) / grid_shape[1]) * (y_extent_max - y_extent_min)
            ax.axhline(y_pos, color='gray', linewidth=0.5, zorder=2)

        # Step 9: Formatting
        ax.set_xlim(0, grid_shape[0])
        ax.set_ylim(grid_shape[1], 0)  # Invert y-axis to match top-left origin
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Valid States Overlayed on Map with Index Numbers')

        # Step 10: Save the figure instead of showing it
        plt.savefig('valid_states_grid.png', dpi=300, bbox_inches='tight')


    def visualize_reward(self, image_path="map.png", grid_shape=(19, 23), x_offset=13, y_offset=52):
        """
        Visualizes the reward function by overlaying a 19x23 grid on map.png, 
        with grid squares colored based on their reward values.
        Saves the visualization as 'reward_grid.png'.
        """
        
        # Ensure reward array is the correct size
        if len(self.reward) != grid_shape[0] * grid_shape[1]:
            print("Error: Reward array size does not match expected grid dimensions.")
            return

        # Reshape reward array into a grid
        reward_grid = np.array(self.reward).reshape(grid_shape)

        # Normalize reward values for coloring (min-max scaling)
        min_reward, max_reward = np.min(reward_grid), np.max(reward_grid)
        norm_rewards = (reward_grid - min_reward) / (max_reward - min_reward + 1e-5)  # Avoid div by 0

        # Load the background image
        try:
            img = mpimg.imread(image_path)
            img_height, img_width, _ = img.shape
        except FileNotFoundError:
            print("Error: The file map.png was not found.")
            return

        # Compute grid offsets and extents
        x_extent_min = x_offset / img_width * grid_shape[0]
        x_extent_max = (img_width - x_offset) / img_width * grid_shape[0]
        y_extent_min = y_offset / img_height * grid_shape[1]
        y_extent_max = (img_height - x_offset) / img_height * grid_shape[1]

        # Set up figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[0, grid_shape[0], grid_shape[1], 0], aspect='auto', zorder=0)  # Keep top-left origin

        # Set up a colormap for rewards
        cmap = plt.cm.coolwarm  # Blue (low) → Red (high)
        norm = plt.Normalize(vmin=min_reward, vmax=max_reward)

        # Draw the grid and color each square based on reward
        for x in range(grid_shape[0]):  # 19 columns
            for y in range(grid_shape[1]):  # 23 rows
                x_pos = x_extent_min + ((x + self.x_min - 1) / grid_shape[0]) * (x_extent_max - x_extent_min)
                y_pos = y_extent_min + ((y + self.y_min - 2) / grid_shape[1]) * (y_extent_max - y_extent_min)

                # Get the reward value and corresponding color
                index = self.coord_trans(x, y)
                reward_value = self.reward[index]
                color = cmap(norm(reward_value))

                # Draw colored grid square
                ax.add_patch(plt.Rectangle((x_pos, y_pos), 1, 1, color=color, alpha=0.7, zorder=1))


        # Draw grid lines
        for x in range(grid_shape[0] + 1):  
            x_pos = x_extent_min + ((x + self.x_min - 1) / grid_shape[0]) * (x_extent_max - x_extent_min)
            ax.axvline(x_pos, color='gray', linewidth=0.5, zorder=2)

        for y in range(grid_shape[1] + 1):  
            y_pos = y_extent_min + ((y + self.y_min - 2) / grid_shape[1]) * (y_extent_max - y_extent_min)
            ax.axhline(y_pos, color='gray', linewidth=0.5, zorder=2)

        # Formatting
        ax.set_xlim(0, grid_shape[0])
        ax.set_ylim(grid_shape[1], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Reward Visualization on Grid')

        # Add colorbar for reward values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Reward Value")

        # Save the figure
        plt.savefig('reward_grid.png', dpi=300, bbox_inches='tight')
