import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd 
import json

class LTLAgent:
    """
    An agent that learns from trajectory visitation frequencies. 
    The agent identifies permitted and obligated transitions from each state.
    
    Transitions are obligated the first time they are seen. If a different transition from the same state is seen,
    then that transition and all previously obligated transitions from that state are now permitted.
    """
    def __init__(self, n_states, goal, gamma=0.9, epsilon=0.3, mini_epsilon=0.1, decay=0.9999, x_max=19, y_max=23):
        """
        Initializes the LTLAgent.

        Parameters:
        -----------
        n_states : int
            The total number of possible states.
        x_max : int
            The width of the grid (number of columns).
        y_max : int
            The height of the grid (number of rows).
        """
        self.n_states = n_states
        self.goal = goal
        self.gamma = gamma
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.action_space = ['NORTH', 'SOUTH', 'WEST', 'EAST']  
        self.transitions = {state: {'obligated': set(), 'permitted': set()} for state in range(n_states)}
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.x_min = 1
        self.x_max = x_max
        self.y_min = 2
        self.y_max = y_max
    

    def learn_from_trajectories(self, trajectories):
        """
        Determines what states are obligated/permitted based on agent trajectories.
        If a state only has one transition out, that transition is obligated. If a state
        has multiple transitions out, all those transitions are permitted. 

        Args:
            trajectories:
                A list of agent trajectories, each of which is a list of (state, action) tuples
        
        """
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                state, action = trajectory[i]
                
                
                # If this is the first action seen from this state, mark it as obligated
                if state not in self.transitions:
                    self.transitions[state] = {'obligated': set(), 'permitted': set()}
                    self.transitions[state]['obligated'].add(action)
                
                # If this state has only one action, it stays obligated
                if len(self.transitions[state]['obligated']) == 1 and action not in self.transitions[state]['obligated']:
                    self.transitions[state]['permitted'].update(self.transitions[state]['obligated'])
                    self.transitions[state]['permitted'].add(action)
                    self.transitions[state]['obligated'].clear()
                
                # If the action has not been seen before and it's the first action, keep it obligated
                if action not in self.transitions[state]['obligated'] and action not in self.transitions[state]['permitted']:
                    self.transitions[state]['obligated'].add(action)
    
    def trans(self, state):
        """
        Extracts relevant state variables from the current state for learning.
        For the LTL agent's current task that is simply its position.
        """

        # round the position to the nearest int (for simplicity)
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        position = (x, y)

        return json.dumps({'position': position}, sort_keys=True)
    
    def check_add(self, state):

        serialized_state = self.trans(state)
        if serialized_state not in self.qtable.index:
            self.qtable_norms.loc[serialized_state] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
    
    
    def learning(self, action, state, next_state, reward):
        """
        Updates qtable after agent receives a reward

        Args:
            action: 
                Agent's current action
            state: 
                Current agent state
            next_state:
                State obtained after applying action in current state
            reward:
                Agent's reward after performing action
        """

        self.check_add(state)
        self.check_add(next_state)


        if reward != 0:
            
            q_sa = self.qtable.loc[self.trans(state), action]
            max_next_q_sa = self.qtable.loc[self.trans(next_state), :].max()
            new_q_sa = q_sa + self.alpha * (reward + self.gamma * max_next_q_sa - q_sa)
            self.qtable.loc[self.trans(state), action] = new_q_sa
    
    def choose_action(self, state):
        p = np.random.uniform(0, 1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            q_values = self.qtable_norms.loc[self.trans(state)].to_list()

        
    
    def state_to_coords(self, state, granularity=1):
        """Convert a state index to (x, y) coordinates."""
        total_x_values = round((self.x_max - self.x_min) / granularity) + 1
        y_index = state // total_x_values
        x_index = state % total_x_values
        x = self.x_min + x_index * granularity
        y = self.y_min + y_index * granularity
        return x, y

    def coords_to_state(self, x, y, granularity=1):
        """Convert (x, y) coordinates to a state index."""
        total_x_values = self.x_max - self.x_min + 1
        x_index = round(x) - self.x_min
        y_index = round(y) - self.y_min
        return y_index * total_x_values + x_index
    
    def visualize(self):
        """
        Visualize the grid overlaid on the map.png image, accounting for pixel offsets.

        Returns:
        --------
        None
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        fig, ax = plt.subplots(figsize=(10, 10))

        # Load and display the background image
        try:
            img = mpimg.imread("map.png")
            img_height, img_width, _ = img.shape
            print(f"Image dimensions: {img_width}x{img_height}")

            # Adjust grid extents based on image dimensions and pixel offsets
            x_extent_start = 13 / img_width * self.x_max
            x_extent_end = (img_width - 13) / img_width * self.x_max
            y_extent_start = 52 / img_height * self.y_max
            y_extent_end = (img_height - 13) / img_height * self.y_max

            ax.imshow(img, extent=[0, self.x_max, 0, self.y_max], aspect='auto', zorder=0)
        except FileNotFoundError:
            print("Error: The file map.png was not found.")
            return

        # Calculate grid bounds within the described pixel offsets
        grid_x_min = 13 / img_width * self.x_max
        grid_x_max = (img_width - 13) / img_width * self.x_max
        grid_y_min = 13 / img_height * self.y_max
        grid_y_max = (img_height - 52) / img_height * self.y_max

        basket_x, basket_y = 2, 7  # Coordinates for the basket square

        # Draw grid lines within the bounds
        for x in range(self.x_max + 1):
            x_pos = grid_x_min + (x / self.x_max) * (grid_x_max - grid_x_min)
            ax.axvline(x_pos, color='black', linewidth=0.5, zorder=1)
        for y in range(self.y_max + 1):
            y_pos = grid_y_min + (y / self.y_max) * (grid_y_max - grid_y_min)
            ax.axhline(y_pos, color='black', linewidth=0.5, zorder=1)

        # Tint the basket square yellow
        basket_x_min = grid_x_min + (basket_x / self.x_max) * (grid_x_max - grid_x_min)
        basket_x_max = grid_x_min + ((basket_x + 1) / self.x_max) * (grid_x_max - grid_x_min)
        basket_y_min = grid_y_min + (basket_y / self.y_max) * (grid_y_max - grid_y_min)
        basket_y_max = grid_y_min + ((basket_y + 1) / self.y_max) * (grid_y_max - grid_y_min)
        ax.add_patch(
            plt.Rectangle(
                (basket_x_min, basket_y_min),
                basket_x_max - basket_x_min,
                basket_y_max - basket_y_min,
                color='yellow',
                alpha=0.5,
                zorder=2
            )
        )

        # Add arrows for transitions within the grid bounds
        edge_arrow_directions = {
            'NORTH': (0, 0.45, 0, 0.3),
            'SOUTH': (0, -0.45, 0, -0.3),
            'EAST': (0.45, 0, 0.3, 0),
            'WEST': (-0.45, 0, -0.3, 0)
        }
        edge_arrow_offsets = {'NORTH': (0.1, 0), 'SOUTH': (-0.1, 0), 'EAST': (0, 0.1), 'WEST': (0, -0.1)}
        direction_encoding = {'NORTH': 1, 'SOUTH': 2, 'EAST': 3, 'WEST': 4}

        for y in range(self.y_max):
            for x in range(self.x_max):
                state = (self.y_max - 1 - y) * self.x_max + x
                if state in self.transitions and (self.transitions[state]['obligated'] or self.transitions[state]['permitted']):
                    center_x = grid_x_min + (x + 0.5) / self.x_max * (grid_x_max - grid_x_min)
                    center_y = grid_y_min + (y + 0.5) / self.y_max * (grid_y_max - grid_y_min)
                    for direction, (dx, dy, arrow_dx, arrow_dy) in edge_arrow_directions.items():
                        neighbor_x = x + (1 if direction == 'EAST' else -1 if direction == 'WEST' else 0)
                        neighbor_y = y + (1 if direction == 'SOUTH' else -1 if direction == 'NORTH' else 0)
                        neighbor_state = (self.y_max - 1 - neighbor_y) * self.x_max + neighbor_x

                        # Skip arrows leading to invalid states
                        if neighbor_x < 0 or neighbor_x >= self.x_max or neighbor_y < 0 or neighbor_y >= self.y_max:
                            continue
                        if neighbor_state not in self.transitions or (
                            not self.transitions[neighbor_state]['obligated'] and not self.transitions[neighbor_state]['permitted']):
                            continue

                        if direction_encoding[direction] in self.transitions[state]['obligated']:
                            color = 'black'
                        elif direction_encoding[direction] in self.transitions[state]['permitted']:
                            color = 'gray'
                        else:
                            continue

                        offset_x, offset_y = edge_arrow_offsets[direction]
                        ax.arrow(center_x + dx * (grid_x_max - grid_x_min) / self.x_max + offset_x,
                                 center_y + dy * (grid_y_max - grid_y_min) / self.y_max + offset_y,
                                 arrow_dx * (grid_x_max - grid_x_min) / self.x_max * 0.8,
                                 arrow_dy * (grid_y_max - grid_y_min) / self.y_max * 0.8,
                                 head_width=0.05, head_length=0.05, fc=color, ec=color, zorder=3)

        plt.xlim(0, self.x_max)
        plt.ylim(0, self.y_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.title('Grid Showing Permitted and Obligated Transitions')
        plt.show()



    