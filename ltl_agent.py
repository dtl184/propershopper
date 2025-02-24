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
    def __init__(self, n_states, goal, alpha=0.5, gamma=0.9, epsilon=0.8, mini_epsilon=0.05, decay=0.9999, filename=None, x_max=19, y_max=24):
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.action_space = ['NORTH', 'SOUTH', 'EAST', 'WEST']  
        self.qtable = pd.DataFrame(columns=[i for i in range(len(self.action_space))])
        self.trajectories = self.load_trajectories(filename)
        self.transitions = self.learn_from_trajectories()
        self.x_min = 1
        self.x_max = x_max
        self.y_min = 2
        self.y_max = y_max
    
    def trans(self, state, granularity=0.15):
        """
        Extracts relevant state variables from the current state for learning.
        For the LTL agent's current task that is simply its position.
        """
        x = round(state['observation']['players'][0]['position'][0] / granularity) * granularity
        y = round(state['observation']['players'][0]['position'][1] / granularity) * granularity
        position = (x, y)
        return json.dumps({'position': position}, sort_keys=True)
    
    def load_trajectories(self, file_name):
        """
        Load trajectories from a text file.
        """
        trajectories = []
        with open(file_name, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    trajectory = eval(line)
                    trajectories.append(trajectory)
        return trajectories



    def check_add(self, state):
        """
        Ensures the state is in the Q-table, adding it if not present and it's not an obligated state.

        Args:
            state (dict): The state to check and potentially add to the Q-table.
        """
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))

        state_index = self.coords_to_state(x, y)

        if state_index == -1:
            return

        # If state is obligated no need to add to q table
        if len(self.transitions[state_index][0]) > 0:  
            return

        serialized_state = self.trans(state)
        if serialized_state not in self.qtable.index:
            self.qtable.loc[serialized_state] = pd.Series(np.zeros(len(self.action_space)), index=[i for i in range(len(self.action_space))])
    

    def learn_from_trajectories(self):
        """
        Determines what states are obligated/permitted based on agent trajectories.

        Args:
            trajectories: A list of agent trajectories, each of which is a list of (state, action) tuples.
        """
        transitions = {state: (tuple(), tuple()) for state in range(self.n_states)}
        for trajectory in self.trajectories:
            for i in range(len(trajectory) - 1):
                state_index, action = trajectory[i]


                # Get current obligated and permitted transitions
                obligated, permitted = transitions[state_index]

                # If a new action is observed, update transitions
                if action not in obligated and action not in permitted:
                    if len(obligated) == 0 and len(permitted) == 0:
                        # No obligated transitions, make this action obligated
                        transitions[state_index] = ((action,), permitted)
                    else:
                        # Move all obligated to permitted and add the new action
                        transitions[state_index] = (tuple(), obligated + permitted + (action,))

        return transitions
    
    
    def learning(self, action, state, next_state, reward):
        """
        Updates Q-table after the agent receives a reward, but only for permitted transitions.

        Args:
            action: The agent's current action.
            state: The current agent state.
            next_state: The state obtained after applying the action in the current state.
            reward: The reward received after performing the action.
        """
        self.check_add(state)
        self.check_add(next_state)
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        state_index = self.coords_to_state(x, y)

        _, permitted = self.transitions[state_index]
        if action not in permitted:
            # Do not update if the action is not permitted
            return

        q_sa = self.qtable.loc[self.trans(state), action]

        # Handle obligated states
        x_next = int(round(next_state['observation']['players'][0]['position'][0]))
        y_next = int(round(next_state['observation']['players'][0]['position'][1]))
        next_state_index = self.coords_to_state(x_next, y_next)
        
        if len(self.transitions[next_state_index][0]) == 1:  # Check if obligated
            # If next_state is obligated, ignore max_next_q_sa
            new_q_sa = q_sa + self.alpha * (reward - q_sa)
        else:
            # Standard Q-learning update for non-obligated states
            max_next_q_sa = self.qtable.loc[self.trans(next_state), :].max()
            new_q_sa = q_sa + self.alpha * (reward + self.gamma * max_next_q_sa - q_sa)

        self.qtable.loc[self.trans(state), action] = new_q_sa

    
    def choose_action(self, state):
        """
        Selects an action based on the following rules:
        - Always takes the obligated action if the state has one.
        - In permitted states, explores with probability epsilon or exploits the best action.
        - If no transitions are defined, defaults to a random action.

        Parameters:
            state (dict): The current state of the agent.

        Returns:
            int: The chosen action index.
        """
        self.check_add(state)
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        state_index = self.coords_to_state(x, y)

        if state_index == -1:
            return random.choice(range(len(self.action_space)))

        obligated, permitted = self.transitions[state_index]

        

        if len(obligated) > 0:
            if random.uniform(0, 1) < .95:
                return obligated[0]
            else:
                return random.choice(range(len(self.action_space)))


        if len(permitted) > 0:
            if np.random.uniform(0, 1) < self.epsilon:
                action = random.choice(permitted)
            else:
                q_values = self.qtable.loc[self.trans(state)]
                action =  max(permitted, key=lambda action: q_values[action])
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                q_values = self.qtable.loc[self.trans(state)]
                action =  max(range(len(self.action_space)), key=lambda action: q_values[action])
            else:
                action = random.choice(range(len(self.action_space)))

        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay
        return action



    
    def state_to_coords(self, state, granularity=.15):
        """
        Convert a state index back to (x, y) coordinates based on the granularity.

        Args:
            state (int): State index.
            granularity (float): The granularity to discretize the position.

        Returns:
            tuple: (x, y) coordinates of the state.
        """
        total_x_values = round((self.x_max - self.x_min) / 1) + 1
        y_index = state // total_x_values
        x_index = state % total_x_values
        x = self.x_min + x_index * granularity
        y = self.y_min + y_index * granularity
        return x, y

    def coords_to_state(self, x, y, granularity=1):
        """
        Convert (x, y) coordinates to a state index based on the granularity.

        Args:
            x (float): X-coordinate of the position.
            y (float): Y-coordinate of the position.
            granularity (float): The granularity to discretize the position.

        Returns:
            int: State index.
        """
        total_x_values = round((self.x_max - self.x_min) / granularity) + 1
        x_index = round(x / granularity) - round(self.x_min / granularity)
        y_index = round(y / granularity) - round(self.y_min / granularity)
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

        # Draw grid lines within the bounds
        for x in range(self.x_max + 1):
            x_pos = grid_x_min + (x / self.x_max) * (grid_x_max - grid_x_min)
            ax.axvline(x_pos, color='black', linewidth=0.5, zorder=1)
        for y in range(self.y_max + 1):
            y_pos = grid_y_min + (y / self.y_max) * (grid_y_max - grid_y_min)
            ax.axhline(y_pos, color='black', linewidth=0.5, zorder=1)

        # Add arrows for transitions within the grid bounds
        edge_arrow_directions = {
            'NORTH': (0, -0.45, 0, -0.3),  # Flipped: arrow points downward
            'SOUTH': (0, 0.45, 0, 0.3),   # Flipped: arrow points upward
            'EAST': (-0.45, 0, -0.3, 0),  # Flipped: arrow points left
            'WEST': (0.45, 0, 0.3, 0)     # Flipped: arrow points right
        }
        edge_arrow_offsets = {'NORTH': (-0.1, 0), 'SOUTH': (0.1, 0), 'EAST': (0, -0.1), 'WEST': (0, 0.1)}
        direction_encoding = {'NORTH': 1, 'SOUTH': 2, 'EAST': 3, 'WEST': 4}

        for y in range(self.y_max):
            for x in range(self.x_max):
                state = (self.y_max - 1 - y) * self.x_max + x
                if state in self.transitions and (self.transitions[state][0] or self.transitions[state][1]):
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
                            not self.transitions[neighbor_state][0] and not self.transitions[neighbor_state][1]):
                            continue

                        if direction_encoding[direction] in self.transitions[state][0]:
                            color = 'black'
                        elif direction_encoding[direction] in self.transitions[state][1]:
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
    
    def count_transition_states(self):
        """
        Counts the number of states with obligated actions, permitted actions, and those with neither.
        """
        obligated_count = 0
        permitted_count = 0
        neither_count = 0

        for state, (obligated, permitted) in self.transitions.items():
            if obligated:
                obligated_count += 1
            elif permitted:
                permitted_count += 1
            else:
                neither_count += 1

        print(f'Total states: {self.n_states}\n Obligated: {obligated_count}\n Permitted: {permitted_count}\n Limbo: {neither_count}')









    def save_qtable(self):
        self.qtable.to_json('qtable_foo.json') 


    