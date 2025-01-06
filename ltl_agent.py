import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LTLAgent:
    """
    An agent that learns from trajectory visitation frequencies. 
    The agent identifies permitted and obligated transitions from each state.
    
    Transitions are obligated the first time they are seen. If a different transition from the same state is seen,
    then that transition and all previously obligated transitions from that state are now permitted.
    """
    def __init__(self, n_states, x_max=19, y_max=23):
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
        self.action_space = ['NORTH', 'SOUTH', 'WEST', 'EAST']  # Possible actions as (dx, dy) tuples
        self.transitions = {state: {'obligated': set(), 'permitted': set()} for state in range(n_states)}
        self.x_min = 1
        self.x_max = x_max
        self.y_min = 2
        self.y_max = y_max
    
    def learn_from_trajectories(self, trajectories):
        """
        Processes the list of trajectories to determine which transitions are permitted and obligated.

        Parameters:
        -----------
        trajectories : list of list of tuples (state, action)
            Each trajectory is a list of (state, action) pairs.
        """
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                state, action = trajectory[i]
                next_state, _ = trajectory[i + 1]
                
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

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid background
        for y in range(self.y_max):
            for x in range(self.x_max):
                state = (self.y_max - 1 - y) * self.x_max + x  # Calculate the state index from (x, y) position (top-left origin)
                if (x, y) == (2, 6):  # Color the specific square at (3, 19) yellow
                    facecolor = 'yellow'
                elif state in self.transitions and (self.transitions[state]['obligated'] or self.transitions[state]['permitted']):
                    facecolor = 'white'  # White if the state has transitions
                else:
                    facecolor = 'black'  # Black if the state has no transitions
                rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=facecolor)
                ax.add_patch(rect)
        
        total_x_values = self.x_max
        action_directions = {'NORTH': (0, -0.6), 'SOUTH': (0, 0.6), 'EAST': (0.6, 0), 'WEST': (-0.6, 0)}  # Tail length increased
        
        # Add tiny arrows to the edge of white squares pointing to adjacent white squares based on actual transitions
        edge_arrow_directions = {
            'NORTH': (0, 0.45, 0, 0.3), 
            'SOUTH': (0, -0.45, 0, -0.3), 
            'EAST': (0.45, 0, 0.3, 0), 
            'WEST': (-0.45, 0, -0.3, 0)
        }
        
        edge_arrow_offsets = {
            'NORTH': (0.1, 0),  # Small horizontal offset to avoid overlap
            'SOUTH': (-0.1, 0),
            'EAST': (0, 0.1),
            'WEST': (0, -0.1)
        }

        direction_encoding = {
            'NORTH': 1,  # Small horizontal offset to avoid overlap
            'SOUTH': 2,
            'EAST': 3,
            'WEST': 4
        }
        
        for y in range(self.y_max):
            for x in range(self.x_max):
                state = (self.y_max - 1 - y) * self.x_max + x
                if state in self.transitions and (self.transitions[state]['obligated'] or self.transitions[state]['permitted']):
                    center_x = x + 0.5
                    center_y = y + 0.5
                    
                    for direction, (dx, dy, arrow_dx, arrow_dy) in edge_arrow_directions.items():
                        if direction_encoding[direction] in self.transitions[state]['obligated']:
                            color = 'black'  # Obligated transition
                        elif direction_encoding[direction] in self.transitions[state]['permitted']:
                            color = 'gray'  # Permitted transition
                        else:
                            continue
                        
                        offset_x, offset_y = edge_arrow_offsets[direction]
                        ax.arrow(center_x + dx + offset_x, center_y + dy + offset_y, arrow_dx * 0.8, arrow_dy * 0.8, head_width=0.05, head_length=0.05, fc=color, ec=color)

        plt.xlim(0, self.x_max)
        plt.ylim(0, self.y_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.title('Grid Showing Permitted and Obligated Transitions')
        plt.show()
