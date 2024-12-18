import random
import numpy as np
import matplotlib.pyplot as plt
class LTLAgent:
    """
    An agent that learns from trajectory visitation frequencies. 
    The agent identifies permitted and obligated states based on visitation counts.
    
    If a state is visited once in the trajectories, it is considered permitted.
    If a state is visited more than once, it is considered obligated.
    """
    def __init__(self, n_states):
        """
        Initializes the LTLAgent.

        Parameters:
        -----------
        n_states : int
            The total number of possible states.
        """
        self.n_states = n_states
        self.action_space = ['NORTH', 'SOUTH', 'EAST', 'WEST']  # Possible actions as (dx, dy) tuples
        self.state_visit_counts = np.zeros(n_states, dtype=int)  # Visitation counts for each state
        self.permitted_states = set()  # States that are permitted
        self.obligated_states = set()  # States that are obligated
        self.x_min = 1
        self.x_max = 19
        self.y_min = 2
        self.y_max = 24
        self.goal_state = self.coords_to_state(3, 19)  # State corresponding to (3, 19)
    
    def visualize(self):
        """
        Create a 2D grid visualization of the obligated and permitted states.
        
        Parameters:
        -----------
        obligated_states : set of int
            Set of state indices that are obligated.
        permitted_states : set of int
            Set of state indices that are permitted.
        
        Returns:
        --------
        None
        """
        # Initialize the grid to white (0 = white, 0.5 = light gray, 1 = black)
        grid = np.ones((self.y_max, self.x_max))  # Rows (y) x Columns (x)
        
        # Calculate the number of states per row
        total_x_values = self.x_max
        
        # Set colors for obligated and permitted states
        for state in self.obligated_states:
            y_index = state // total_x_values
            x_index = state % total_x_values
            if 0 <= y_index < self.y_max and 0 <= x_index < self.x_max:
                grid[y_index, x_index] = 0  # Dark (obligated) = black
        
        for state in self.permitted_states:
            y_index = state // total_x_values
            x_index = state % total_x_values
            if 0 <= y_index < self.y_max and 0 <= x_index < self.x_max:
                # Only update if not already an obligated state
                if grid[y_index, x_index] != 0:
                    grid[y_index, x_index] = 0.5  # Light (permitted) = light gray
        
        # Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray', origin='upper')
        plt.xticks(ticks=np.arange(-0.5, self.x_max, 1), labels=[])
        plt.yticks(ticks=np.arange(-0.5, self.y_max, 1), labels=[])
        plt.grid(color='black', linestyle='-', linewidth=1)
        plt.title('Grid Showing Permitted and Obligated States')
        plt.show()


    def learn_from_trajectories(self, trajectories):
        """
        Processes the list of trajectories to determine which states are permitted and obligated.

        Parameters:
        -----------
        trajectories : list of list of tuples (state, action)
            Each trajectory is a list of (state, action) pairs.
        """
        for trajectory in trajectories:
            for state, _ in trajectory:
                self.state_visit_counts[state] += 1

        for state, count in enumerate(self.state_visit_counts):
            if count == 1:
                self.permitted_states.add(state)
            elif count > 1:
                self.obligated_states.add(state)

    def choose_action(self, current_state):
        """
        Chooses the next action for the agent based on the current state.
        
        Parameters:
        -----------
        current_state : int
            The current state index of the agent.

        Returns:
        --------
        int
            The action (1 to 4) corresponding to the chosen action.
        """
        current_x, current_y = current_state
        act_dict = {'NORTH': 0, 'SOUTH': 1, 'EAST': 2, 'WEST': 3}
        
        obligated_actions = []
        permitted_actions = []
        action_to_state_map = {}

        for action in self.action_space:
            dx, dy = 0, 0
            if action == 'NORTH':
                dx, dy = 0, -1
            elif action == 'SOUTH':
                dx, dy = 0, 1
            elif action == 'EAST':
                dx, dy = 1, 0
            elif action == 'WEST':
                dx, dy = -1, 0
            
            next_x = current_x + dx
            next_y = current_y + dy

            if self.x_min <= next_x <= self.x_max and self.y_min <= next_y <= self.y_max:
                next_state = self.coords_to_state(next_x, next_y)
                action_to_state_map[action] = next_state
                
                if next_state in self.obligated_states:
                    obligated_actions.append((action, next_state))
                elif next_state in self.permitted_states:
                    permitted_actions.append((action, next_state))

        # Priority 1: If there are obligated actions, choose the one closest to the goal
        if obligated_actions:
            min_distance = min(abs(next_state - self.goal_state) for _, next_state in obligated_actions)
            best_actions = [action for action, next_state in obligated_actions if abs(next_state - self.goal_state) == min_distance]
            chosen_action = random.choice(best_actions)
            return act_dict[chosen_action]
        
        # Priority 2: If there are permitted actions, choose the one closest to the goal
        if permitted_actions:
            min_distance = min(abs(next_state - self.goal_state) for _, next_state in permitted_actions)
            best_actions = [action for action, next_state in permitted_actions if abs(next_state - self.goal_state) == min_distance]
            chosen_action = random.choice(best_actions)
            return act_dict[chosen_action]
        
        # Priority 3: If no obligated or permitted actions are available, choose randomly from possible actions
        random_action = random.choice(self.action_space)
        return act_dict[random_action]

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
