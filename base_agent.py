import random
import numpy as np
import json

class BaseAgent:
    """
    Optimized Q-learning agent with an epsilon-greedy exploration strategy.
    """

    def __init__(self, goal=None, alpha=0.5, gamma=0.9, epsilon=0.3, mini_epsilon=0.05, decay=0.9999, x_max=19, y_max=24):
        """
        Initializes the BaseAgent.

        Parameters:
        -----------
        goal : tuple
            The goal position (x, y).
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon : float
            Initial exploration probability.
        mini_epsilon : float
            Minimum exploration probability.
        decay : float
            Epsilon decay factor per step.
        x_max : int
            The width of the grid (number of columns).
        y_max : int
            The height of the grid (number of rows).
        """
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.action_space = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'INTERACT']
        self.num_actions = len(self.action_space)
        self.qtable = {}  # ✅ Use dictionary instead of Pandas (faster lookups)
        self.x_min, self.x_max = 1, x_max
        self.y_min, self.y_max = 2, y_max

    def state_index(self, state):
        """
        Computes a unique integer index for a given state.

        Returns:
            int: The computed state index.
        """
        x, y = state['observation']['players'][0]['position']
        total_x_values = self.x_max - self.x_min + 1
        index = (round(y) - self.y_min) * total_x_values + (round(x) - self.x_min)
        return max(index, 0)

    def check_add(self, state_index):
        """
        Ensures the state is in the Q-table, adding it if not present.

        Args:
            state_index (int): The computed state index.
        """
        if state_index not in self.qtable:
            self.qtable[state_index] = np.zeros(self.num_actions, dtype=np.float32)  # ✅ Use NumPy for fast operations

    def learning(self, action, state, next_state, reward):
        """
        Updates Q-table after the agent receives a reward.

        Args:
            action (int): The agent's current action.
            state (dict): The current agent state.
            next_state (dict): The next agent state.
            reward (float): The reward received after performing the action.
        """
        state_idx = self.state_index(state)
        next_state_idx = self.state_index(next_state)

        self.check_add(state_idx)
        self.check_add(next_state_idx)

        q_sa = self.qtable[state_idx][action]
        max_next_q_sa = np.max(self.qtable[next_state_idx])
        self.qtable[state_idx][action] += self.alpha * (reward + self.gamma * max_next_q_sa - q_sa)

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (dict): The current state of the agent.

        Returns:
            int: The chosen action index.
        """
        state_idx = self.state_index(state)
        self.check_add(state_idx)

        if np.random.uniform(0, 1) <= self.epsilon:
            action = random.randint(0, self.num_actions - 1)  # ✅ Faster than `random.choice(range(len(self.action_space)))`
        else:
            q_values = self.qtable[state_idx]
            if np.all(q_values == 0):  # If all Q-values are zero, pick a random action
                action = random.choice(range(len(self.action_space)))
            else:
                action = np.argmax(q_values) # ✅ Faster than Pandas `.idxmax()`

        # Decay epsilon
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay

        return action
    
    def reset_qtable(self):
        self.qtable = {}

    def save_qtable(self):
        """Save the Q-table as a JSON file, converting NumPy arrays to lists."""
        qtable_serializable = {state: q_values.tolist() for state, q_values in self.qtable.items()}  # ✅ Convert arrays to lists
        with open('qtable_ltl.json', 'w') as f:
            json.dump(qtable_serializable, f)  # ✅ Now JSON-serializable

    def load_qtable(self, filename):
        """Load the Q-table from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.qtable = {int(k): np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
        except FileNotFoundError:
            print("No existing Q-table found. Starting fresh.")
