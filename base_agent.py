import random
import numpy as np
import pandas as pd
import json

class BaseAgent:
    """
    A pure Q-learning agent with epsilon-greedy exploration strategy.
    """

    def __init__(self, goal, alpha=0.5, gamma=0.9, epsilon=0.8, mini_epsilon=0.05, decay=0.9999, x_max=19, y_max=24):
        """
        Initializes the BaseAgent.

        Parameters:
        -----------
        n_states : int
            The total number of possible states.
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
        self.action_space = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        self.qtable = pd.DataFrame(columns=[i for i in range(len(self.action_space))])
        self.x_min = 1
        self.x_max = x_max
        self.y_min = 2
        self.y_max = y_max

    def trans(self, state):
        """
        Extracts relevant state variables from the current state for learning.
        """
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        position = (x, y)
        return json.dumps({'position': position}, sort_keys=True)

    def check_add(self, state):
        """
        Ensures the state is in the Q-table, adding it if not present.

        Args:
            state (dict): The state to check and potentially add to the Q-table.
        """
        serialized_state = self.trans(state)
        if serialized_state not in self.qtable.index:
            self.qtable.loc[serialized_state] = pd.Series(np.zeros(len(self.action_space)), index=[i for i in range(len(self.action_space))])

    def learning(self, action, state, next_state, reward):
        """
        Updates Q-table after the agent receives a reward.

        Args:
            action: The agent's current action.
            state: The current agent state.
            next_state: The state obtained after applying the action in the current state.
            reward: The reward received after performing the action.
        """
        self.check_add(state)
        self.check_add(next_state)

        q_sa = self.qtable.loc[self.trans(state), action]
        max_next_q_sa = self.qtable.loc[self.trans(next_state), :].max()
        new_q_sa = q_sa + self.alpha * (reward + self.gamma * max_next_q_sa - q_sa)
        self.qtable.loc[self.trans(state), action] = new_q_sa

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (dict): The current state of the agent.

        Returns:
            int: The chosen action index.
        """
        self.check_add(state)

        if np.random.uniform(0, 1) < self.epsilon:
            # Explore
            action = random.choice(range(len(self.action_space)))
        else:
            # Exploit
            q_values = self.qtable.loc[self.trans(state)]
            action = q_values.idxmax()

        # Decay epsilon
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay

        return action

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

    def save_qtable(self):
        """Save the Q-table to a JSON file."""
        self.qtable.to_json('qtable_new.json')
