import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.9, mini_epsilon=0.05, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(data=np.zeros((20000, 7)), columns=[i for i in range(self.action_space)])  # generate the initial table 
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn

        obs = []
        # x range: (0, 20)
        # y range: (0, 25)
        # obs[0: 2]: Player position
        obs.append(int((state['observation']['players'][0]['position'][0] + 1) / granularity))
        obs.append(int((state['observation']['players'][0]['position'][1] + 1) / granularity))
        # obs[2]: Player direction
        obs.append(int(state['observation']['players'][0]['direction']) )
        # obs[3]: Whether player is holding a cart 
        obs.append(state['observation']['players'][0]['curr_cart'] != -1 )

        obs = obs[0] + obs[1] * 40 + obs[2] * 1600 + obs[3] * 6400

        return obs
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        obs = self.trans(state)
        next_obs = self.trans(next_state)
        td_error = rwd + self.gamma * np.max(self.qtable.loc[next_obs]) - self.qtable.loc[obs][action] 
        self.qtable.loc[obs, action] += self.alpha * td_error
        self.epsilon *= self.decay 
        self.epsilon = max(self.epsilon, self.mini_epsilon)

    def choose_action(self, state):
        # implement the action selection for the fully trained agent 
    #    print(type(state))
    #    print('action space', self.action_space)
    #    print('state keys: ', state.keys())
    #    print('command result: ', state['command_result'])
    #    print('observation keys: ', state['observation'].keys())
    #    print('observation[carts]', state['observation']['carts'])
    #    print('observation[baskets]', state['observation']['baskets'])
    #    print('observation[players]: ', state['observation']['players'])
    #    print('observation[carts]: ', state['observation']['carts'])
       ob = self.trans(state)
       max_v = np.max(self.qtable.loc[ob])
       candidates = [i for i in range(self.action_space) if self.qtable.loc[ob, i] == max_v]
       return np.random.choice(candidates) if np.random.rand() > self.epsilon else np.random.randint(self.action_space)
