import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class HardcodedAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.9, epsilon=0.8, mini_epsilon=0.05, decay=0.9999, granularity=0.5):
        self.action_space = action_space
        # self.action_space = 4
        self.name = 'whole agent'
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table 
        self.norm_table = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.granularity = granularity 

        self.has_basket = False 
        self.has_milk = False 
        self.has_garlic = False 
        self.has_banana = False 
        self.checked_out = False
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        #It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = player_info['position'].copy()
        position[0] = int((1 + position[0]) / granularity)
        position[1] = int((1 + position[1]) / granularity)
        return json.dumps({
                            'position': position, 
                            'has_basket': int(self.has_basket), 
                            'has_garlic': int(self.has_garlic),
                            'has_milk': int(self.has_milk), 
                            'has_banana': int(self.has_banana),
                            'checked_out': int(self.checked_out)
                           }, sort_keys=True)
    
    def toggle_state(self, obj):
        if obj == 'basket':
            self.has_basket = not self.has_basket 
        elif obj == 'garlic':
            self.has_garlic = not self.has_garlic 
        elif obj == 'milk':
            self.has_milk = not self.has_milk
        elif obj == 'banana':
            self.has_banana = not self.has_banana
        elif obj == 'checked_out':
            self.checked_out = not self.checked_out

    def check_add(self, state):
        if self.trans(state) not in self.qtable.index:
            self.qtable.loc[self.trans(state)] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
            self.norm_table.loc[self.trans(state)] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        self.check_add(state)
        self.check_add(next_state)
        obs = self.trans(state)
        next_obs = self.trans(next_state)
        td_error = rwd + self.gamma * self.qtable.loc[next_obs, :].max() - self.qtable.loc[obs, action] 
        self.qtable.loc[obs, action] += self.alpha * td_error 

        if len(next_state['violations']) != 0:
            self.norm_table.loc[obs, action] = 1

    def choose_action(self, state):
        self.check_add(state)
        ob = self.trans(state)
        obs = eval(ob)
        # print(f'\\"checked_out\\": {obs['checked_out']}, \\"has_banana\\": {obs['has_banana']}, \\"has_basket\\": {obs['has_basket']}, \\"has_garlic\\": {obs['has_garlic']}, \\"has_milk\\": {obs['has_milk']}, \\"position\\": {obs['position']}')
        return np.argmax(self.qtable.loc[ob, :])
    
    def save_qtables(self):
        self.qtable.to_json(self.name + '.json') 
        self.norm_table.to_json(self.name + '_norm.json')

    def load_qtables(self):
        self.qtable = pd.read_json(self.name + '.json')
        self.norm_table = pd.read_json(self.name + '_norm.json')
