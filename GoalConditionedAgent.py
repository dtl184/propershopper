import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

from navigation_agent import NavigationAgent

goal_list = ['']

class GoalConditionedAgent:
    '''
    Agent selects actions based on goal.
    '''
    # here are some default parameters, you can use different ones
    def __init__(self, obj_list, alpha=0.5, gamma=0.9, epsilon=0.8, mini_epsilon=0.05, decay=0.9999): 
        self.action_space = 4
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time 

        skill_set = [
            'navigate ' + item for item in obj_list
        ] 
        skill_set.extend(['navigate cart ' + item for item in obj_list])

        self.qtable = {
            item: pd.DataFrame(columns=[i for i in range(4)]) if item.split()[0] == 'navigate' else pd.DataFrame(columns=[i for i in range(4)]) for item in skill_set
        } 

        self.navigation_agents = [NavigationAgent(item) for item in skill_set if item.split()[0] == 'navigate']
        self.norm_table = pd.DataFrame(columns=[i for i in range(self.action_space)])

        self.current_goal_id = None
        self.cart_position = None
        self.hardcodded_actions = ['drop cart', 'get cart'] # Actions to be design with hard code 
        self.plan = None 
        self.use_cart = False


    def parse_shopping_list(self, state):
        player_info = state['observation']['players'][0] 
        shopping_list = player_info['shopping_list'] 

        self.use_cart = np.sum(player_info['list_quant']) > 6
        self.plan = []
        self.plan.append('navigate cart' if self.use_cart else 'navigate basket') 
        self.plan.append('get cart' if self.use_cart else 'get basket')
        for id, item in enumerate(shopping_list):
            self.plan.append('navigate ' + item) 
            for _ in range(player_info['list_quant'][id]):
                self.plan.append('get ' + item) 

        self.plan.append('navigate checkout') 
        self.plan.append('checkout')
        self.plan.append('navigate leave')
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        #It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = player_info['position'].copy()
        position[0] = int((1 + position[0]) / granularity)
        position[1] = int((1 + position[1]) / granularity)
        has_cart = player_info['cart']
        return json.dumps({'position': position}, sort_keys=True)
    
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
            # print('violated')
            self.norm_table.loc[obs, action] = 1

        # self.epsilon = max(self.epsilon, self.mini_epsilon)

    def choose_action(self, state):
        self.check_add(state)
        ob = self.trans(state)
        max_v = self.qtable.loc[ob, :].max()
        candidates = [i for i in range(self.action_space) if self.norm_table.loc[ob, i] != 1]
        if 'a' in candidates:
            print(candidates)
        if len(candidates) == 0: 
            candidates = range(self.action_space)
        greedy_candidates = [i for i in candidates if self.qtable.loc[ob, i] == max_v] 
        if len(greedy_candidates) == 0:
            greedy_candidates = candidates
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay
        return np.random.choice(candidates) if np.random.rand() < self.epsilon else np.random.choice(greedy_candidates)
    
    def load_qtables(self):

        pass # TODO: load qtables from json 

    def save_qtables(self):
        pass