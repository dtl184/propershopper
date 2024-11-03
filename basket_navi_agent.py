import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

from constants import obj_pos_dict

class BasketNaviAgent:
    def __init__(self, action_space, name, alpha=0.5, gamma=0.9, temp=1, epsilon=0.3, mini_epsilon=0.1, decay=0.9999):
        self.action_space = action_space 
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.qtable_norms = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.qtable_x = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.qtable_y = pd.DataFrame(columns=[i for i in range(self.action_space)]) 
    
    def trans(self, state, goal_x=None, goal_y=None, granularity=0.15):
        # You should design a function to transform the huge state into a learnable state for the agent
        #It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = [int(player_info['position'][0] / granularity) * granularity, int(player_info['position'][1] / granularity) * granularity]
        # if self.target[0] is not None:
        #     return  json.dumps({'position': position, 'goal_x': self.target[0]}, sort_keys=True)
        # if self.target[1] is not None:
        #     return json.dumps({'position': position, 'goal_y': self.target[1]}, sort_keys=True)
        return json.dumps({'position': position}, sort_keys=True)


    
    def check_add(self, state, goal_x = None, goal_y = None):
        #print(goal_x, goal_y)
        serialized_state = self.trans(state)
        if serialized_state not in self.qtable_norms.index:
            self.qtable_norms.loc[serialized_state] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        if goal_x is not None:
            serialized_state_x = self.trans(state, goal_x=goal_x)
            if serialized_state_x not in self.qtable_x.index:
                self.qtable_x.loc[serialized_state_x] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        if goal_y is not None:
            serialized_state_y = self.trans(state, goal_y=goal_y)
            if serialized_state_y not in self.qtable_y.index:
                self.qtable_y.loc[serialized_state_y] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])



    def learning(self, action, state, next_state,  goal_x = None, goal_y = None, reward_x = 0, reward_y =0 , reward_norms = 0):

        self.check_add(state)
        self.check_add(next_state)
        if goal_x is not None:
            self.check_add(state, goal_x = goal_x)
            self.check_add(next_state, goal_x = goal_x)
        if goal_y is not None:
            self.check_add(state, goal_y = goal_y)
            self.check_add(next_state, goal_y = goal_y)
        
        q_sa = self.qtable_norms.loc[self.trans(state), action]
        max_next_q_sa = self.qtable_norms.loc[self.trans(next_state), :].max()
        new_q_sa = q_sa + self.alpha * (reward_norms + self.gamma * max_next_q_sa - q_sa)
        self.qtable_norms.loc[self.trans(state), action] = new_q_sa

        if reward_x != 0:
            
            q_sa = self.qtable_x.loc[self.trans(state, goal_x=goal_x), action]
            max_next_q_sa = self.qtable_x.loc[self.trans(next_state, goal_x=goal_x), :].max()
            new_q_sa = q_sa + self.alpha * (reward_x + self.gamma * max_next_q_sa - q_sa)
            self.qtable_x.loc[self.trans(state, goal_x=goal_x), action] = new_q_sa

        if reward_y != 0:
            q_sa = self.qtable_y.loc[self.trans(state, goal_y=goal_y), action]
            max_next_q_sa = self.qtable_y.loc[self.trans(next_state, goal_y=goal_y), :].max()
            new_q_sa = q_sa + self.alpha * (reward_y + self.gamma * max_next_q_sa - q_sa)
            self.qtable_y.loc[self.trans(state, goal_y=goal_y), action] = new_q_sa
    


    def action_prob(self, state, goal_x = None, goal_y = None):
        self.check_add(state)

        self.check_add(state, goal_x=goal_x)

        self.check_add(state, goal_y=goal_y)

        p = np.random.uniform(0, 1)
        self.epsilon *= 0.99
        if p <= self.epsilon:
            return np.array([1/self.action_space for i in range(self.action_space)])
        else:
            if goal_x is None and goal_y is None:
                prob = F.softmax(torch.tensor(self.qtable_norms.loc[self.trans(state)].to_list(), dtype=torch.float), dim=0).detach().numpy()
            elif goal_x is None:
                #combine prob from three qtables
                prob = F.softmax(torch.tensor(self.qtable_y.loc[self.trans(state, goal_y=goal_y)].to_list(), dtype=torch.float), dim=0).detach().numpy()
                prob = prob / prob.sum()
            elif goal_y is None:
                prob = F.softmax(torch.tensor(self.qtable_x.loc[self.trans(state, goal_x=goal_x)].to_list(), dtype=torch.float), dim=0).detach().numpy()
                prob = prob / prob.sum()
            else:
                prob = ( F.softmax(torch.tensor(self.qtable_x.loc[self.trans(state, goal_x=goal_x)].to_list(), dtype=torch.float), dim=0).detach().numpy() 
                      + F.softmax(torch.tensor(self.qtable_y.loc[self.trans(state, goal_y=goal_y)].to_list(), dtype=torch.float), dim=0).detach().numpy() )
                prob = prob / prob.sum()
            return prob

    def choose_action(self, state, goal_x = None, goal_y = None):
        self.check_add(state)
        p = np.random.uniform(0, 1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            #prob = F.softmax(torch.tensor(self.qtable_norms.loc[self.trans(state)].to_list()), dim=0).detach().numpy()
            prob = self.action_prob(state, goal_x=goal_x, goal_y=goal_y)
            action = np.random.choice([i for i in range(self.action_space)], p=prob)
            cnt = 0
            while self.qtable_norms.loc[self.trans(state)].to_list()[action] < 0:
                cnt += 1
                action = np.random.choice([i for i in range(self.action_space)], p=prob)
                if cnt > 10:
                    action = np.random.choice([i for i in range(self.action_space)])
           # print("prob:", prob)
            return action 
        
    def save_qtables(self):
        self.qtable_x.to_json(self.name + '_x.json')
        self.qtable_y.to_json(self.name + '_y.json')
        self.qtable_norms.to_json(self.name + '_norms.json')

    def load_qtables(self):
        self.qtable_x = pd.read_json(self.name + '_x.json')
        self.qtable_y = pd.read_json(self.name + '_y.json')
        self.qtable_norms = pd.read_json(self.name + '_norms.json')
