import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
from base_agent import BaseAgent
from get_obj_agent import GetObjAgent
from checkout_agent import CheckoutAgent


class HyperPlanner:
    '''
    Planner for selecting agents for different goals.
    '''
    def __init__(self, obj_list, agent_class=BaseAgent): 
        self.agent_class = agent_class
        self.agent = agent_class(goal=None)
        self.qtables = {}  # Dictionary to store Q-tables per task in memory
        self.current_goal_id = 0
        self.plan = None 
        self.use_cart = False 

    def reset(self):
        self.current_goal_id = 0
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
            item = item.lower().replace(' ', '_')
            self.plan.append('navigate ' + item)
            if not self.use_cart:
                for _ in range(player_info['list_quant'][id]):
                    self.plan.append('get ' + item) 
            else:
                self.plan.append('drop cart')
                for _ in range(player_info['list_quant'][id]):
                    self.plan.append('get ' + item)
                self.plan.append('get cart') 

        self.plan.append('navigate checkout') 
        if not self.use_cart:
            self.plan.append('pay checkout') 
        self.plan.append('navigate exit') 

    def get_task(self):
        return self.plan[self.current_goal_id] if self.current_goal_id < len(self.plan) else None
    
    def update(self):
        self.change_qtable()
        self.current_goal_id += 1

    def get_agent(self):
        return self.agent
    
    def plan_finished(self):
        return self.current_goal_id == len(self.plan)
    
    def change_qtable(self):
        current_task = self.get_task()
        if current_task:
            task, target = current_task.split()
            qtable_key = f"{task}_{target}"
            if qtable_key not in self.qtables:
                self.qtables[qtable_key] = {}  # Store Q-table in memory
            self.agent.qtable = self.qtables[qtable_key]
