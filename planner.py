import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

from planning_agent import PlanningAgent
from get_obj_agent import GetObjAgent
from checkout_agent import CheckoutAgent

goal_list = ['']

class HyperPlanner:
    '''
    Planner for selecting agents for different goals.
    '''
    # here are some default parameters, you can use different ones
    def __init__(self, obj_list): 

        skill_set = [
            'navigate ' + item for item in obj_list
        ] 

        self.agents = [PlanningAgent(item, item.split()[1]) for item in skill_set if item.split()[0] == 'navigate'] 

        skill_set.extend(['get ' + item for item in obj_list])
        self.agents.extend([GetObjAgent(item, item.split()[1]) for item in skill_set if item.split()[0] == 'get']) 

        skill_set.extend(['pay checkout'])
        self.agents.append(CheckoutAgent('pay checkout', 'checkout'))

        for a in self.agents:
            a.load_qtables()

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
        # self.plan.append('toggle')
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
                    # self.plan.append('putincart') 
                self.plan.append('get cart') 

        self.plan.append('navigate checkout') 
        if not self.use_cart:
            self.plan.append('pay checkout') 
        # else:
        #     raise NotImplementedError('Cart payment is not implemented yet.')
        self.plan.append('navigate exit') 

    def get_task(self):
        return self.plan[self.current_goal_id] if self.current_goal_id < len(self.plan) else None
    
    def update(self):
        self.current_goal_id += 1

    def get_agent(self):
        goal = self.plan[self.current_goal_id] if self.current_goal_id < len(self.plan) else None
        for agent in self.agents:
            if agent.name == goal:
                return agent 
        return None 
    
    def plan_finished(self):
        return self.current_goal_id == len(self.plan)
