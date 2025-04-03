import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

from env_files.constants import *


class PlanningAgent:
    '''
    Path planning agent for fast navigation
    '''
    def __init__(self, name, target):
        self.name = name 
        self.target = target
        self.action_directions = [[0, -1], [0, 1], [1, 0], [-1, 0]] 
        self.action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        self.map = None
    
    def build_map(self, granularity=0.2):
        self.map = np.zeros((int(max_x/granularity), int(max_y/granularity)))

        # set up horizontal blocks
        for block in horizontal_blocks:
            # print(block)
            # print(block[1])
            x1, y1 = int(block[0][0]/granularity), int(block[0][1]/granularity)
            x2, y2 = int(block[1][0]/granularity), int(block[1][1]/granularity)
            assert y1 == y2
            # if block[0][1] == 16.8:
            #     for i in range(x1, x2):
            #         # print(i, y1)
            self.map[x1:x2, y1] = 99999999

        # set up vertical blocks
        for block in vertical_blocks:
            x1, y1 = int(block[0][0]/granularity), int(block[0][1]/granularity)
            x2, y2 = int(block[1][0]/granularity), int(block[1][1]/granularity) 
            assert x1 == x2
            self.map[x1, y1:y2] = 99999999 
        
        start = obj_pos_dict[self.target]
        start_x, start_y = int(start[0]/granularity), int(start[1]/granularity) 
        self.map[start_x, start_y] = 0
        queue = [(start_x, start_y, 0)] 

        while len(queue) != 0:
            x, y, d = queue.pop(0) 
            if self.map[x, y] != 0:
                continue
            self.map[x, y] = d
            for direction in self.action_directions:
                x_, y_ = x + direction[0], y + direction[1]
                if x_ >= 0 and x_ < self.map.shape[0] and y_ >= 0 and y_ < self.map.shape[1]:
                    if self.map[x_, y_] == 0:
                        queue.append((x_, y_, d + 1)) 
        
        for x in range(self.map.shape[0]):
            for y in range(self.map.shape[1]):
                if self.map[x, y] == 0 and (x, y) != (start_x, start_y):
                    self.map[x, y] = 99999999  

    def learning(self, action, rwd, state, next_state):
        pass

    def save_qtables(self):
        pass 

    def load_qtables(self):
        self.build_map()

    def choose_action(self, state):
        player_info = state['observation']['players'][0]
        position = player_info['position'] 

        cur_position = [int(position[0] / 0.2), int(position[1] / 0.2)] 
        values = [0, 0, 0, 0]
        for i, d in enumerate(self.action_directions):
            x, y = cur_position[0] + d[0], cur_position[1] + d[1]
            values[i] = self.map[x, y]
        action = np.argmin(values) 
        if values[action] == 99999999:
            return np.random.randint(4), False
        return action, values[action] < 2