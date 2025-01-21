import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from basket_navi_agent import BasketNaviAgent 
from constants import obj_list, obj_pos_dict

import pickle
import argparse
import pandas as pd
import numpy as np

reach_cnt = 0
global min_x 
global min_y 
cart = False

target_list = [[3, 18], [3, 18], [3, 18]]

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def calculate_reward(previous_state, current_state, goal_x = None, goal_y = None, norms = None):

    global min_x
    global min_y

    reward_x = -1
    reward_y = -1
    reward_norms = 0
    #print("Norms: ", norms)
    if norms is not None and norms != '':
        reward_norms -= 100 * len(norms)
        reward_x -= 100 * len(norms)
        reward_y -= 100 * len(norms)
        # print("Violation")
    
    if goal_x is not None:
        dis_x = abs(current_state['observation']['players'][0]['position'][0] - goal_x)
        dis_y = abs(current_state['observation']['players'][0]['position'][1] - goal_y)
        if dis_x < min_x:
            min_x = dis_x
            reward_x = 10
        if dis_y < min_y:
            min_y = dis_y
            reward_y = 10

    if abs(current_state['observation']['players'][0]['position'][0] - goal_x) < 0.2 and abs(current_state['observation']['players'][0]['position'][1] - goal_y) < 0.2:
        reward_x = 1000
        reward_y = 1000
        global reach_cnt
        reach_cnt += 1
        print("Goal reached:", reach_cnt)

    return reward_x, reward_y, reward_norms

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_id',
        '-t',
        type=int,
    )
    args = parser.parse_args()

    target_pos = obj_pos_dict[obj_list[args.target_id]]
    target_list = [obj_list[args.target_id+i] for i in range(3)]

    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    # agent = BasketNaviAgent(action_space)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    # agent.qtable_norms = pd.read_json('qtable_norms.json')
    # agent.qtable_x = pd.read_json('qtable_x.json')
    # agent.qtable_y = pd.read_json('qtable_y.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    for target in target_list:
        agent_name = 'navigate ' + target 
        agent = BasketNaviAgent(action_space, agent_name) 
        agent.qtable_norms = pd.read_json('json_files/qtable_norms.json')
        agent.qtable_x = pd.read_json('json_files/qtable_x.json')
        agent.qtable_y = pd.read_json('json_files/qtable_y.json')
        t = obj_pos_dict[target]
        training_time = 200
        episode_length = 1000
        for i in range(training_time):
            # global min_x
            # global min_y
            min_x = 1000
            min_y = 1000
            sock_game.send(str.encode("0 RESET"))  # reset the game
            output = recv_socket_data(sock_game)
            state = json.loads(output)
            #print(state)
            #episode_length = 100 + i
            cnt = 0
            # agent.qtable_norms.to_json('qtable_norms.json')
            # agent.qtable_x.to_json('qtable_x.json')
            # agent.qtable_y.to_json('qtable_y.json')
            # create a switch that randomly chooses 0 or 1
            switch = random.randint(0, 1)
            reward_cnt = 0
            while not state['gameOver']:
                cnt += 1
                # Choose an action based on the current state
                if abs(state['observation']['players'][0]['position'][0] - t[0]) < 0.2:
                    switch = 1
                if abs(state['observation']['players'][0]['position'][1] - t[1]) < 0.2:
                    switch = 0 

                if switch == 0:
                    action_index = agent.choose_action(state, goal_x=t[0])  
                else:
                    action_index = agent.choose_action(state, goal_y=t[1])

                action = "0 " + action_commands[action_index]

                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env

                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

                reward_x, reward_y, reward_norms = calculate_reward(state, next_state, t[0], t[1], norms)  # You need to define this function
                if reward_x < 0 and reward_y < 0:
                    reward_cnt += 1
                
                if reward_cnt > 20:
                    reward_cnt = 0
                    switch = 1 - switch

                agent.learning(action_index,  state, next_state, t[0], t[1], reward_x, reward_y, reward_norms)

                # Update state
                state = next_state
                if cnt > episode_length or (abs(state['observation']['players'][0]['position'][0] - t[0]) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t[1]) < 0.2):
                    break
        # Additional code for end of episode if needed
        agent.save_qtables()

    # Close socket connection
    sock_game.close()

