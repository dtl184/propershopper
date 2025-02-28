# Scripts for training basket agents

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class 
from planner import HyperPlanner
from get_obj_agent import GetObjAgent
from constants import *
import pickle
import pandas as pd
import argparse
from termcolor import colored

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5 

inventory = {} 
payed_items = {}

def init_inventory():
    for i in obj_list:
        inventory[i] = 0 
        payed_items[i] = 0

def calculate_reward(previous_state, current_state, target, task):
    if task == 'navigate': 
        return None 
    norm_penalty = -50
    target_location = obj_pos_dict[target] 

    if task == 'get':
        if len(current_state['violations']) != 0:
            #print(f'violated norm: {current_state['violations']}')
            return norm_penalty
        prev_player = previous_state['observation']['players'][0]
        curr_player = current_state['observation']['players'][0]

        prev_dist = euclidean_distance(prev_player['position'], target_location) 
        curr_dist = euclidean_distance(curr_player['position'], target_location) 

        dist_gain = prev_dist - curr_dist 
        if target == 'basket': 
            if len(current_state['observation']['carts']) != 0:
                return -600
            if len(current_state['observation']['baskets']) != 0:
                return 100 
        else:
            # the agent is guaranteed to have a basket by this point
            basket_contents = current_state['observation']['baskets'][0]['contents'] 
            for i, b in enumerate(basket_contents):
                basket_contents[i] = b.replace(' ', '_')
            if target in basket_contents:
                target_id = basket_contents.index(target) 
                cur_num = current_state['observation']['baskets'][0]['contents_quant'][target_id] 
                # print(cur_num)
                if cur_num > inventory[target]:
                    inventory[target] = cur_num
                    return 100

        return dist_gain-0.0001
    
    if task == 'pay':
        if len(current_state['violations']) != 0:
            #print(f'violated norm: {current_state['violations']}')
            return norm_penalty 
        if len(current_state['observation']['baskets'][0]['purchased_contents']) != 0: 
            return 100 # Basket agent check out is guaranteed to be successful
        return 0 # Sparse reward since checkout position is fixed and extremely easy 

special_food_list = ['prepared_foods', 'fresh_fish']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument('--episode_length', type=int, default=100, help="Maximum steps per episode")
    parser.add_argument('--num_experiments', type=int, default=10, help="Number of experiments to run")
    parser.add_argument('--num_episodes', type=int, default=1500, help="Number of episodes per experiment")
    args = parser.parse_args()

    planner = HyperPlanner(obj_list)
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT)) 

    training_time = 100
    episode_length = 800
    for i in range(1, training_time+1):
        # start = False 
        planner.reset()

        sock_game.send(str.encode("0 RESET"))  # reset the game

        state = recv_socket_data(sock_game)
        state = json.loads(state)
        planner.parse_shopping_list(state)
        agent = planner.get_agent()

        print(f'Current Plan: {planner.plan}')
        init_inventory()
        while not planner.plan_finished(): 
            done = False
            cnt = 0
            current_task = planner.get_task() 
            print(f'Current Task: {current_task}')
            while not done:
                cnt += 1
                action_index, finish = agent.choose_action(state)
                action = "0 " + agent.action_commands[action_index] 
                # print(action)
                sock_game.send(str.encode(action))  # send action to env 

                next_state = recv_socket_data(sock_game)  # get observation from env
                # fill with random state if state is empty
                next_state = json.loads(next_state) 
                if action == '0 INTERACT' and current_task.split()[0] == 'get':
                    sock_game.send(str.encode(action))  # remove dialogue box 
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                    if current_task.split()[1] in special_food_list:
                        sock_game.send(str.encode(action))  # remove dialogue box 
                        next_state = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(next_state) 

                if action == '0 INTERACT' and current_task.split()[0] == 'pay':
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                    sock_game.send(str.encode('0 INTERACT'))  # remove dialogue box 
                    next_state = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(next_state) 
                # if next_state['observation']['players'][0]['position'][0] < 0 and abs(next_state['observation']['players'][0]['position'][1] - exit_pos[1]) < 1:
                #     next_state['gameOver'] = True

                # Define the reward based on the state and next_state
                reward = calculate_reward(state, next_state, target=current_task.split()[1], task=current_task.split()[0])  # You need to define this function 
                if reward == 100 or finish:
                    print("Success!") 
                    done = True 
                    # print(state['observation']['players'][0])

                agent.learning(action_index, reward, state, next_state)

                # Update state
                state = next_state

                if cnt > episode_length:
                    print("\nepisode count exceeded\n")
                    break
            agent.save_qtables()
            if not done:
                break # mission failed. Restart everything 
            else:
                planner.update() 
                agent = planner.get_agent() 
                if agent is None:
                    print(colored('Whole task succeeded', 'green'))
    sock_game.close()

