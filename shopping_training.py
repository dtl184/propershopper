import json
import random
import socket
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from env import SupermarketEnv
from utils import recv_socket_data
from base_agent import BaseAgent
from planner import HyperPlanner
from get_obj_agent import GetObjAgent
from socket_env import obj_pos_dict, obj_state_index_dict
from constants import *
import pickle
import pandas as pd
import argparse
from termcolor import colored
import logging
import os
def euclidean_distance(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

inventory = {}
payed_items = {}

logging.basicConfig(
    filename='shopping.log',
    filemode='w',  # Overwrites the log file each time. Use 'a' to append.
    level=logging.DEBUG,  # Capture all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Ensures previous configurations don’t interfere
)

def init_inventory():
    for i in obj_list:
        inventory[i] = 0
        payed_items[i] = 0

def calculate_reward(previous_state, current_state, target, task, index):
    reached = False
    global min_distance

    if task == 'navigate':
        prev_player = previous_state['observation']['players'][0]
        curr_player = current_state['observation']['players'][0]
        target_pos = obj_pos_dict[target]
        prev_dist = euclidean_distance(prev_player['position'], target_pos)
        curr_dist = euclidean_distance(curr_player['position'], target_pos)
        dist_gain = prev_dist - curr_dist

        if curr_dist < min_distance:
            min_distance = curr_dist
            return reached, 10

        if index in obj_state_index_dict[target]:
            reached = True
            return reached, 100
        
        return reached, -1

    if task == 'get':
        if target == 'basket':
            if len(current_state['observation']['baskets']) != 0:
                reached = True
                return reached, 100
        else:
            if len(current_state['observation']['baskets']) == 0:
                return reached, -5
            basket_contents = current_state['observation']['baskets'][0]['contents']
            if target in basket_contents:
                target_id = basket_contents.index(target)
                if current_state['observation']['baskets'][0]['contents_quant'][target_id] > 0:
                    reached = True
                    return reached, 100

        return reached, -1
    
    if task == 'pay':
        if len(current_state['observation']['baskets']) == 0:
            return reached, -5
        if len(current_state['observation']['baskets'][0]['purchased_contents']) != 0:
            reached = True
            return reached, 100
        return reached, 0

def plot_results(subgoals_reached, plan_completion_rates):
    plt.figure(figsize=(10, 5))
    plt.plot(subgoals_reached, label="Avg Subgoals Reached per Episode", marker='o')
    plt.xlabel("Experiment")
    plt.ylabel("Avg Subgoals Reached")
    plt.title("Subgoal Completion Across Experiments")
    plt.legend()
    plt.grid()
    plt.savefig("subgoals_reached.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(plan_completion_rates, label="Avg Plan Completion Rate", marker='o', color='r')
    plt.xlabel("Experiment")
    plt.ylabel("Plan Completion Rate")
    plt.title("Plan Completion Rate Across Experiments")
    plt.legend()
    plt.grid()
    plt.savefig("plan_completion_rate.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument('--num_experiments', type=int, default=10, help="Number of experiments to run")
    parser.add_argument('--num_episodes', type=int, default=100000, help="Number of episodes per experiment")
    parser.add_argument('--subgoal_time', type=int, default=300, help="Number of time steps per subgoal")
    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    all_subgoals_reached = []
    all_plan_completion_rates = []
    
    for experiment in range(args.num_experiments):
        logging.info(f'Starting experiment {experiment}\n')
        planner = HyperPlanner(obj_list, agent_class=BaseAgent)
        experiment_subgoals = []
        experiment_plan_completions = []
        
        for episode in range(args.num_episodes):
            planner.reset()
            sock_game.send(str.encode("0 RESET"))
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            planner.parse_shopping_list(state)     

            subgoals_completed = 0
            global min_distance
            min_distance = 1000
            while not planner.plan_finished():
                agent = planner.get_agent()
                if agent is None:
                    break
                planner.change_qtable()  # Ensure correct Q-table is used

                task_name = planner.get_task().replace(" ", "_")
                
                # qtable_path = os.path.join("hybrid_shopping_qtables", f"qtable_{task_name}.json")
                # if os.path.exists(qtable_path):
                #     agent.load_qtable(qtable_path)
                # else:
                #     continue
                
                logging.info(f"Experiment {experiment} Episode {episode} Current Task: {planner.get_task()}")
                steps = 0
                goal_reached = False
                
                while steps < 100 and not goal_reached:
                    steps += 1
                    action_index = agent.choose_action(state)
                    action = "0 " + agent.action_space[action_index]
                    num = 6 if action_index <= 3 else 1

                    for _ in range(num):  
                        action = "0 " + agent.action_space[action_index]
                        sock_game.send(str.encode(action))
                        next_state = recv_socket_data(sock_game)
                        next_state = json.loads(next_state) if next_state else None

                    goal_reached, reward = calculate_reward(state,
                                                            next_state,
                                                            target=planner.get_task().split()[1],
                                                            task=planner.get_task().split()[0],
                                                            index=agent.state_index(state))
                    agent.learning(action_index, state, next_state, reward)
                    state = next_state
                
                # if the subgoal was just reached, continue plan. Otherwise restart episode
                if goal_reached:
                    #logging.info('Subgoal reached!')
                    subgoals_completed += 1
                    planner.update()
                else:
                    break 

                if planner.plan_finished():
                    logging.info('Plan completed!\n')

                
            
            experiment_subgoals.append(subgoals_completed)
            experiment_plan_completions.append(int(planner.plan_finished()))
        
        all_subgoals_reached.append(np.mean(experiment_subgoals))
        all_plan_completion_rates.append(np.mean(experiment_plan_completions))

    #planner.save_qtables()
    
    plot_results(all_subgoals_reached, all_plan_completion_rates)
    sock_game.close()
