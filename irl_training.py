import argparse
import socket
import json
from irl_agent import IRLAgent
import numpy as np
from utils import recv_socket_data
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg") 
import torch
from ltl_agent import LTLAgent
import ast
import pandas as pd
import math
from base_agent import BaseAgent
def load_trajectories(file_name):
    """
    Reads trajectories from a text file and stores them as a list of lists.

    Parameters:
        file_name (str): The name of the file containing the trajectories.

    Returns:
        list: A list where each trajectory is a separate list of tuples.
    """
    trajectories = []

    with open(file_name, "r") as file:
        for line in file:
            # Strip whitespace and newlines
            line = line.strip()
            if line:  # Ensure the line is not empty
                # Evaluate the line as a Python list
                trajectory = eval(line)
                trajectories.append(trajectory)

    return trajectories

def save_reward(reward, filename="reward.txt"):
    with open(filename, "w") as file:
        for value in reward:
            file.write(f"{value}\n")
    print(f"Reward saved to {filename}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help="Port to connect to the environment"
    )
    args = parser.parse_args()

    trajectories = load_trajectories('trajectories.txt')

   # trajectories = pad_trajectories("trajectories.txt")

    agent = IRLAgent(n_states=437, trajectories=trajectories)
    #agent.visualize_valid_states()


    with open("learned_reward.txt", "r") as file:
        data = file.read().strip()  # Read and remove extra spaces/newlines

    # # Safely evaluate the array string
    lst = ast.literal_eval(data)

    lst = np.array(lst)

    # Set the reward in the agent
    agent.set_reward(lst)

    # with open("state.json", "r") as file:
    #     state = json.load(file)

    #agent.generate_transition_matrix()

    #agent.learn_reward()

    #save_reward(agent.reward)

    #agent.visualize_reward()

   # agent = BaseAgent(goal = (3, 18), epsilon=0)
   # agent.qtable = pd.read_json('qtable_base.json')
    agent = LTLAgent(n_states=437, goal = (3, 18), epsilon=0, filename='base_trajectories.txt')
    agent.count_transition_states()



    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    
    for episode in range(100):
        sock_game.send(str.encode("0 RESET"))  
        state = recv_socket_data(sock_game)
        state = json.loads(state)

        for i in range(100):

            # Agent chooses the next action based on the state

            action_index = agent.choose_action(state)

            for _ in range(6):
                action = "0 " + agent.action_space[action_index]

                # Send the action to the environment
                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
                next_state = json.loads(next_state)
            
            x, y = state['observation']['players'][0]['position']
            if math.sqrt((agent.goal[0] - x) ** 2 + (agent.goal[1] - y) ** 2) <= 1:
                break



            # Update the state
            state = next_state

    sock_game.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
