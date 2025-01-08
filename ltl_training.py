import argparse
import socket
import json
from irl_agent import IRLAgent
import numpy as np
from utils import recv_socket_data
import matplotlib.pyplot as plt

from ltl_agent import LTLAgent

def calculate_rewardnext_state, goal_x = None, goal_y = None, norms = None):

    global min_x
    global min_y

    reward_x = -1
    reward_y = -1
    
    if goal_x is not None:
        dis_x = abs(next_state['observation']['players'][0]['position'][0] - goal_x)
        dis_y = abs(next_state['observation']['players'][0]['position'][1] - goal_y)
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

    agent = LTLAgent(n_states=437)

    agent.learn_from_trajectories(trajectories)

    # agent.visualize()


    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    sock_game.send(str.encode("0 RESET"))  
    state = recv_socket_data(sock_game)
    state = json.loads(state)

    done = False
    while not done:
        # Agent chooses the next action based on the state
        x = int(round(state['observation']['players'][0]['position'][0]))
        y = int(round(state['observation']['players'][0]['position'][1]))
        action_index = agent.choose_action((x, y))
        action = "0 " + agent.action_space[action_index]
        print(f"Sending action: {action}")

        # Send the action to the environment
        sock_game.send(str.encode(action))
        next_state = recv_socket_data(sock_game)
        next_state = json.loads(next_state)

        # Calculate reward here
        reward_x, reward_y = calculate_reward

        # Update the state
        state = next_state

    sock_game.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
