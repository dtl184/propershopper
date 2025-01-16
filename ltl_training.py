import argparse
import socket
import json
from irl_agent import IRLAgent
import numpy as np
from utils import recv_socket_data
import matplotlib.pyplot as plt

from ltl_agent import LTLAgent

def calculate_reward(next_state, goal=None):
    global min_distance  # Track the minimum distance to the goal
    global reach_cnt  # Count how many times the goal is reached

    # Default reward
    reward = -1
    
    if goal is not None:
        
        agent_position = next_state['observation']['players'][0]['position']
        
        # calculate the agent's distance to the goal from the next state
        distance = ((agent_position[0] - goal[0]) ** 2 + (agent_position[1] - goal[1]) ** 2) ** 0.5
        
        # reward the agent more the closer it gets to the goal
        if distance < min_distance:
            min_distance = distance
            reward = 10
        
        # Check if the goal has been reached
        if distance < 0.2:  # Tolerance for reaching the goal
            reward = 1000
            reach_cnt += 1
            print("Goal reached:", reach_cnt)

    return reward




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

    agent = LTLAgent(n_states=437, goal=(3.5, 17.0))
    agent.learn_from_trajectories(trajectories)

    agent.visualize()


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
        reward = calculate_reward(next_state, goal)

        # Update the state
        state = next_state

    sock_game.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
