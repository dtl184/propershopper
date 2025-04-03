import argparse
import socket
import json
import numpy as np
import pandas as pd
from base_agent import BaseAgent
from env_files.utils import recv_socket_data
import matplotlib.pyplot as plt
import logging
min_distance = float('inf')

logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

def calculate_reward(state, next_state, goal=None, agent=None):
    """
    Calculate reward based on the distance to the goal.
    """
    global min_distance  # Track the minimum distance to the goal
    reward = -1
    reached = False

    if goal is not None:
        curr_position = state['observation']['players'][0]['position']
        next_position = next_state['observation']['players'][0]['position']

        curr_distance = ((round(curr_position[0]) - goal[0]) ** 2 + (round(curr_position[1]) - goal[1]) ** 2) ** 0.5
        next_distance = ((round(next_position[0]) - goal[0]) ** 2 + (round(next_position[1]) - goal[1]) ** 2) ** 0.5

        reward = 0

        if next_distance < min_distance:
            min_distance = next_distance
            reward = 10

        if agent.state_index(state) in [306, 307, 325, 326]:
            reward = 1000
            reached = True

    return reached, reward - 0.05


def save_experiment_rewards(experiment_rewards, filename="experiment_rewards.csv"):
    """
    Save experiment rewards to a CSV file.
    """
    experiment_data = [[i + 1, rewards] for i, rewards in enumerate(experiment_rewards)]
    df = pd.DataFrame(experiment_data, columns=["Experiment Number", "Rewards"])
    df.to_csv(filename, index=False)


def save_trajectory(trajectory, filename="base_trajectories.txt"):
    """
    Save trajectory to a file.
    """
    with open(filename, "a") as file:
        file.write(str(trajectory) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--episode_length", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()

    agent = BaseAgent(goal=(3, 18), epsilon=0)
    agent.load_qtable('qtable_base_trained.json')

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    experiment_rewards = []
    num_episodes = 100  # Generate 1000 trajectories
    trajs_added = 0
    i = 0
    while i < num_episodes:
        i += 1
        global min_distance
        min_distance = 1000

        sock_game.send(str.encode("0 RESET"))
        state = recv_socket_data(sock_game)
        state = json.loads(state)

        trajectory = []
        cnt = 0
        cur_ep_return = 0
        goal_reached = 0
        reach_count = 0

        while not goal_reached:
            cnt += 1
            action_index = agent.choose_action(state)

            for _ in range(6):
                action = "0 " + agent.action_space[action_index]
                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
                next_state = json.loads(next_state)

            if next_state is None:
                break

            goal_reached, reward = calculate_reward(state, next_state, agent.goal, agent=agent)
            cur_ep_return += reward
            state = next_state

            if cnt >= args.episode_length or goal_reached or next_state["gameOver"]:
                break

        if goal_reached:
            reach_count += 1
            #save_trajectory(trajectory)  # Save the episode's trajectory

        logging.info(f"Episode {i} Goal Reached {goal_reached}")
    
    logging.info(f'Reach rate: {reach_count / num_episodes}')

    sock_game.close()


if __name__ == "__main__":
    main()
