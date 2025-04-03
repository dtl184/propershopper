import argparse
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
from ltl_agent import LTLAgent
from env_files.utils import recv_socket_data
import os
import logging

logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

def load_trajectories(file_name):
    """
    Load trajectories from a text file.
    """
    trajectories = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                trajectory = eval(line)
                trajectories.append(trajectory)
    return trajectories

def calculate_reward(state, next_state=None, goal=None, agent=None):
    """
    Calculate reward based on distance to the goal.
    """
    reward = -1
    reached = False


    # curr_position = state['observation']['players'][0]['position']
    # next_position = next_state['observation']['players'][0]['position']

    # curr_distance = ((round(curr_position[0]) - goal[0]) ** 2 + (round(curr_position[1]) - goal[1]) ** 2) ** 0.5
    # next_distance = ((round(next_position[0]) - goal[0]) ** 2 + (round(next_position[1]) - goal[1]) ** 2) ** 0.5

    if agent.state_index(state) in [306, 307, 325, 326]:
        reward = 1000
        reached = True

    # if curr_distance < 1.2:  
    #     reward = 1000
    #     reached = True
    #     print("Goal reached!")



    return reached, reward


def save_goal_reach_data(results, filename="ltl_goal_reach_data.txt"):
    """
    Save goal reach data to a text file.
    Each row represents an experiment, and each column represents whether the goal was reached in that episode (1 = Yes, 0 = No).
    """
    np.savetxt(filename, results, fmt="%d")


def plot_goal_reach_rate(results=None, filename="ltl_goal_reach_data.txt"):
    """
    Plot the average proportion of episodes where the goal was reached across experiments.
    """
    if results is None:
        try:
            results = np.loadtxt(filename, dtype=int)
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return

    avg_goal_reach = np.mean(results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_goal_reach, label="Goal Reach Rate per Episode", linewidth=2, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Proportion of Episodes Reaching Goal")
    plt.ylim([0, None])
    plt.title("Proportion of Episodes Where Goal Was Reached Across Experiments")
    plt.legend()
    plt.grid()
    plt.savefig('ltl_goal_reached_graph.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--num_experiments", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--num_episodes", type=int, default=1500, help="Number of episodes per experiment")
    parser.add_argument("--episode_length", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()

    trajectories = load_trajectories('base_trajectories.txt')

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    all_experiment_results = np.zeros((args.num_experiments, args.num_episodes), dtype=int)
    experiment_durations = []

    for experiment in range(args.num_experiments):
        logging.info(f"\nStarting Experiment {experiment}")

        agent = LTLAgent(n_states=437, goal=(3,18), filename='base_trajectories.txt')
        agent.learn_from_trajectories()

        for episode in range(args.num_episodes):
            sock_game.send(str.encode("0 RESET"))  # Reset environment
            state = recv_socket_data(sock_game)
            state = json.loads(state) if state else None

            if state is None:
                all_experiment_results[experiment, episode] = 0
                continue

            cnt = 0
            goal_reached = 0

            while not goal_reached:
                cnt += 1
                action_index = agent.choose_action(state)

                for _ in range(6):  # Take repeated actions
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                    next_state = json.loads(next_state) if next_state else None

                if next_state is None:
                    break

                goal_reached, reward = calculate_reward(state, next_state, agent.goal, agent=agent)

                agent.learning(action_index, state, next_state, reward)

                if cnt >= args.episode_length or next_state["gameOver"]:
                    break

                
                state = next_state

            all_experiment_results[experiment, episode] = goal_reached
            logging.info(f"Experiment {experiment}, Episode {episode}, Goal Reached: {goal_reached}")

    sock_game.close()

    # Save results
    save_goal_reach_data(all_experiment_results)
    plot_goal_reach_rate(all_experiment_results)


if __name__ == "__main__":
    main()
