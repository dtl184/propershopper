import argparse
import socket
import json
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('~/daniel_training/propershopper/')
from base_agent import BaseAgent
from utils import recv_socket_data

min_distance = float('inf')


def calculate_reward(state, next_state, goal=None):
    """
    Calculate reward based on distance to the goal.
    """
    global min_distance  
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
        
        if curr_distance < 1.2:  
            reward = 1000
            reached = True
            print("Goal reached!")

    return reached, reward - 0.05


def save_goal_reach_data(results, filename="goal_reach_data.txt"):
    """
    Save goal reach data to a text file.
    Each row represents an experiment, and each column represents whether the goal was reached in that episode (1 = Yes, 0 = No).
    """
    np.savetxt(filename, results, fmt="%d")


def plot_goal_reach_rate(results):
    """
    Plot the average proportion of episodes where the goal was reached across experiments.
    """
    avg_goal_reach = np.mean(results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_goal_reach, label="Goal Reach Rate per Episode", linewidth=2, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Proportion of Episodes Reaching Goal")
    plt.title("Proportion of Episodes Where Goal Was Reached Across Experiments")
    plt.legend()
    plt.grid()
    plt.savefig('goal_reach_rate.png')
    plt.savefig('goal_reached_graph.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--num_experiments", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--num_episodes", type=int, default=1500, help="Number of episodes per experiment")
    parser.add_argument("--episode_length", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    all_experiment_results = np.zeros((args.num_experiments, args.num_episodes), dtype=int)  # Store all results

    for experiment in range(args.num_experiments):
        print(f"\nStarting Experiment {experiment + 1}/{args.num_experiments}")

        agent = BaseAgent(goal=(3, 18))
        #agent.qtable = pd.read_json('qtable_base.json')  # Reset Q-table for each experiment

        for episode in range(args.num_episodes):
            sock_game.send(str.encode("0 RESET"))  # Reset environment
            state = recv_socket_data(sock_game)
            state = json.loads(state) if state else None

            if state is None:
                all_experiment_results[experiment, episode] = 0
                continue

            cnt = 0
            goal_reached_flag = 0  # 1 if goal reached, 0 otherwise

            while not state['gameOver']:
                cnt += 1
                action_index = agent.choose_action(state)

                for _ in range(6):  # Take repeated actions
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                    next_state = json.loads(next_state) if next_state else None

                if next_state is None:
                    break

                # Check if goal was reached
                goal_reached, _ = calculate_reward(state, next_state, agent.goal)
                if goal_reached:
                    goal_reached_flag = 1
                    break  # Stop early if goal is reached

                if cnt >= args.episode_length or next_state["gameOver"]:
                    break

                state = next_state

            all_experiment_results[experiment, episode] = goal_reached_flag
            print(f"Experiment {experiment + 1}, Episode {episode + 1}, Goal Reached: {goal_reached_flag}")

    sock_game.close()

    # Save and plot results
    save_goal_reach_data(all_experiment_results)
    plot_goal_reach_rate(all_experiment_results)


if __name__ == "__main__":
    main()
