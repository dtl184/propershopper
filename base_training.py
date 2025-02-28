import argparse
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from base_agent import BaseAgent  # ✅ Replaced LTLAgent with BaseAgent
from utils import recv_socket_data
import os

def calculate_reward(state, next_state=None, goal=None):
    """
    Calculate reward based on distance to the goal.
    """
    reward = -1
    reached = False

    if goal is not None:
        # curr_position = state['observation']['players'][0]['position']
        # next_position = next_state['observation']['players'][0]['position']

        # curr_distance = ((round(curr_position[0]) - goal[0]) ** 2 + (round(curr_position[1]) - goal[1]) ** 2) ** 0.5
        # next_distance = ((round(next_position[0]) - goal[0]) ** 2 + (round(next_position[1]) - goal[1]) ** 2) ** 0.5

        if state in [306, 307, 325, 326]:
            reward = 1000
            reached = True
            print('Goal Reached!')

        # if curr_distance < 1.2:  
        #     reward = 1000
        #     reached = True
        #     print("Goal reached!")



    return reached, reward - 0.05


def save_goal_reach_data(results, filename="new_goal_reach_data.txt"):
    """
    Append goal reach data to a text file.
    Each row represents an experiment, and each column represents whether the goal was reached in that episode (1 = Yes, 0 = No).
    """
    # Check if file exists
    if os.path.exists(filename):
        # Load existing data
        existing_data = np.loadtxt(filename, dtype=int)
        
        # Ensure existing_data is 2D (handles case where there's only one experiment saved)
        if existing_data.ndim == 1:
            existing_data = existing_data.reshape(1, -1)

        # Stack new results with existing data
        updated_data = np.vstack([existing_data, results])
    else:
        updated_data = np.array(results)  # No previous data, just save current results

    # Save the combined data back to the file
    np.savetxt(filename, updated_data, fmt="%d")



def plot_goal_reach_rate(results=None, filename="new_goal_reach_data.txt"):
    """
    Plot the average proportion of episodes where the goal was reached across experiments.
    If results is provided, it will be plotted. Otherwise, it loads data from filename.
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
    plt.savefig('new_goal_reached_graph.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="Port to connect to the environment")
    parser.add_argument("--num_experiments", type=int, default=100, help="Number of experiments to run")
    parser.add_argument("--num_episodes", type=int, default=2500, help="Number of episodes per experiment")
    parser.add_argument("--episode_length", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = args.port
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    all_experiment_results = np.zeros((args.num_experiments, args.num_episodes), dtype=int)  # Store all results
    experiment_durations = []  # Store time taken per experiment

    for experiment in range(args.num_experiments):
        print(f"\nStarting Experiment {experiment + 1}/{args.num_experiments}")

        start_time = time.time()  # Start timer for the experiment

        agent = BaseAgent(goal=(3, 18))  # ✅ Replaced LTLAgent with BaseAgent

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
                goal_reached, reward = calculate_reward(agent.state_index(state), next_state, agent.goal)
                if goal_reached:
                    goal_reached_flag = 1
                    break  # Stop early if goal is reached

                agent.learning(action_index, state, next_state, reward)

                if cnt >= args.episode_length or next_state["gameOver"]:
                    break

                state = next_state

            all_experiment_results[experiment, episode] = goal_reached_flag
            print(f"Experiment {experiment + 1}, Episode {episode + 1}, Goal Reached: {goal_reached_flag}")

        # Record experiment duration
        experiment_time = time.time() - start_time
        experiment_durations.append(experiment_time)
        print(f"Experiment {experiment + 1} completed in {experiment_time:.2f} seconds")
    
    #agent.save_qtable()

    sock_game.close()

    # Save results
    
    #save_goal_reach_data(all_experiment_results)
    #plot_goal_reach_rate(all_experiment_results)

    # Compute and print average time per experiment
    avg_experiment_time = np.mean(experiment_durations)
    print(f"\nAverage training time per experiment: {avg_experiment_time:.2f} seconds")
    print(f"Total training time: {sum(experiment_durations):.2f} seconds")


if __name__ == "__main__":
    main()
