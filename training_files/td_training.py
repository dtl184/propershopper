import argparse
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from td_agent import TDAgent
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

def calculate_reward(state, next_state=None, goal=None):
    """
    Calculate reward based on distance to the goal.
    """
    reward = -1
    reached = False

    if state in [306, 307, 325, 326]:
        reward = 1000
        reached = True


    return reached, reward


def save_goal_reach_data(results, filename="50step_goal_reach_data.txt"):
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



def plot_goal_reach_rate(results=None, filename="q_goal_reach_data.txt"):
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
    plt.savefig('td_goal_reached_graph.png')
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
        logging.info(f"\nStarting Experiment {experiment}")

        agent = TDAgent(type='nstep', n=1) 

        for episode in range(args.num_episodes):
            sock_game.send(str.encode("0 RESET"))  # Reset environment
            state = recv_socket_data(sock_game)
            state = json.loads(state) if state else None

            if state is None:
                all_experiment_results[experiment, episode] = 0
                continue

            agent.trajectory = []
            cnt = 0
            goal_reached = 0  # 1 if goal reached, 0 otherwise
            action_index = 0

            if agent.type == 'sarsa':
                action_index = agent.choose_action(state)

            while not goal_reached:

                if agent.type == 'Q' or agent.type == 'nstep':
                    action_index = agent.choose_action(state)

                for _ in range(6):  
                    action = "0 " + agent.action_space[action_index]
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                    next_state = json.loads(next_state) if next_state else None

                if next_state is None:
                    break

                goal_reached, reward = calculate_reward(agent.state_index(state), next_state, agent.goal)

                if agent.type == 'nstep':
                    if goal_reached:
                        T = cnt + 1
                    else:
                        T = 10000000
                    agent.trajectory.append((agent.state_index(state), action_index, reward))
                    agent.nstep_learning(action_index, state, next_state, cnt, T)
                else:
                    action_index = agent.learning(action_index, state, next_state, reward)

                if cnt >= args.episode_length:
                    break

                state = next_state
                cnt += 1

            all_experiment_results[experiment, episode] = goal_reached
            logging.info(f"Experiment {experiment}, Episode {episode}, Goal Reached: {goal_reached}")

        # Record experiment duration
    
    #agent.save_qtable()

    sock_game.close()

    # Save results
    
    save_goal_reach_data(all_experiment_results)
    plot_goal_reach_rate(all_experiment_results)



if __name__ == "__main__":
    main()
